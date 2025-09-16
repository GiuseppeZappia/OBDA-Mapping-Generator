import ollama
import re
import uuid
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set, Union, Optional
import logging
from collections import defaultdict, deque
import os
from pathlib import Path
from rdflib import Graph, RDF, RDFS, OWL, XSD


MAIN_IP="http://172.21.212.174:11434"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OBDAMappingGenerator:
    def __init__(self, ollama_host: str = MAIN_IP, model_name: str = "deepseek-r1:70b"):
        self.client = ollama.Client(host=ollama_host)
        self.model_name = model_name
        
        # Storage for generated mappings and context
        self.core_mappings = []
        self.processed_tables = set()
        self.schema_info = {}
        self.ontology_content = ""
        self.ontology_classes = []
        self.ontology_object_properties = []
        self.ontology_data_properties = []
        self.used_prefixes = set()
        
        # Property type mappings
        self.property_ranges = {}  # property -> range type (object/data)
        self.class_domains = {}    # property -> domain class
        
        # CSV-specific attributes
        self.csv_data = None
        self.inferred_schema = {}
        
    def detect_file_type(self, data_file: str) -> str:
        """
        Automatically detect if the input file is SQL schema or CSV data
        """
        file_extension = Path(data_file).suffix.lower()
        
        if file_extension == '.csv':
            return 'csv'
        elif file_extension in ['.sql', '.ddl']:
            return 'sql'
        else:
            # Try to detect based on content
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    content = f.read(1000).upper()  # Read first 1000 chars
                    
                if any(keyword in content for keyword in ['CREATE TABLE', 'ALTER TABLE', 'DROP TABLE']):
                    return 'sql'
                elif ',' in content and '\n' in content:
                    # Likely CSV if contains commas and newlines
                    return 'csv'
                else:
                    logger.warning(f"Cannot determine file type for {data_file}, defaulting to SQL")
                    return 'sql'
            except Exception as e:
                logger.error(f"Error reading file {data_file}: {e}")
                return 'sql'
    
    def infer_column_type_from_csv(self, series: pd.Series) -> str:
        """
        Infer SQL column type from pandas Series data
        """
        # Remove null values for analysis
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return 'TEXT'
        
        # Check if all values are numeric
        if pd.api.types.is_numeric_dtype(non_null_series):
            if pd.api.types.is_integer_dtype(non_null_series):
                max_val = non_null_series.max()
                min_val = non_null_series.min()
                if min_val >= -2147483648 and max_val <= 2147483647:
                    return 'INTEGER'
                else:
                    return 'BIGINT'
            else:
                return 'DECIMAL'
        
        # Check if it's datetime
        try:
            pd.to_datetime(non_null_series.iloc[0])
            if any(keyword in str(non_null_series.iloc[0]).lower() for keyword in ['date', 'time', ':']):
                return 'TIMESTAMP'
        except:
            pass
        
        # Check if it's boolean
        unique_values = set(non_null_series.astype(str).str.lower().unique())
        if unique_values.issubset({'true', 'false', '1', '0', 'yes', 'no', 't', 'f'}):
            return 'BOOLEAN'
        
        # Default to VARCHAR with estimated length
        max_length = non_null_series.astype(str).str.len().max()
        if max_length <= 50:
            return f'VARCHAR({min(255, max(50, max_length + 10))})'
        elif max_length <= 255:
            return f'VARCHAR({max_length + 20})'
        else:
            return 'TEXT'
    
    def detect_foreign_keys_from_csv(self, df: pd.DataFrame, column_name: str) -> List[Dict]:
        """
        Detect potential foreign key relationships from CSV data
        """
        foreign_keys = []
        
        # Check if column name suggests it's a foreign key
        fk_patterns = [
            r'(.+)[-_]id$',  # matches "animal-id", "tag_id", etc.
            r'(.+)[-_]key$', # matches "animal-key", "tag_key", etc.
            r'(.+)[-_]ref$', # matches "animal-ref", etc.
        ]
        
        for pattern in fk_patterns:
            match = re.match(pattern, column_name.lower().replace('-', '_'))
            if match:
                referenced_entity = match.group(1)
                
                # Check if there might be a referenced table
                # This is a heuristic - in real scenarios you might have multiple CSVs
                # or need to define relationships explicitly
                foreign_keys.append({
                    'column': column_name,
                    'referenced_table': f"{referenced_entity}s",  # pluralize
                    'referenced_column': column_name,
                    'confidence': 'medium'  # indicate this is inferred
                })
                break
        
        return foreign_keys
    
    def detect_primary_key_from_csv(self, df: pd.DataFrame) -> List[str]:
        """
        Detect primary key candidates from CSV data
        """
        primary_key_candidates = []
        
        for column in df.columns:  
            col_lower = column.lower().replace('-', '_')
            
            # Check for obvious ID columns
            if any(pattern in col_lower for pattern in ['id', 'key', 'identifier']):
                # Check uniqueness
                if df[column].nunique() == len(df.dropna(subset=[column])):
                    primary_key_candidates.append(column)
        
        # If no obvious ID column, look for unique columns
        if not primary_key_candidates:
            for column in df.columns:
                if df[column].nunique() == len(df.dropna(subset=[column])):
                    primary_key_candidates.append(column)
        
        # Return the first candidate or all columns if none found
        if primary_key_candidates:
            return [primary_key_candidates[0]]  # Use first unique column
        else:
            # Fallback: use first column
            return [df.columns[0]] if len(df.columns) > 0 else []
    
    def parse_csv_to_schema(self, csv_file: str) -> Dict[str, Dict]:
        """
        Parse CSV file and infer schema information
        """
        logger.info(f"Parsing CSV file: {csv_file}")
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            self.csv_data = df
            
            # Infer table name from filename
            table_name = Path(csv_file).stem.lower().replace(' ', '_').replace('-', '_')
            
            logger.info(f"Inferred table name: {table_name}")
            logger.info(f"CSV columns: {list(df.columns)}")
            logger.info(f"CSV shape: {df.shape}")
            
            # Analyze each column
            columns = []
            all_foreign_keys = []
            
            for column in df.columns:
                # Clean column name
                clean_column = column.replace('-', '_').replace(' ', '_').lower()
                
                # Infer column type
                col_type = self.infer_column_type_from_csv(df[column])
                
                # Check for null values to determine if NOT NULL
                has_nulls = df[column].isnull().any()
                properties = "" if has_nulls else "NOT NULL"
                
                columns.append({
                    "name": clean_column,
                    "original_name": column,
                    "type": col_type,
                    "properties": properties
                })
                
                # Detect potential foreign keys
                fks = self.detect_foreign_keys_from_csv(df, clean_column)
                all_foreign_keys.extend(fks)
            
            # Detect primary key
            original_pk = self.detect_primary_key_from_csv(df)
            primary_keys = [col.replace('-', '_').replace(' ', '_').lower() for col in original_pk]
            
            logger.info(f"Detected primary key(s): {primary_keys}")
            logger.info(f"Detected {len(all_foreign_keys)} potential foreign key(s)")
            
            # Create schema structure
            schema = {
                table_name: {
                    "columns": columns,
                    "constraints": [],
                    "foreign_keys": all_foreign_keys,
                    "primary_keys": primary_keys,
                    "inferred_from_csv": True,
                    "csv_file": csv_file,
                    "row_count": len(df)
                }
            }
            print(schema)
            return schema
            
        except Exception as e:
            logger.error(f"Error parsing CSV file {csv_file}: {e}")
            raise
    


    def parse_ontology_structure(self, ontology_file: str):
        """
        Extract classes, properties, and other info from ontology (TTL or RDF/XML)
        """
        g = Graph()
        g.parse(ontology_file)  # autodetect format (ttl, rdf/xml, n3, ecc.)

        classes = [str(s).split("#")[-1] for s in g.subjects(RDF.type, OWL.Class)]

        object_properties = [str(s).split("#")[-1] for s in g.subjects(RDF.type, OWL.ObjectProperty)]

        data_properties = [str(s).split("#")[-1] for s in g.subjects(RDF.type, OWL.DatatypeProperty)]

        annotation_properties = [str(s).split("#")[-1] for s in g.subjects(RDF.type, OWL.AnnotationProperty)]

        individuals = [str(s).split("#")[-1] for s in g.subjects(RDF.type, OWL.NamedIndividual)]

        subclass_relations = [(str(s).split("#")[-1], str(o).split("#")[-1]) for s, o in g.subject_objects(RDFS.subClassOf)]
        subproperty_relations = [(str(s).split("#")[-1], str(o).split("#")[-1]) for s, o in g.subject_objects(RDFS.subPropertyOf)]

        property_ranges = {}
        class_domains = {}

        for s, o in g.subject_objects(RDFS.range):
            prop = str(s).split("#")[-1]
            o_str = str(o)

            if o_str.startswith(str(XSD)):
                property_ranges[prop] = "data"
            elif o_str.startswith(str(OWL)) or o_str.startswith(str(RDFS)):
                property_ranges[prop] = "object"
            else:
                property_ranges[prop] = "unknown"

        for s, o in g.subject_objects(RDFS.domain):
            prop = str(s).split("#")[-1]
            domain_class = str(o).split("#")[-1]
            class_domains[prop] = domain_class


        self.ontology_classes = list(set(classes))
        self.ontology_object_properties = list(set(object_properties))
        self.ontology_data_properties = list(set(data_properties))
        self.property_ranges = property_ranges
        self.class_domains = class_domains

        logger.info(f"Extracted {len(self.ontology_classes)} classes, "
                    f"{len(self.ontology_object_properties)} object properties, "
                    f"{len(self.ontology_data_properties)} data properties, "
                    f"{len(individuals)} individuals, "
                    f"{len(annotation_properties)} annotation properties")

        logger.info(f"Classes: {self.ontology_classes}")
        logger.info(f"Object Properties: {self.ontology_object_properties}")
        logger.info(f"Data Properties: {self.ontology_data_properties}")
        logger.info(f"Annotation Properties: {annotation_properties}")
        logger.info(f"Individuals: {individuals}")
        logger.info(f"Subclass relations: {subclass_relations}")
        logger.info(f"Subproperty relations: {subproperty_relations}")

    
    def parse_sql_schema(self, sql_text: str) -> Dict[str, Dict]:
        """
        Parse SQL schema to extract table information with enhanced foreign key and primary key detection
        """
        # Remove single line comments "-- ..."
        sql_text = re.sub(r'--.*', '', sql_text)
        # Find all CREATE TABLE ... ( ... );
        table_blocks = re.findall(
            r'CREATE TABLE\s+([\w\."]+)\s*\((.*?)\);',
            sql_text,
            flags=re.DOTALL | re.IGNORECASE
        )
        schema = {}
        for table_name, body in table_blocks:
            columns = []
            constraints = []
            foreign_keys = []
            primary_keys = []
            
            # Split by line cleaning spaces
            for line in body.strip().split("\n"):
                line = line.strip().rstrip(",")  # removing final commas
                if not line:
                    continue
                    
                # Check for foreign key constraints
                if 'FOREIGN KEY' in line.upper():
                    # Extract referenced table and columns
                    fk_match = re.search(r'FOREIGN KEY\s*\(\s*([^)]+)\s*\)\s*REFERENCES\s+([\w\.]+)\s*\(\s*([^)]+)\s*\)', line, re.IGNORECASE)
                    if fk_match:
                        fk_column = fk_match.group(1).strip().strip('"')
                        referenced_table = fk_match.group(2).strip().strip('"')
                        referenced_column = fk_match.group(3).strip().strip('"')
                        foreign_keys.append({
                            'column': fk_column,
                            'referenced_table': referenced_table,
                            'referenced_column': referenced_column
                        })
                        
                # Check for primary key constraints
                if line.upper().startswith("PRIMARY KEY"):
                    pk_match = re.search(r'PRIMARY KEY\s*\(\s*([^)]+)\s*\)', line, re.IGNORECASE)
                    if pk_match:
                        pk_columns = [col.strip().strip('"') for col in pk_match.group(1).split(',')]
                        primary_keys.extend(pk_columns)
                        
                # If it is a constraint line
                if line.upper().startswith("CONSTRAINT") or line.upper().startswith("PRIMARY KEY") or line.upper().startswith("FOREIGN KEY"):
                    constraints.append(line)
                else:
                    # Column parse: name, type, properties
                    match = re.match(r'"?([\w]+)"?\s+([\w\(\)]+)(.*)', line)
                    if match:
                        col_name, col_type, rest = match.groups()
                        col_properties = rest.strip()
                        
                        # Check if this column is a primary key
                        if 'PRIMARY KEY' in col_properties.upper():
                            primary_keys.append(col_name)
                        
                        # Check if this column is a foreign key (inline definition)
                        fk_inline = re.search(r'REFERENCES\s+([\w\.]+)\s*\(\s*([^)]+)\s*\)', col_properties, re.IGNORECASE)
                        if fk_inline:
                            referenced_table = fk_inline.group(1).strip().strip('"')
                            referenced_column = fk_inline.group(2).strip().strip('"')
                            foreign_keys.append({
                                'column': col_name,
                                'referenced_table': referenced_table,
                                'referenced_column': referenced_column
                            })
                        
                        columns.append({
                            "name": col_name,
                            "type": col_type,
                            "properties": col_properties
                        })
                        
            schema[table_name] = {
                "columns": columns,
                "constraints": constraints,
                "foreign_keys": foreign_keys,
                "primary_keys": primary_keys if primary_keys else [columns[0]['name']] if columns else [],
                "inferred_from_csv": False
            }
        
        return schema
    
    def determine_processing_order(self, tables: Dict[str, Dict]) -> List[str]:
        """
        Determine optimal table processing order using topological sort
        based on foreign key dependencies and table complexity
        """
        logger.info("Determining optimal table processing order...")
        
        # Build dependency graph based on foreign keys
        dependencies = defaultdict(set)
        dependents = defaultdict(set)
        
        for table_name, table_info in tables.items():
            for fk_info in table_info.get('foreign_keys', []):
                referenced_table = fk_info['referenced_table']
                if referenced_table in tables:
                    dependencies[table_name].add(referenced_table)
                    dependents[referenced_table].add(table_name)
        
        # Calculate table complexity scores
        complexity_scores = {}
        for table_name, table_info in tables.items():
            score = 0
            # Base score from number of columns
            score += len(table_info['columns'])
            # Penalty for foreign keys (more complex dependencies)
            score += len(table_info.get('foreign_keys', [])) * 2
            # Bonus for being referenced by others (important base tables)
            score -= len(dependents[table_name]) * 3
            # Penalty for composite primary keys
            score += len(table_info.get('primary_keys', [])) * 1.5
            complexity_scores[table_name] = score
        
        # Topological sort with complexity consideration
        processing_order = []
        in_degree = {table: len(dependencies[table]) for table in tables}
        queue = deque()
        
        # Start with tables that have no dependencies
        for table in tables:
            if in_degree[table] == 0:
                queue.append(table)
        
        # Sort initial queue by complexity (simpler first)
        queue = deque(sorted(queue, key=lambda x: complexity_scores[x]))
        
        while queue:
            current_table = queue.popleft()
            processing_order.append(current_table)
            
            # Update in-degrees of dependent tables
            next_candidates = []
            for dependent in dependents[current_table]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    next_candidates.append(dependent)
            
            # Sort candidates by complexity and add to queue
            next_candidates.sort(key=lambda x: complexity_scores[x])
            queue.extend(next_candidates)
        
        # Handle circular dependencies if any remain
        remaining_tables = [table for table in tables if table not in processing_order]
        if remaining_tables:
            logger.warning(f"Circular dependencies detected for tables: {remaining_tables}")
            # Sort remaining by complexity and add
            remaining_tables.sort(key=lambda x: complexity_scores[x])
            processing_order.extend(remaining_tables)
    
        return processing_order

    def extract_used_concepts_from_mapping(self, mapping_text: str):
        """
        Extract ontology concepts and prefixes used in a mapping
        """
        # Extract prefixes used (look for namespace:concept patterns)
        prefix_matches = re.findall(r'(\w+):', mapping_text)
        self.used_prefixes.update(prefix_matches)
    
    def generate_composite_key_template(self, primary_keys: List[str]) -> str:
        """
        Generate template for composite primary keys
        """
        if len(primary_keys) == 1:
            return f"{{{primary_keys[0]}}}"
        else:
            # Create composite key template
            key_parts = [f"{{{key}}}" for key in primary_keys]
            return "_".join(key_parts)
    
    def identify_foreign_key_properties(self, table_info: Dict) -> Dict[str, Dict]:
        """
        Identify which columns are foreign keys and should be mapped as object properties
        """
        fk_mappings = {}
        for fk_info in table_info.get('foreign_keys', []):
            column = fk_info['column']
            referenced_table = fk_info['referenced_table']
            referenced_column = fk_info['referenced_column']
            
            fk_mappings[column] = {
                'referenced_table': referenced_table,
                'referenced_column': referenced_column,
                'is_foreign_key': True
            }
        
        return fk_mappings
    
    def determine_property_type(self, column_name: str, column_type: str, fk_mappings: Dict) -> str:
        """
        Determine if a property should be object or data property
        """
        # Check if it's a foreign key
        if column_name in fk_mappings:
            return 'object'
        
        # Check if column name suggests it's a reference
        if column_name.endswith('_id') or column_name.endswith('ID'):
            # Could be a foreign key not properly declared
            return 'object'
        
        # Default to data property
        return 'data'
    
    def validate_columns_exist(self, mapping_text: str, table_columns: List[str]) -> str:
        """
        Validate that all columns referenced in the mapping actually exist in the table.
        Removes entire mapping blocks (mappingId + target + source) if any column is invalid.
        """
        if not mapping_text.strip():
            return mapping_text
        
        lines = mapping_text.split('\n')
        valid_blocks = []
        current_block = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            #If it is an empty line and we have a current block, we process it
            if not line:
                if current_block:
                    # Validate current block
                    if self._is_mapping_block_valid(current_block, table_columns):
                        valid_blocks.extend(current_block)
                        valid_blocks.append('')  # Add empty line between blocks
                    else:
                        logger.warning(f"Removing invalid mapping block: {current_block[0] if current_block else 'Unknown'}")
                    current_block = []
                else:
                    #Empty line without a current block, keeping it
                    valid_blocks.append('')
                i += 1
                continue
            
            # If a new mapping starts
            if line.startswith('mappingId'):
                # If we have a previous block we validate it
                if current_block:
                    if self._is_mapping_block_valid(current_block, table_columns):
                        valid_blocks.extend(current_block)
                        valid_blocks.append('')
                    else:
                        logger.warning(f"Removing invalid mapping block: {current_block[0] if current_block else 'Unknown'}")
                
                # Start a new block
                current_block = [line]
                
                # Find the target and source lines that follow
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line:
                        break
                    if next_line.startswith(('target', 'source')):
                        current_block.append(next_line)
                        j += 1
                    elif next_line.startswith('mappingId'):
                        # Found a new mappingId, we stop here
                        break
                    else:
                        # Unrecognized line in mapping, we skip it
                        j += 1
                
                i = j
                continue
            # If it is not the start of a mapping and we do not have a current block,
            # probably it is an isolated line to ignore
            if not current_block:
                i += 1
                continue
            
            i += 1
        
        # Process the last block if it exists
        if current_block:
            if self._is_mapping_block_valid(current_block, table_columns):
                valid_blocks.extend(current_block)
            else:
                logger.warning(f"Removing invalid final mapping block: {current_block[0] if current_block else 'Unknown'}")
        
        return '\n'.join(valid_blocks)

    def _is_mapping_block_valid(self, mapping_block: List[str], table_columns: List[str]) -> bool:
        """
        Validate an entire mapping block (mappingId + target + source).
        Returns True if every used columns exist in the table.
        """
        if not mapping_block:
            return False
        
        # Find source line in the block
        source_line = None
        for line in mapping_block:
            if line.strip().startswith('source'):
                source_line = line.strip()
                break
        
        if not source_line:
            logger.warning("No source line found in mapping block")
            return False
        
        # Extract cplumns from the SELECT statement
        columns_used = self._extract_columns_from_source(source_line)
        
        if not columns_used:
            logger.warning("No columns extracted from source line")
            return False
        
        # Check that all columns exists in the schema
        for col in columns_used:
            clean_col = self._clean_column_name(col)
            if clean_col and clean_col not in table_columns:
                logger.warning(f"Column '{clean_col}' not found in table columns: {table_columns}")
                return False
        
        return True

    def _extract_columns_from_source(self, source_line: str) -> List[str]:
        """
        Extracts the columns names from source statement of an OBDA mapping block.
        """
        # Find SELECT ... FROM  pattern
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', source_line, re.IGNORECASE)
        if not select_match:
            return []
        
        select_part = select_match.group(1).strip()
        
        # Manage SELECT * case
        if select_part.strip() == '*':
            logger.warning("SELECT * found - cannot validate specific columns")
            return []
        
        # Spliy by commas and clean
        columns = []
        for col in select_part.split(','):
            col = col.strip()
            if col:
                columns.append(col)
        
        return columns

    def _clean_column_name(self, column_name: str) -> str:
        """
        Cleans a column name by removing special characters, quotes, etc.
        """
        if not column_name:
            return ""
        
        # Removes blank spaces
        clean_name = column_name.strip()
        
        # Removes quotes, backticks ecc
        clean_name = re.sub(r'[{}"\'`\[\]]', '', clean_name)
        
        # Removes any alias (AS or space)
        clean_name = re.split(r'\s+(?:AS\s+)?', clean_name, flags=re.IGNORECASE)[0]
        
        # Removes table prefixes (table.column -> column)
        if '.' in clean_name:
            clean_name = clean_name.split('.')[-1]
        
        return clean_name.strip()
    
    def validate_mapping_syntax(self, mapping_text: str) -> str:
        """
        Validate and clean mapping syntax, removing invalid lines
        """
        lines = mapping_text.split('\n')
        valid_lines = []
        current_mapping = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_mapping:
                    valid_lines.extend(current_mapping)
                    valid_lines.append('')
                    current_mapping = []
                continue
            
            # Skip comments and explanatory text
            if (line.startswith('#') or 
                line.startswith('//') or 
                'CHANGES MADE:' in line.upper() or
                'Note:' in line or
                'These mappings' in line or
                'The mappings are' in line or
                line.startswith('```')):
                continue
            
            # Validate OBDA syntax
            if (line.startswith('mappingId') or 
                line.startswith('target') or 
                line.startswith('source')):
                current_mapping.append(line)
            elif current_mapping:
                # If we have started a mapping but encounter invalid line, skip it
                logger.warning(f"Skipping invalid line in mapping: {line}")
        
        # Add final mapping if exists
        if current_mapping:
            valid_lines.extend(current_mapping)
        
        return '\n'.join(valid_lines)

    def clean_mapping_response(self, response_text: str) -> str:
        """
        Clean the LLM response to extract only valid OBDA mapping content
        """
        # Remove markdown code blocks
        response_text = re.sub(r'```obda\s*\n', '', response_text)
        response_text = re.sub(r'```\s*$', '', response_text)
        response_text = re.sub(r'```.*?\n', '', response_text)
        
        # Validate syntax and remove invalid lines
        cleaned_mapping = self.validate_mapping_syntax(response_text)
        
        # Fix common syntax issues
        cleaned_mapping = self.fix_common_syntax_issues(cleaned_mapping)
        
        # Extract used concepts for later reference
        if cleaned_mapping:
            self.extract_used_concepts_from_mapping(cleaned_mapping)
        
        return cleaned_mapping
    
    def fix_common_syntax_issues(self, mapping_text: str) -> str:
        """
        Fix common OBDA syntax issues
        """
        lines = mapping_text.split('\n')
        fixed_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                fixed_lines.append('')
                continue
                
            # Fix target lines - ensure proper syntax
            if line.startswith('target'):
                # Ensure proper spacing and semicolon ending
                if not line.endswith('.') and not line.endswith(';'):
                    line = line + ' .'
                # Fix common target syntax issues
                line = re.sub(r'target\s*', 'target       ', line)
                
            # Fix source lines
            elif line.startswith('source'):
                line = re.sub(r'source\s*', 'source       ', line)
                
            # Fix mappingId lines
            elif line.startswith('mappingId'):
                line = re.sub(r'mappingId\s*', 'mappingId    ', line)
                
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)

    def create_clean_ontology_for_table(self, table_name: str) -> str:
        """
        Create a clean ontology excerpt focusing only on relevant classes and properties,
        removing any schema references or examples that might confuse the model
        """
        # Extract prefixes
        prefix_pattern = r'@prefix\s+[^.]*\.\s*\n'
        prefixes = re.findall(prefix_pattern, self.ontology_content)
        
        # Create clean ontology with only class and property definitions
        clean_ontology = ''.join(prefixes) + '\n'
        
        # Add only class and property definitions
        clean_ontology += "# Ontology Classes:\n"
        for cls in self.ontology_classes:
            clean_ontology += f":{cls} a owl:Class .\n"
        
        clean_ontology += "\n# Object Properties (use for foreign keys and references):\n"
        for prop in self.ontology_object_properties:
            clean_ontology += f":{prop} a owl:ObjectProperty .\n"
        
        clean_ontology += "\n# Data Properties (use for literal values):\n"
        for prop in self.ontology_data_properties:
            clean_ontology += f":{prop} a owl:DatatypeProperty .\n"
        
        # Add property ranges information
        clean_ontology += "\n# Property Type Information:\n"
        for prop, prop_type in self.property_ranges.items():
            if prop_type == 'object':
                clean_ontology += f"# :{prop} -> Object Property (expects URI/IRI as value)\n"
            else:
                clean_ontology += f"# :{prop} -> Data Property (expects literal value)\n"
        
        return clean_ontology

    def generate_mapping_prompt(self, table_name: str, table_info: Dict) -> str:
        """
        Generate enhanced prompt for mapping creation with foreign key and composite key handling
        """
        # Handle CSV-specific information
        is_csv_derived = table_info.get('inferred_from_csv', False)
        
        # Extract schema name from table_name if it contains a dot
        schema_name = "public"
        clean_table_name = table_name
        if "." in table_name:
            parts = table_name.split(".")
            schema_name = parts[0]
            clean_table_name = parts[1]

        # Get primary key columns (handle composite keys)
        primary_keys = table_info.get('primary_keys', [])
        if not primary_keys and table_info['columns']:
            # Fallback to first column or _id columns
            for col in table_info['columns']:
                if col['name'].endswith('_id') or 'PRIMARY KEY' in col.get('properties', '').upper():
                    primary_keys = [col['name']]
                    break
            if not primary_keys:
                primary_keys = [table_info['columns'][0]['name']]

        # Generate composite key template
        key_template = self.generate_composite_key_template(primary_keys)

        # Identify foreign keys
        fk_mappings = self.identify_foreign_key_properties(table_info)

        # Create explicit column list with type information
        available_columns = [col['name'] for col in table_info['columns']]
        column_details = []
        for col in table_info['columns']:
            col_detail = f"- {col['name']} ({col['type']})"
            if col.get('properties'):
                col_detail += f" {col['properties']}"
            
            # Add original CSV column name if different
            if is_csv_derived and col.get('original_name') != col['name']:
                col_detail += f" [CSV: {col['original_name']}]"
            
            # Add foreign key information
            if col['name'] in fk_mappings:
                fk_info = fk_mappings[col['name']]
                confidence = fk_info.get('confidence', 'high')
                col_detail += f" [FK -> {fk_info['referenced_table']}.{fk_info['referenced_column']} ({confidence})]"
            
            column_details.append(col_detail)

        # Use clean ontology without schema references
        clean_ontology = self.create_clean_ontology_for_table(table_name)

        # Create foreign key mapping instructions
        fk_instructions = ""
        if fk_mappings:
            fk_instructions = "\n\nFOREIGN KEY MAPPING RULES:\n"
            for col, fk_info in fk_mappings.items():
                referenced_table = fk_info['referenced_table'].split('.')[-1] if '.' in fk_info['referenced_table'] else fk_info['referenced_table']
                confidence = fk_info.get('confidence', 'high')
                fk_instructions += f"- Column '{col}' references {fk_info['referenced_table']}.{fk_info['referenced_column']} ({confidence} confidence)\n"
                fk_instructions += f"  -> Map as OBJECT PROPERTY linking to {referenced_table} entity\n"
                fk_instructions += f"  -> Target: :EntityClass/{{{key_template}}} :propertyName :ReferencedClass/{{{col}}} .\n"
                fk_instructions += f"  -> Source: SELECT {', '.join(primary_keys)}, {col} FROM {schema_name}.{clean_table_name} WHERE {col} IS NOT NULL\n"

        # Add CSV-specific information
        csv_context = ""
        if is_csv_derived:
            csv_context = f"\n\nCSV CONTEXT:\n"
            csv_context += f"- Data inferred from CSV file: {table_info.get('csv_file', 'unknown')}\n"
            csv_context += f"- Total rows in CSV: {table_info.get('row_count', 'unknown')}\n"
            csv_context += f"- Column names have been normalized (hyphens -> underscores, lowercase)\n"
            csv_context += f"- Foreign key relationships are inferred based on column naming patterns\n"

        prompt = f"""You are an OBDA mapping expert. Generate OBDA mappings for table '{clean_table_name}' with CORRECT handling of foreign keys and composite primary keys.

CRITICAL CONSTRAINT: You can ONLY use these exact columns from table {clean_table_name}:
{', '.join(available_columns)}

TABLE: {schema_name}.{clean_table_name}
PRIMARY KEY: {', '.join(primary_keys)} (composite key template: {key_template})

AVAILABLE COLUMNS (use ONLY these):
{chr(10).join(column_details)}

ONTOLOGY CLASSES AND PROPERTIES:
{clean_ontology}

{fk_instructions}

{csv_context}

STRICT REQUIREMENTS:

1. COMPOSITE PRIMARY KEYS:
   - Use complete composite key in URI template: :ClassName/{key_template}
   - For single PK: :ClassName/{{{primary_keys[0]}}}
   - For composite PK: :ClassName/{key_template}

2. FOREIGN KEY HANDLING:
   - Foreign key columns ({', '.join(fk_mappings.keys()) if fk_mappings else 'none'}) MUST be mapped as OBJECT PROPERTIES
   - Use URI references, NOT literals: :ClassName/{{{key_template}}} :objectProperty :ReferencedClass/{{{col}}} .
   - NEVER map foreign keys as data properties with literals

3. PROPERTY TYPE SELECTION:
   - Object Properties: Use for foreign keys and entity references
   - Data Properties: Use for literal values (text, numbers, dates)

4. MAPPING FORMAT:
mappingId    table_{clean_table_name}_to_class
target       :ClassName/{key_template} a :ClassName .
source       SELECT {', '.join(primary_keys)} FROM {schema_name}.{clean_table_name}

For Data Properties:
mappingId    table_{clean_table_name}_to_data_property_name
target       :ClassName/{key_template} :dataPropertyName "{{column_name}}"^^xsd:datatype .
source       SELECT {', '.join(primary_keys)}, column_name FROM {schema_name}.{clean_table_name} WHERE column_name IS NOT NULL

For Object Properties (Foreign Keys):
mappingId    table_{clean_table_name}_to_object_property_name
target       :ClassName/{key_template} :objectPropertyName :ReferencedClass/{{foreign_key_column}} .
source       SELECT {', '.join(primary_keys)}, foreign_key_column FROM {schema_name}.{clean_table_name} WHERE foreign_key_column IS NOT NULL

5. XSD DATATYPES:
   - xsd:string for text (char, varchar, text)
   - xsd:integer for integers
   - xsd:decimal for numeric/decimal
   - xsd:double for float/double
   - xsd:dateTime for timestamps/dates
   - xsd:boolean for boolean

6. VALIDATION RULES:
   - NEVER use columns not in the available list: {', '.join(available_columns)}
   - Always use complete primary key template for entity URIs
   - Use "/" separator in URIs
   - Include WHERE IS NOT NULL for nullable columns
   - Map foreign keys as object properties with URI values

CRITICAL: Foreign keys ({', '.join(fk_mappings.keys()) if fk_mappings else 'none'}) must be mapped as OBJECT PROPERTIES with URI values!

OUTPUT ONLY THE MAPPING RULES:"""
        
        return prompt
    
    def generate_delta_mapping(self, table_name: str, table_info: Dict) -> Dict:
        """
        Generate mapping for a single table (delta mapping) with enhanced validation
        """
        logger.info(f"Generating mapping for table: {table_name}")
        
        # Get available columns for validation
        available_columns = [col['name'] for col in table_info['columns']]
        
        # Generate prompt with focused ontology
        prompt = self.generate_mapping_prompt(table_name, table_info)
        
        # Generate mapping using LLM
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.0, "top_p": 0.9},
                stream=False
            )
            
            mapping_content = response['response'].strip()
            logger.info("LLM Response received")
            
            # Clean the response
            clean_mapping = self.clean_mapping_response(mapping_content)
            
            # Validate that only existing columns are used
            validated_mapping = self.validate_columns_exist(clean_mapping, available_columns)
            
            return {
                'table_name': table_name,
                'mapping_rule': validated_mapping,
                'description': f"Mapping for table {table_name}",
                'provenance': f"Generated from {'CSV file' if table_info.get('inferred_from_csv') else 'SQL schema'} {table_name}",
                'columns_mapped': available_columns,
                'primary_keys': table_info.get('primary_keys', []),
                'foreign_keys': table_info.get('foreign_keys', []),
                'inferred_from_csv': table_info.get('inferred_from_csv', False)
            }
            
        except Exception as e:
            logger.error(f"Error generating mapping for table {table_name}: {e}")
            return None
    
    def validate_and_refine_mapping(self, delta_mapping: Dict) -> Dict:
        """
        Validate and refine the generated delta mapping with enhanced foreign key validation
        """
        logger.info(f"Validating mapping for table: {delta_mapping['table_name']}")
        
        # Get available columns for this table
        table_info = self.schema_info[delta_mapping['table_name']]
        available_columns = [col['name'] for col in table_info['columns']]
        primary_keys = table_info.get('primary_keys', [])
        foreign_keys = table_info.get('foreign_keys', [])
        
        # Identify foreign key mappings
        fk_mappings = self.identify_foreign_key_properties(table_info)
        key_template = self.generate_composite_key_template(primary_keys)
        
        validation_prompt = f"""IF AND ONLY IF NEEDED Validate and fix this OBDA mapping with STRICT foreign key and composite key handling.

AVAILABLE COLUMNS FOR TABLE {delta_mapping['table_name']}:
{', '.join(available_columns)}

PRIMARY KEYS: {', '.join(primary_keys)} (Use template: {key_template})

FOREIGN KEYS:
{chr(10).join([f"- {fk['column']} -> {fk['referenced_table']}.{fk['referenced_column']}" + (f" ({fk.get('confidence', 'high')})" if 'confidence' in fk else "") for fk in foreign_keys])}

ONTOLOGY CLASSES: {', '.join(self.ontology_classes)}
OBJECT PROPERTIES: {', '.join(self.ontology_object_properties)}
DATA PROPERTIES: {', '.join(self.ontology_data_properties)}

CSV-DERIVED: {'Yes' if table_info.get('inferred_from_csv') else 'No'}

MAPPING TO VALIDATE:
{delta_mapping['mapping_rule']}

CRITICAL FIXES REQUIRED:

1. COMPOSITE KEY CORRECTION:
   - Entity URIs MUST use complete primary key template: :ClassName/{key_template}
   - NEVER use incomplete keys that cause collisions

2. FOREIGN KEY CORRECTION:
   - Columns {', '.join([fk['column'] for fk in foreign_keys])} are FOREIGN KEYS
   - Must be mapped as OBJECT PROPERTIES with URI values
   - Correct format: :ClassName/{key_template} :objectProperty :ReferencedClass/{{fk_column}} .
   - NEVER as literals: "{{fk_column}}"^^xsd:string

3. PROPERTY TYPE VALIDATION:
   - Use OBJECT PROPERTIES from: {', '.join(self.ontology_object_properties)}
   - Use DATA PROPERTIES from: {', '.join(self.ontology_data_properties)}

4. SYNTAX FIXES:
   - Remove markdown formatting
   - Fix OBDA syntax (mappingId, target, source format)
   - Target statements end with '.'
   - Valid URI patterns with '/' separator

5. COLUMN VALIDATION:
   - Only use columns: {', '.join(available_columns)}
   - Remove any non-existent columns

EXAMPLE CORRECTIONS:

WRONG (foreign key as literal):
target   :Entity/{key_template} :someProperty "{{fk_column}}"^^xsd:string .

CORRECT (foreign key as object):
target   :Entity/{key_template} :someProperty :ReferencedEntity/{{fk_column}} .

WRONG (incomplete composite key):
target   :Entity/{{single_id}} a :Entity .

CORRECT (complete composite key):
target   :Entity/{key_template} a :Entity .

OUTPUT ONLY THE CORRECTED MAPPING:"""
        
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=validation_prompt,
                options={"temperature": 0.0},
                stream=False
            )
            
            validated_mapping_text = self.clean_mapping_response(response['response'].strip())
            
            # Final validation of columns
            final_mapping = self.validate_columns_exist(validated_mapping_text, available_columns)
            
            if final_mapping and len(final_mapping) > 10:
                delta_mapping['mapping_rule'] = final_mapping
                logger.info(f"Mapping for {delta_mapping['table_name']} validated and refined")
            
            return delta_mapping
                
        except Exception as e:
            logger.error(f"Error validating mapping for {delta_mapping['table_name']}: {e}")
            return delta_mapping
    
    def integrate_mapping(self, delta_mapping: Dict):
        """
        Integrate delta mapping into core mappings
        """
        if delta_mapping and delta_mapping.get('mapping_rule'):
            self.core_mappings.append(delta_mapping)
            self.processed_tables.add(delta_mapping['table_name'])
            logger.info(f"Integrated mapping for table: {delta_mapping['table_name']}")
    
    def determine_required_prefixes(self) -> Dict[str, str]:
        """
        Determine which prefixes are actually needed based on used concepts and ontology
        """
        # Extract prefixes from ontology
        required_prefixes = {'': 'http://example.org/ontology#'}  # Default prefix
        
        # Extract prefixes from ontology content
        prefix_pattern = r'@prefix\s+(\w*):?\s*<([^>]+)>'
        prefixes = re.findall(prefix_pattern, self.ontology_content)
        for prefix, uri in prefixes:
            if prefix:
                required_prefixes[prefix] = uri
            else:
                required_prefixes[''] = uri  # Default prefix
        
        # Add standard prefixes if used
        standard_prefixes = {
            'owl': 'http://www.w3.org/2002/07/owl#',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
            'xsd': 'http://www.w3.org/2001/XMLSchema#'
        }
        
        for prefix in self.used_prefixes:
            if prefix in standard_prefixes:
                required_prefixes[prefix] = standard_prefixes[prefix]
        
        return required_prefixes
    
    def write_obda_file(self, output_file: str):
        """
        Write the generated mappings to an OBDA file with required prefixes
        """
        logger.info("Writing OBDA file...")
        
        # Determine required prefixes
        required_prefixes = self.determine_required_prefixes()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write OBDA file header with required prefixes
            f.write("[PrefixDeclaration]\n")
            for prefix, uri in required_prefixes.items():
                if prefix:
                    f.write(f"{prefix}:\t{uri}\n")
                else:
                    f.write(f":\t{uri}\n")
            f.write("\n")
            
            f.write("[MappingDeclaration] @collection [[\n\n")
            
            # Write all mappings with cleaned content
            for i, mapping in enumerate(self.core_mappings):
                # Ensure the mapping rule is clean
                clean_rule = self.validate_mapping_syntax(mapping['mapping_rule'])
                f.write(f"{clean_rule}\n")
                
                if i < len(self.core_mappings) - 1:
                    f.write("\n")
            
            f.write("\n]]\n")
        
        logger.info(f"OBDA file written with {len(required_prefixes)} prefixes and {len(self.core_mappings)} mappings")

    def generate_obda_mappings(self, data_file: str, ontology_file: str, output_file: str):
        """
        Main method to generate OBDA mappings with automatic file type detection and enhanced handling
        """
        try:
            logger.info("Starting OBDA mapping generation with automatic file type detection")
            
            # Detect file type and parse accordingly
            file_type = self.detect_file_type(data_file)
            logger.info(f"Detected file type: {file_type}")
            
            if file_type == 'csv':
                # Parse CSV file to infer schema
                self.schema_info = self.parse_csv_to_schema(data_file)
                logger.info(f"Inferred schema from CSV with {len(self.schema_info)} table(s)")
            else:
                # Parse SQL schema file
                with open(data_file, 'r', encoding='utf-8') as f:
                    schema_content = f.read()
                self.schema_info = self.parse_sql_schema(schema_content)
                logger.info(f"Parsed SQL schema with {len(self.schema_info)} table(s)")
            
            # Load and parse ontology
            with open(ontology_file, 'r', encoding='utf-8') as f:
                self.ontology_content = f.read()
            
            # Parse ontology structure to extract classes and properties
            self.parse_ontology_structure(ontology_file)


            logger.info(f"Loaded ontology with {len(self.ontology_classes)} classes, {len(self.ontology_object_properties)} object properties, and {len(self.ontology_data_properties)} data properties")
            
            # Log table and foreign key information
            for table_name, table_info in self.schema_info.items():
                logger.info(f"Table {table_name}:")
                logger.info(f"  - Columns: {[col['name'] for col in table_info['columns']]}")
                logger.info(f"  - Primary keys: {table_info.get('primary_keys', [])}")
                if table_info.get('foreign_keys'):
                    fk_info = [f"{fk['column']}->{fk['referenced_table']}.{fk['referenced_column']}" + (f"({fk.get('confidence', 'high')})" if 'confidence' in fk else "") for fk in table_info['foreign_keys']]
                    logger.info(f"  - Foreign keys: {fk_info}")
                if table_info.get('inferred_from_csv'):
                    logger.info(f"  - Inferred from CSV: {table_info.get('csv_file')}")
                    logger.info(f"  - Row count: {table_info.get('row_count', 'unknown')}")
            
            # Determine processing order using improved algorithm
            processing_order = self.determine_processing_order(self.schema_info)
            logger.info(f"Processing order: {processing_order}")
            
            # Iterative processing
            for table_name in processing_order:
                if table_name not in self.processed_tables:
                    table_info = self.schema_info[table_name]
                    
                    logger.info(f"Processing table {table_name}")
                    
                    # Generate delta mapping
                    delta_mapping = self.generate_delta_mapping(table_name, table_info)
                    
                    if delta_mapping:
                        # Validate and refine
                        refined_mapping = self.validate_and_refine_mapping(delta_mapping)
                        
                        # Integrate into core mappings
                        self.integrate_mapping(refined_mapping)
                    else:
                        logger.warning(f"Failed to generate mapping for table {table_name}")
            
            # Write OBDA file
            self.write_obda_file(output_file)
            logger.info(f"OBDA mapping generation completed. Output saved to: {output_file}")
            
            # Summary
            csv_tables = sum(1 for t in self.schema_info.values() if t.get('inferred_from_csv'))
            sql_tables = len(self.schema_info) - csv_tables
            logger.info(f"Summary: Generated mappings for {len(self.core_mappings)} tables ({csv_tables} from CSV, {sql_tables} from SQL)")
            return {
                'success': True,
                'message': f'OBDA mapping generation completed. Output saved to: {output_file}',
                'mappings_count': len(self.core_mappings),
                'tables_processed': list(self.processed_tables)
            }
        except Exception as e:
            logger.error(f"Error in generate_obda_mappings: {e}")
            return {
                'success': False,
                'error': str(e),
                'mappings_count': 0,
                'tables_processed': []
            }

# def main():
#     """
#     Enhanced main function that can handle both SQL schema files and CSV files
#     """
#     # Initialize the generator
#     generator = OBDAMappingGenerator(
#         ollama_host=MAIN_IP,
#         model_name="deepseek-r1:70b"
#     )
    
#     # Usage with CSV file (the system will auto-detect)

#     # generator.generate_obda_mappings(
#     #     data_file="ThermochronTracking Elephants Kruger 2007.csv", 
#     #     ontology_file="new_ontology.xml",
#     #     output_file="generated_mappings_from_csv.obda"
#     # )
    
#     # Example usage with SQL schema file
#     # generator.generate_obda_mappings(
#     #     data_file="schema-A.sql",  # Can be .csv or .sql
#     #     ontology_file="ontology.ttl", 
#     #     output_file="generated_mappings_from_sql.obda"
#     # )

def create_generator(ollama_host=None, model_name=None):
    """Factory function per creare un'istanza del generatore"""
    return OBDAMappingGenerator(
        ollama_host=ollama_host or MAIN_IP,
        model_name=model_name or "deepseek-r1:70b"
    )