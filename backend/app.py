from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import os
import uuid
import threading
import time
import json
import queue
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from mapping_generator import create_generator
import logging
import sys

# Setup logging with custom handler for streaming
class WebSocketLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_queues = {}  # job_id -> queue
    
    def emit(self, record):
        log_msg = self.format(record)
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Determine job_id from record if it finds it 
        job_id = getattr(record, 'job_id', None)
        
        log_entry = {
            'timestamp': timestamp,
            'level': record.levelname.lower(),
            'message': log_msg,
            'job_id': job_id
        }
        
        # Send to every queue that have this job_id or to all if no specific job_id
        target_queues = []
        if job_id and job_id in self.log_queues:
            target_queues = [self.log_queues[job_id]]
        elif job_id:
            # If there is a job_id but no queue, send to all active queues
            target_queues = list(self.log_queues.values())
            
        for q in target_queues:
            try:
                q.put_nowait(log_entry)
            except queue.Full:
                pass  # Ignore if the queue is full

# Initialize the custom handler GLOBALLY
web_log_handler = WebSocketLogHandler()
web_log_handler.setFormatter(logging.Formatter('%(message)s'))

# Setup logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logging.getLogger('werkzeug').setLevel(logging.WARNING)

root_logger = logging.getLogger()
if not any(isinstance(h, WebSocketLogHandler) for h in root_logger.handlers):
    root_logger.addHandler(web_log_handler)

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configurations
UPLOAD_FOLDER = '/home/cs.aau.dk/la51mw/giu/uploads'
OUTPUT_FOLDER = '/home/cs.aau.dk/la51mw/giu/output'
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
ALLOWED_DATA_EXTENSIONS = {'csv', 'sql'}
ALLOWED_ONTOLOGY_EXTENSIONS = {'ttl', 'owl', 'rdf', 'xml'}
CLEANUP_AFTER_HOURS = 24

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Creation necessary directories
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Job tracking
active_jobs = {}
completed_jobs = {}

def allowed_file(filename, allowed_extensions):
    """Verify if the file extension matches is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def cleanup_old_files():
    """Periodic Cleanup of old files"""
    cutoff_time = datetime.now() - timedelta(hours=CLEANUP_AFTER_HOURS)
    
    for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
        if not os.path.exists(folder):
            continue
            
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            try:
                if os.path.isfile(filepath):
                    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                    if file_time < cutoff_time:
                        os.remove(filepath)
                        logger.info(f"Cleaned up old file: {filepath}")
            except Exception as e:
                logger.error(f"Error cleaning up file {filepath}: {e}")
    
    # Cleanup job tracking
    jobs_to_remove = []
    for job_id, job_info in completed_jobs.items():
        if job_info['completed_at'] < cutoff_time:
            jobs_to_remove.append(job_id)
    
    for job_id in jobs_to_remove:
        del completed_jobs[job_id]

def start_cleanup_thread():
    """Start the cleanup thread """
    def cleanup_worker():
        while True:
            time.sleep(3600)  # Every hour
            cleanup_old_files()
    
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()

class JobLoggerFilter(logging.Filter):
    """Filter to add  job_id to every log record"""
    def __init__(self, job_id):
        super().__init__()
        self.job_id = job_id
    
    def filter(self, record):
        record.job_id = self.job_id
        return True

def setup_job_logging(job_id):
    """Complete Setup of logging for a specific job """
    # Create filter for this job
    job_filter = JobLoggerFilter(job_id)
    
    # List of every logger that could be used
    loggers_to_setup = [
        logging.getLogger(),  # Root logger
        logging.getLogger('mapping_generator'),
        logging.getLogger('__main__'),
        logging.getLogger(__name__),
    ]
    
    filters_added = []
    
    for logger_obj in loggers_to_setup:
        logger_obj.addFilter(job_filter)
        filters_added.append((logger_obj, job_filter))
    
    return filters_added

def cleanup_job_logging(filters_added):
    """Removes job-specific filters"""
    for logger_obj, job_filter in filters_added:
        try:
            logger_obj.removeFilter(job_filter)
        except:
            pass

def generate_mappings_synchronous(job_id, data_filepath, ontology_filepath, output_filepath, ollama_host, model_name):
    # Setup logging per this job
    filters_added = setup_job_logging(job_id)
    
    try:
        logger.info(f"Starting OBDA generation for job {job_id}")
        logger.info(f"Data file: {os.path.basename(data_filepath)}")
        logger.info(f"Ontology file: {os.path.basename(ontology_filepath)}")
        
        # Create generator
        logger.info("Creating mapping generator...")
        generator = create_generator(ollama_host, model_name)
        
        # If the generator has a specific logger, it will be also configured
        if hasattr(generator, 'logger'):
            generator.logger.addFilter(JobLoggerFilter(job_id))
        
        logger.info("Initializing mapping generation...")
        
        # Generate mappings
        logger.info("Starting OBDA mapping generation process...")
        result = generator.generate_obda_mappings(
            data_file=data_filepath,
            ontology_file=ontology_filepath,
            output_file=output_filepath
        )
        
        # Move job from active to completed
        job_info = active_jobs.pop(job_id, {})
        completed_jobs[job_id] = {
            **job_info,
            'completed_at': datetime.now(),
            'status': 'completed' if result['success'] else 'failed',
            'result': result
        }
        
        # Cleanup temporary files
        try:
            os.remove(data_filepath)
            os.remove(ontology_filepath)
        except Exception as cleanup_error:
            logger.warning(f"Cleanup warning: {cleanup_error}")
        
        if result['success']:
            logger.info(f"Generation completed successfully!")
            logger.info(f"Mappings generated: {result['mappings_count']}")
            logger.info(f"Tables processed: {result['tables_processed']}")
            logger.info(f"Output file ready for download")
        else:
            logger.error(f"Generation failed: {result['error']}")
            
        return result
            
    except Exception as gen_error:
        logger.error(f"Generation error: {str(gen_error)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Remove the job from active and add it to failed ones
        job_info = active_jobs.pop(job_id, {})
        completed_jobs[job_id] = {
            **job_info,
            'completed_at': datetime.now(),
            'status': 'failed',
            'error': str(gen_error)
        }
        
        # Cleanup temporary file also in case of error
        try:
            if os.path.exists(data_filepath):
                os.remove(data_filepath)
            if os.path.exists(ontology_filepath):
                os.remove(ontology_filepath)
        except Exception as cleanup_error:
            logger.warning(f"Cleanup warning after error: {cleanup_error}")
        
        return {
            'success': False,
            'error': str(gen_error),
            'mappings_count': 0,
            'tables_processed': []
        }
    
    finally:
        # Cleanup logging filters
        cleanup_job_logging(filters_added)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_jobs': len(active_jobs),
        'completed_jobs': len(completed_jobs)
    })

@app.route('/api/logs/<job_id>', methods=['GET'])
def stream_logs(job_id):
    """Endpoint per streaming dei log in tempo reale"""
    def generate_logs():
        # Create a queue for this client
        log_queue = queue.Queue(maxsize=2000)
        web_log_handler.log_queues[job_id] = log_queue
        
        try:
            # Send initial log
            yield f"data: {json.dumps({'timestamp': datetime.now().strftime('%H:%M:%S'), 'level': 'info', 'message': f'Connected to logs for job {job_id}'})}\n\n"
            
            heartbeat_counter = 0
            while True:
                try:
                    # Wait for a new log 
                    log_entry = log_queue.get(timeout=30)
                    yield f"data: {json.dumps(log_entry)}\n\n"
                    # Reset heartbeat counter when we receive a real log
                    heartbeat_counter = 0
                    
                except queue.Empty:
                    heartbeat_counter += 1
                    if heartbeat_counter >= 5:  
                        yield f"data: {json.dumps({'timestamp': datetime.now().strftime('%H:%M:%S'), 'level': 'debug', 'message': 'Connection active...'})}\n\n"
                        heartbeat_counter = 0

                # Verify if the job is completed
                if job_id in completed_jobs:
                    # Wait for possibly final logs
                    time.sleep(1)
                    # Empty the queue
                    final_logs = []
                    while not log_queue.empty():
                        try:
                            log_entry = log_queue.get_nowait()
                            final_logs.append(log_entry)
                        except queue.Empty:
                            break
                    
                    # Send all final logs
                    for log_entry in final_logs:
                        yield f"data: {json.dumps(log_entry)}\n\n"
                    
                    # Send completion message
                    yield f"data: {json.dumps({'timestamp': datetime.now().strftime('%H:%M:%S'), 'level': 'info', 'message': f'Job {job_id} completed - stream ending'})}\n\n"
                    break
                    
        except GeneratorExit:
            pass
        except Exception as e:
            yield f"data: {json.dumps({'timestamp': datetime.now().strftime('%H:%M:%S'), 'level': 'error', 'message': f'Stream error: {str(e)}'})}\n\n"
        finally:
            # Removes queue when client disconnects
            if job_id in web_log_handler.log_queues:
                del web_log_handler.log_queues[job_id]
    
    return Response(
        generate_logs(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Cache-Control'
        }
    )

@app.route('/api/generate-mappings', methods=['POST'])
def generate_mappings():
    """Endpoint principale per generare i mappings OBDA - VERSIONE SINCRONA"""
    try:
        # Verify files presence
        if 'data_file' not in request.files or 'ontology_file' not in request.files:
            return jsonify({
                'error': 'Both data_file and ontology_file are required'
            }), 400
        
        data_file = request.files['data_file']
        ontology_file = request.files['ontology_file']
        output_filename = request.form.get('output_filename', 'mapping.obda')
        
        # Verify files names
        if data_file.filename == '' or ontology_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Verify extensions
        if not allowed_file(data_file.filename, ALLOWED_DATA_EXTENSIONS):
            return jsonify({
                'error': f'Invalid data file format. Allowed: {", ".join(ALLOWED_DATA_EXTENSIONS)}'
            }), 400
        
        if not allowed_file(ontology_file.filename, ALLOWED_ONTOLOGY_EXTENSIONS):
            return jsonify({
                'error': f'Invalid ontology file format. Allowed: {", ".join(ALLOWED_ONTOLOGY_EXTENSIONS)}'
            }), 400
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Create safe files names
        data_filename = secure_filename(f"{job_id}_{data_file.filename}")
        ontology_filename = secure_filename(f"{job_id}_{ontology_file.filename}")
        
        # Uses frontend given name and add the .obda extension if not present
        if not output_filename.endswith('.obda'):
            output_filename += '.obda'
        
        output_filename_safe = secure_filename(f"{job_id}_{output_filename}")
        
        # Complete paths
        data_filepath = os.path.join(UPLOAD_FOLDER, data_filename)
        ontology_filepath = os.path.join(UPLOAD_FOLDER, ontology_filename)
        output_filepath = os.path.join(OUTPUT_FOLDER, output_filename_safe)
        
        # Save file
        data_file.save(data_filepath)
        ontology_file.save(ontology_filepath)
        
        logger.info(f"Job {job_id}: Files uploaded - Data: {data_file.filename}, Ontology: {ontology_file.filename}")
        
        # Register job as active
        active_jobs[job_id] = {
            'started_at': datetime.now(),
            'data_file': data_file.filename,
            'ontology_file': ontology_file.filename,
            'output_filename': output_filename,  # Save original file name
            'status': 'processing'
        }
        
        # Optional parameter that can be added from frontend if needed in future (to change model or host) 
        ollama_host = request.form.get('ollama_host')
        model_name = request.form.get('model_name')
        
        logger.info(f"Job {job_id}: Starting OBDA generation with ollama_host={ollama_host}, model_name={model_name}")
        
        def thread_wrapper():
            time.sleep(0.5) 
            generate_mappings_synchronous(job_id, data_filepath, ontology_filepath, output_filepath, ollama_host, model_name)
        
        worker_thread = threading.Thread(target=thread_wrapper, daemon=True)
        worker_thread.start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Generation started successfully',
            'log_stream_url': f'/api/logs/{job_id}',
            'status_url': f'/api/job-status/{job_id}'
        })
                
    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<job_id>', methods=['GET'])
def download_mappings(job_id):
    """Download generated OBDA file"""
    try:
        if job_id not in completed_jobs:
            return jsonify({'error': 'Job not found'}), 404
        
        job_info = completed_jobs[job_id]
        
        # Veriftìy that job has been completed successfully
        if job_info['status'] != 'completed':
            return jsonify({'error': 'Job not completed successfully'}), 400
        
        # Finf output file looking for job_id
        output_files = []
        for filename in os.listdir(OUTPUT_FOLDER):
            if filename.startswith(job_id):
                output_files.append(filename)
        
        if not output_files:
            return jsonify({'error': 'Output file not found'}), 404
        
        output_filename = output_files[0]
        output_filepath = os.path.join(OUTPUT_FOLDER, output_filename)
        
        if not os.path.exists(output_filepath):
            return jsonify({'error': 'Output file does not exist'}), 404
        
        # Use the given name from frontend
        download_name = job_info.get('output_filename', 'mappings.obda')
        
        logger.info(f"Job {job_id}: File downloaded as {download_name}")
        
        return send_file(
            output_filepath,
            as_attachment=True,
            download_name=download_name,
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        logger.error(f"Download error for job {job_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/job-status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Obtain job status """
    if job_id in active_jobs:
        job_info = active_jobs[job_id]
        return jsonify({
            'job_id': job_id,
            'status': 'processing',
            'started_at': job_info['started_at'].isoformat(),
            'data_file': job_info['data_file'],
            'ontology_file': job_info['ontology_file'],
            'output_filename': job_info.get('output_filename', 'mapping.obda')
        })
    elif job_id in completed_jobs:
        job_info = completed_jobs[job_id]
        response_data = {
            'job_id': job_id,
            'status': job_info['status'],
            'started_at': job_info['started_at'].isoformat(),
            'completed_at': job_info['completed_at'].isoformat(),
            'data_file': job_info['data_file'],
            'ontology_file': job_info['ontology_file'],
            'output_filename': job_info.get('output_filename', 'mapping.obda')
        }
        
        if job_info['status'] == 'completed' and 'result' in job_info:
            response_data.update({
                'mappings_count': job_info['result']['mappings_count'],
                'tables_processed': job_info['result']['tables_processed'],
                'download_url': f'/api/download/{job_id}'
            })
        elif job_info['status'] == 'failed':
            response_data['error'] = job_info.get('error', 'Unknown error')
            
        return jsonify(response_data)
    else:
        return jsonify({'error': 'Job not found'}), 404

@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """List of all active and completed jobs"""
    jobs_list = []
    
    # Actives Jobs
    for job_id, job_info in active_jobs.items():
        jobs_list.append({
            'job_id': job_id,
            'status': 'processing',
            'started_at': job_info['started_at'].isoformat(),
            'data_file': job_info['data_file'],
            'ontology_file': job_info['ontology_file'],
            'output_filename': job_info.get('output_filename', 'mapping.obda')
        })
    
    # Completed Jobs limited to last 50
    completed_items = sorted(
        completed_jobs.items(),
        key=lambda x: x[1]['completed_at'],
        reverse=True
    )[:50]
    
    for job_id, job_info in completed_items:
        job_data = {
            'job_id': job_id,
            'status': job_info['status'],
            'started_at': job_info['started_at'].isoformat(),
            'completed_at': job_info['completed_at'].isoformat(),
            'data_file': job_info['data_file'],
            'ontology_file': job_info['ontology_file'],
            'output_filename': job_info.get('output_filename', 'mapping.obda')
        }
        
        if job_info['status'] == 'completed' and 'result' in job_info:
            job_data.update({
                'mappings_count': job_info['result']['mappings_count'],
                'tables_processed': job_info['result']['tables_processed']
            })
        elif job_info['status'] == 'failed':
            job_data['error'] = job_info.get('error', 'Unknown error')
            
        jobs_list.append(job_data)
    
    return jsonify({
        'jobs': jobs_list,
        'total_active': len(active_jobs),
        'total_completed': len(completed_jobs)
    })

@app.errorhandler(413)
def too_large(e):
    """Handler for too large files"""
    return jsonify({'error': 'File too large. Maximum size allowed is 100MB'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handler fot not found endpoint"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handler per internal server errors"""
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    
    start_cleanup_thread()
    
    logger.info("Starting Flask OBDA Mapping Generator API")
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    logger.info(f"Output folder: {OUTPUT_FOLDER}")
    logger.info(f"Max file size: {MAX_CONTENT_LENGTH / (1024*1024):.0f}MB")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)