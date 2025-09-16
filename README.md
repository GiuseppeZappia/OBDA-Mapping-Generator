# OBDA Mapping Generator – LLM-Powered Tool

This project was developed during my internship at **Aalborg University** and aims to automatically generate **.obda mapping files** for [Ontop](https://ontop-vkg.org/) from a given database schema (SQL or CSV) and ontology (RDF/OWL).  
The system leverages **Large Language Models (LLMs)** running on **GPU-powered HPC infrastructure** via **CLAAUDIA** (Slurm scheduler) to generate high-quality OBDA mappings.  

Additionally, a **user-friendly web application** (Vue 3 + Tailwind) was built to simplify the process for end users, allowing them to upload input files and download the generated mappings.

---

## 🚀 Features

- **Database schema ingestion**  
  - Accepts **SQL DDL** or **CSV** (schema automatically inferred).
- **Ontology parsing**  
  - Supports **TTL, OWL, RDF, XML**.
- **LLM-powered OBDA mapping generation**  
  - Correctly handles primary keys, foreign keys, composite keys, and URI template generation.
  - Cleans and validates the generated mapping syntax.
- **Interactive web interface**  
  - Drag-and-drop upload, progress indicators, and real-time log streaming.
  - **Download option for the generated `.obda` mapping file** directly from the interface.
- **Robust backend**  
  - Implemented with **Flask**, includes job tracking, logging, file cleanup (after 24h), and REST API endpoints.

---

## 🛠️ Architecture Overview

```
+---------------------+           +---------------------+
|     Vue + Tailwind  |  <----->  | Flask REST API      |
|  (Frontend Web App) |           | (Python Backend)    |
+---------------------+           +---------------------+
         |                                    |
         |  Upload schema + ontology           |
         |  Request mapping generation         |
         v                                    v
+------------------------------------------------------+
|      OBDAMappingGenerator (Python)                   |
|  - Schema parsing (SQL/CSV)                          |
|  - Ontology parsing (rdflib)                         |
|  - Prompt engineering for LLM                        |
|  - Execution on Ollama / CLAAUDIA HPC infrastructure |
|  - Mapping validation & cleaning                     |
+------------------------------------------------------+
```

---

## 📦 Requirements

### Backend
- **Python 3.10+**
- Main dependencies:
  - `flask`, `flask-cors`
  - `pandas`, `numpy`, `rdflib`
  - `ollama` (LLM client)
  - `gunicorn` (recommended for production)

Install dependencies:
```bash
pip install -r requirements.txt
```

### Frontend
- [Vue 3](https://vuejs.org/)
- [TailwindCSS](https://tailwindcss.com/)

The UI is a single `index.html` file – no build step required.  
Simply open in a browser or serve with a static file server.

---

## ▶️ Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/GiuseppeZappia/OBDA-Mapping-Generator.git
cd OBDA-Mapping-Generator
```

2. **Start the backend**
```bash
python app.py
```
Backend runs by default at `http://localhost:5000`.

3. **Open the frontend**  
Open `index.html` in a browser (or serve with a static server such as `python -m http.server`).

---

## 🌐 REST API Endpoints

| Endpoint                   | Method | Description |
|---------------------------|--------|-------------|
| `/api/health`             | GET    | Backend health check |
| `/api/generate-mappings`  | POST   | Start OBDA mapping generation (requires `data_file` and `ontology_file`) |
| `/api/logs/<job_id>`      | GET    | Real-time streaming of logs (Server-Sent Events) |
| `/api/job-status/<job_id>`| GET    | Get job status (processing, completed, failed) |
| `/api/download/<job_id>`  | GET    | **Download the generated `.obda` file** |

---

## 🖥️ User Interface

The UI shows:
- **File upload areas** for schema & ontology
- **Job status panel** with progress indicator
- **Real-time log viewer** with auto-scroll
- **Result section** with mapping statistics and direct download link for the generated `.obda`

---

## 🧠 Technical Notes

- **Containerization**: The backend can be easily containerized with Docker for deployment on HPC clusters or servers.
- **Scalability**: Mapping generation runs asynchronously and can leverage multiple GPUs.
- **File cleanup**: Uploaded files and generated outputs are automatically deleted after 24 hours.

---

