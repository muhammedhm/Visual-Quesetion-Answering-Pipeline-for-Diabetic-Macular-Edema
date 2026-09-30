# DeepEyeNet

DeepEyeNet is a research prototype for retinal fundus image analysis focused on diabetic macular edema (DME). It combines visual question answering (VQA) with an optional multimodal retrieval-augmented search workflow. The project includes a FastAPI backend, a React/Vite web client, and a separate Streamlit interface.

> **Medical disclaimer:** This software is for research and informational purposes only. Its output is AI-generated, is not a medical diagnosis, and must not be used to make clinical decisions. Consult a qualified ophthalmologist.

## Features

- Analyze fundus images with five supported questions:
  - Detect hard exudates across the full image.
  - Detect hard exudates in the fovea.
  - Predict DME grade: 0 (none), 1 (mild), or 2 (moderate to severe).
  - Check hard exudates or the optic disc in a highlighted region using a mask.
- Retrieve visually similar dataset cases using CLIP embeddings and a FAISS index.
- User login, query history, and admin user management.
- Input checks for image format, size, dimensions, retinal-image appearance, and query content.
- Evaluate VQA/RAG predictions and optional text-generation metrics.

## Architecture

```text
React + Vite frontend ── HTTP / Bearer token ──> FastAPI (api_app.py)
                                                    ├── LLaVA + optional PEFT adapter (VQA)
                                                    ├── CLIP embeddings + FAISS (RAG retrieval)
                                                    └── SQLite (users, sessions, history)

Streamlit interface (app.py) ─────────────────────> shared inference, retrieval, and database modules
```

The RAG workflow finds similar examples and presents retrieved-case context and structured output. The inference implementation currently makes its VQA answer with the image/question model; treat retrieved examples as supporting context rather than assuming they directly condition the generated answer.

## Requirements

- Python 3.10 or later.
- Node.js and npm for the React client.
- A compatible PyTorch setup. GPU acceleration is recommended; model loading uses 4-bit quantization with BitsAndBytes.
- The DME VQA dataset (`trainqa.json` and its corresponding fundus images) to build or rebuild the RAG index and to run evaluations.
- Model files are not included in this repository. At runtime, Hugging Face downloads the base model `llava-hf/llava-interleave-qwen-0.5b-hf` unless a local model cache is available. If the `./llava-deepeye-adapter/` directory exists, the PEFT adapter is loaded on top of the base model; otherwise, inference falls back to the base model.

Install Python dependencies from the repository root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install frontend dependencies:

```powershell
cd frontend
npm install
cd ..
```

Install a PyTorch build appropriate for your machine and CUDA version if the default dependency installation does not provide a suitable build. Some environments/platforms may not support BitsAndBytes quantization; inference depends on compatible PyTorch, Transformers, PEFT, and BitsAndBytes versions.

## Dataset and RAG index

The dataset is external and is not bundled here. Place or extract it so that the project can access:

```text
dme_vqa/
├── qa/
│   └── trainqa.json
└── visual/
    └── train/
        └── <fundus images>
```

The index builder tries common relative locations and the current configured locations in `rag_engine.py`. If your dataset is elsewhere, pass the paths explicitly:

```powershell
python build_rag_index.py --qa "C:/path/to/dme_vqa/qa/trainqa.json" --images "C:/path/to/dme_vqa/visual/train"
```

A small test index can be built with `--max 50`; rebuild an existing index with `--force`:

```powershell
python build_rag_index.py --max 50
python build_rag_index.py --force --qa "C:/path/to/trainqa.json" --images "C:/path/to/train-images"
```

The builder writes the FAISS index and metadata under `rag_index/`. These are generated artifacts and are ignored by Git. Rebuild them when needed; do not commit the dataset or index unless you have verified that redistribution is permitted.

## Run the application

### React frontend and FastAPI backend

Start the backend from the repository root:

```powershell
uvicorn api_app:app --host 127.0.0.1 --port 8000 --reload
```

Open a second terminal and start the web client:

```powershell
cd frontend
npm run dev
```

Vite prints the local frontend URL (normally `http://localhost:5173`). The client uses `http://localhost:8000` by default. To point it at another backend, create `frontend/.env.local`:

```dotenv
VITE_API_URL=http://localhost:8000
```

Restart the Vite server after changing environment variables. The API provides interactive documentation at `http://127.0.0.1:8000/docs` and a health check at `http://127.0.0.1:8000/health`.

### Streamlit interface

Alternatively, run the Streamlit application from the repository root:

```powershell
streamlit run app.py
```

The Streamlit interface is a separate UI from the React client; run either interface against the shared project modules as needed.

## Initial admin account and configuration

On first database initialization, if there are no users, the API/database creates an administrator using:

- Username: `admin`
- Password: `admin123`

Set these environment variables **before the first database initialization** to use different initial credentials:

```powershell
$env:DEEPEYE_ADMIN_USERNAME = "your-admin-name"
$env:DEEPEYE_ADMIN_PASSWORD = "use-a-strong-password"
```

The default credentials are for local development only. Change them before exposing the service. The database is created as `deepeye.db` in the current working directory. The API's CORS allowlist can be configured with comma-separated origins using `DEEPEYE_CORS_ORIGINS`; by default it allows all origins, which is unsuitable for a public deployment.

## API overview

All application endpoints except `/health` and `/auth/login` require a Bearer token. FastAPI's generated documentation endpoints (`/docs`, `/redoc`, and `/openapi.json`) are also available without authentication by default. Authenticate with `POST /auth/login`, then send `Authorization: Bearer <access_token>` on protected requests.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Health check (public) |
| `POST` | `/auth/login` | Obtain an access token (public) |
| `POST` | `/auth/logout` | Invalidate the current token |
| `GET` | `/auth/me` | Get the current user |
| `GET` | `/vqa/questions` | List supported question types |
| `POST` | `/vqa/chat` | Submit an image, question, and optional region mask for VQA |
| `POST` | `/rag/chat` | Run VQA with retrieval; accepts optional `top_k` and `alpha` form fields |
| `GET` | `/rag/status` | Check whether the retrieval index is available |
| `GET` / `DELETE` | `/history/vqa`, `/history/vqa/{query_id}` | List or delete the current user's VQA history |
| `GET` / `DELETE` | `/history/rag`, `/history/rag/{search_id}` | List or delete the current user's RAG history |
| `GET` / `POST` / `DELETE` | `/admin/users` and `/admin/users/{username}` | Admin-only user management |

For `/vqa/chat` and `/rag/chat`, submit `multipart/form-data` with:

- `image`: required JPG, PNG, BMP, or TIFF fundus image.
- `message`: required supported DME question, or an explain/describe request for all supported questions.
- `mask`: optional image mask for region-focused questions.
- `/rag/chat` additionally accepts `top_k` (default 5, limited to 1–10) and `alpha` (default 0.7).

Use the OpenAPI page at `/docs` for complete request schemas and interactive testing.

## Evaluation

Both evaluation scripts expect the dataset and project files to be available locally. Their default dataset location is `../dme_vqa/` relative to this repository. Supply `--qa-file` and `--image-dir` if your paths differ.

Run a small VQA and RAG metric evaluation:

```powershell
python evaluate_multimodal_rag_metrics.py --architecture both --sample-size 100 --qa-file "C:/path/to/trainqa.json" --image-dir "C:/path/to/train-images"
```

This writes `metrics_summary.json`, `predictions.csv`, confusion matrices, and per-class metrics into `deepeye_eval_outputs/` by default. Use `--sample-size 0` (the default) to evaluate all usable rows, and `--skip-inside` to omit region-mask questions when masks are unavailable.

For ROUGE-L and optional BERTScore text metrics:

```powershell
python -m pip install bert-score
python evaluate_text_metrics_rouge_bertscore.py --architecture both --sample-size 100 --qa-file "C:/path/to/trainqa.json" --image-dir "C:/path/to/train-images"
```

Pass `--skip-bertscore` to calculate ROUGE-L without BERTScore. Text-metric outputs default to `../deepeye_text_metric_outputs/`.

## Generated and local data

The repository's `.gitignore` excludes Python environments/caches, local secrets, the SQLite database, uploaded images, generated indexes/evaluation results, model weights, and frontend build/dependency directories. Keep user uploads, credentials, and datasets private. Ensure you have permission to use any dataset or model weights before downloading, using, or redistributing them.

## Safety and limitations

- This is a research prototype, not a validated medical device or diagnostic system.
- Model predictions can be incorrect or incomplete; do not use them to guide patient care.
- Region-specific questions are intended to use a binary mask; without one, the interface may fall back to the full image.
- RAG requires both the FAISS index and metadata generated from the dataset.
- The first model inference may take time while model weights are downloaded and loaded into memory.
- Local development defaults (including admin credentials, permissive CORS, and local SQLite storage) must be hardened before deployment.
