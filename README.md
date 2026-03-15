# AI Resume Screening System

Production-style AI recruitment platform built around the Kaggle dataset `rhythmghai/resume-screening-dataset-200k-candidates`.

This repository now includes a server-oriented deployment path for:

- Ubuntu 24.04
- NVIDIA GPUs
- Docker + NVIDIA Container Toolkit
- aggressive `server_max` training defaults for large machines like `2x RTX 3090 / 38 vCPU / 200 GB RAM`

## What the app does

- auto-downloads the Kaggle dataset via `kagglehub`
- detects the native classification target `hired`
- creates a documented synthetic regression target `synthetic_candidate_score`
- supports dataset preview, EDA, training, comparison, ranking, export, and candidate explanations
- restores the latest trained model after restart

## Project layout

```text
.
|-- backend/
|   |-- Dockerfile
|   `-- app/
|-- frontend/
|   |-- Dockerfile
|   |-- nginx.conf
|   `-- src/
|-- data/
|-- models/
|-- notebooks/
|-- scripts/
|   `-- ubuntu/
|-- docker-compose.yml
|-- requirements.txt
|-- requirements.server.txt
`-- .env.example
```

## Training modes

The backend supports:

- `classification` -> predicts `hired`
- `regression` -> predicts `synthetic_candidate_score`

Training profiles:

- `rapid`
- `balanced`
- `max_accuracy`
- `server_max`

`server_max` is intended for strong GPU servers and uses:

- larger compare sets
- deeper boosting defaults
- larger MLP widths
- mixed precision
- TF32 on Ampere GPUs
- multi-GPU `DataParallel` for PyTorch
- higher dataloader worker counts
- aggressive CPU threading

## Stronger server training stack

For server deployments the project is prepared to use:

- `CatBoost` with GPU support
- `XGBoost` with CUDA device support
- `HistGradientBoosting`
- `ExtraTrees`
- `RandomForest`
- `PyTorch MLP`

The Docker backend image installs:

- `catboost`
- `xgboost`
- CUDA-enabled PyTorch from the official `cu128` wheels

## Why the Docker image uses CUDA 12.8

The container uses CUDA 12.8 user-space libraries because that is the stable PyTorch CUDA wheel target used here.

If your host has modern NVIDIA drivers and a newer CUDA toolkit such as CUDA 13, that is normally fine: the container uses the host driver through NVIDIA Container Toolkit, and newer drivers are backward compatible with older CUDA user-space runtimes.

## Quick start on Ubuntu 24.04

### 1. Install Docker + NVIDIA Container Toolkit

```bash
sudo bash scripts/ubuntu/bootstrap_ubuntu_24_04_gpu.sh
```

### 2. Start the full stack

```bash
bash scripts/ubuntu/run_gpu_stack.sh
```

This brings up:

- frontend: `http://localhost/`
- backend docs: `http://localhost:8000/docs`
- remote access without a domain also works: `http://SERVER_IP/`

### 3. Trigger max-server training

```bash
bash scripts/ubuntu/train_server_max.sh
```

### 4. Stop everything

```bash
bash scripts/ubuntu/stop_stack.sh
```

## Quick start on Ubuntu 24.04 without Docker

If Docker is unavailable on the server, use the native stack:

### 1. Install native runtime dependencies

```bash
sudo bash scripts/ubuntu/bootstrap_native_ubuntu_24_04.sh
```

### 2. Start backend + nginx frontend

```bash
sudo bash scripts/ubuntu/run_native_stack.sh
```

This brings up:

- web UI: `http://SERVER_IP/`
- backend docs: `http://SERVER_IP:8000/docs`

### 3. Trigger max-server training

```bash
bash scripts/ubuntu/train_native_server_max.sh
```

### 4. Stop the native stack

```bash
bash scripts/ubuntu/stop_native_stack.sh
```

## Docker deployment details

The backend container is configured for high-throughput training:

- `gpus: all`
- `ipc: host`
- `shm_size: 16gb`
- high CPU thread env vars
- `DEFAULT_TRAINING_PROFILE=server_max`
- multi-GPU enabled for the PyTorch path

The frontend container is production-style:

- Vite build at image build time
- served by `nginx`
- `/api/*` proxied to the backend container
- `server_name _;` so the web UI can be opened directly by server IP without a domain

## Environment variables

See `.env.example`.

Most relevant server variables:

- `DEFAULT_TRAINING_PROFILE`
- `MAX_CPU_WORKERS`
- `TORCH_DATA_LOADER_WORKERS`
- `TORCH_COMPILE_ENABLED`
- `TORCH_AMP_ENABLED`
- `TORCH_ALLOW_TF32`
- `TORCH_USE_DATA_PARALLEL`
- `TORCH_EVAL_BATCH_SIZE`
- `SERVER_MAX_CLASSIFICATION_MODELS`
- `SERVER_MAX_REGRESSION_MODELS`
- `NVIDIA_VISIBLE_DEVICES`
- `OMP_NUM_THREADS`
- `MKL_NUM_THREADS`
- `NUMEXPR_MAX_THREADS`

`SERVER_MAX_*_MODELS` accept JSON arrays so you can override the heavy compare-set without changing code.

## Local non-Docker run

### Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## API

- `GET /api/dataset/info`
- `GET /api/dataset/preview`
- `GET /api/eda/summary`
- `POST /api/train/start`
- `POST /api/train/stop`
- `GET /api/train/status`
- `GET /api/train/results`
- `GET /api/models/list`
- `POST /api/models/select/{run_id}`
- `GET /api/models/compare`
- `POST /api/predict`
- `GET /api/candidates/top`
- `GET /api/candidates/{id}`
- `GET /api/candidates/export`
- `GET /api/system/device`

## Dataset logic

On first access the backend runs:

```python
import kagglehub
path = kagglehub.dataset_download("rhythmghai/resume-screening-dataset-200k-candidates")
```

Detected native target:

- `hired`

Synthetic regression target:

- `synthetic_candidate_score`

Synthetic score uses transparent weighted components from:

- academics
- internships
- projects
- technical skills
- certifications
- experience
- soft skills
- resume depth
- university and company context

## Validation already performed

The project was smoke-tested locally by:

- loading the real Kaggle dataset
- building the frontend successfully with `npm run build`
- compiling backend modules with `python -m compileall backend/app`
- running end-to-end classifier training and generating ranked candidates

## Files to look at first

- `backend/app/main.py`
- `backend/app/services/training_service.py`
- `backend/app/ml/torch_model.py`
- `backend/app/ml/model_factory.py`
- `frontend/src/pages/TrainingLabPage.tsx`
- `docker-compose.yml`
- `scripts/ubuntu/run_gpu_stack.sh`
