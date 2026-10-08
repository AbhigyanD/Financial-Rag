# Deployment

## What is deployed

| | Status |
|---|---|
| Local Docker (`docker compose`) | **Run and verified** on 2026-10-08 (commands and output below) |
| Google Cloud Run | **Not deployed.** Commands are written below but **not run**: this machine has no `gcloud` CLI, and deploying needs an interactive Google login and a billing account. |
| Public URL | None yet |

## Target: Google Cloud Run

Chosen because it runs the existing container unchanged, behind managed HTTPS, scaling to zero when idle, with Secret Manager built in, so there is no server to operate. The catch is that its containers are stateless, which conflicts with Chroma's on-disk store (see Persistence below).

## Local run (verified)

```bash
cp .env.example .env            # add GROQ_API_KEY
docker compose up --build -d    # demo page + API on :8000; older Streamlit UI on :8501
```

What was actually run and observed on 2026-10-08, with no LLM key set:

```text
$ docker compose up -d   -> api: healthy, ui: healthy   (Docker HEALTHCHECK on /health and /_stcore/health)
$ curl localhost:8000/info
{"llm_provider":"groq","llm_model":"llama-3.3-70b-versatile","llm_key_configured":false,   (default model has since changed to openai/gpt-oss-120b)
 "embedding_provider":"local","embedding_model":"sentence-transformers/all-MiniLM-L6-v2",...}
$ curl -o /dev/null -w "%{http_code}" localhost:8000/       -> 200   (the demo page)
$ curl -F file=@src/financial_rag/web/samples/northwind_annual_report_fy2025.pdf localhost:8000/ingest
{"source":"northwind_annual_report_fy2025.pdf","pages":5,"empty_pages":0,"chunks_stored":5,"request_id":"dcc81f8644f9"}
$ docker compose exec api id -u      -> 10001   (non-root)
$ docker image ls financial-rag      -> 2.75GB
```

Ingest worked with no network access needed for the model: the embedding model is downloaded during `docker build` into `/app/model_cache`. An earlier run (before the Groq switch) also confirmed there's no `.env` inside the image; `.dockerignore` excludes it.

The image is large (2.75 GB): Chroma, ONNX Runtime, PyMuPDF, and the Streamlit stack (pandas, pyarrow) dominate. Dropping the Streamlit client from the image would be the first cut.

State lives in the named volume `rag-data` mounted at `/data` (`PERSIST_DIRECTORY=/data/chroma`, `EMBEDDING_CACHE_DIR=/data/embedding_cache`). It survives `docker compose down`; `docker compose down -v` deletes it.

## Cloud Run commands (written, NOT run, unverified)

Replace `PROJECT`, `REGION` and `BUCKET`.

```bash
gcloud auth login
gcloud config set project PROJECT
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
    cloudbuild.googleapis.com secretmanager.googleapis.com

# 1. Image
gcloud artifacts repositories create rag --repository-format=docker --location=REGION
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT/rag/financial-rag:latest

# 2. Secret: in Secret Manager, never in the image or in env flags
printf %s "$GROQ_API_KEY" | gcloud secrets create groq-api-key --data-file=-
# The service's runtime service account also needs roles/secretmanager.secretAccessor on it.

# 3. Storage for the vector index (see Persistence)
gcloud storage buckets create gs://BUCKET --location=REGION

# 4. API service
gcloud run deploy financial-rag-api \
    --image REGION-docker.pkg.dev/PROJECT/rag/financial-rag:latest \
    --region REGION --execution-environment gen2 \
    --max-instances 1 --memory 1Gi --timeout 120 \
    --set-secrets GROQ_API_KEY=groq-api-key:latest \
    --add-volume name=data,type=cloud-storage,bucket=BUCKET \
    --add-volume-mount volume=data,mount-path=/data \
    --no-allow-unauthenticated

# The demo page is served by the API itself at /, so no second service is needed.
```

Health check: the container honors `$PORT` and serves `GET /health`. Cloud Run's default startup probe is a TCP check on that port; an HTTP probe on `/health` would be configured on the service (not done here).

`--no-allow-unauthenticated` matters: the API has **no authentication of its own**, and every query spends your Groq quota (or money, on a paid plan). With authentication required, opening the demo page needs an authenticated proxy (e.g. `gcloud run services proxy financial-rag-api`). That wiring isn't done.

## Persistence: the real limitation

Cloud Run containers are stateless: local disk is an in-memory filesystem that disappears when an instance stops. Chroma's `PersistentClient` is SQLite plus index files on local disk. Options:

| Option | What happens | Verdict |
|---|---|---|
| Do nothing | Every cold start begins with an empty index; documents must be re-ingested | Fine for a demo, useless otherwise |
| Mount a GCS bucket at `/data` (commands above) | Index survives restarts | Prototype only. Cloud Storage FUSE doesn't give SQLite the file locking and random-write behavior it expects, so the store can corrupt under concurrent writers. `--max-instances 1` avoids multiple writers but doesn't fix FUSE's semantics. |
| Run Chroma in server mode on a VM with a persistent disk; app uses `chromadb.HttpClient` | Cloud Run becomes truly stateless | **The right fix.** One-file change in `storage.py` (`_get_client`). |
| Managed vector store (e.g. pgvector on Cloud SQL) | Same, plus backups and HA | Better for production; `storage.py` would be rewritten (the rest of the code doesn't import Chroma) |

The embedding cache (`/data/embedding_cache`, SQLite) has the same issue. It's only an optimization, so losing it costs money, not correctness.

## Other options (none deployed)

| Option | Cost model | Cold starts | State | Scaling | Ops effort | Status |
|---|---|---|---|---|---|---|
| **GCP Cloud Run** | Per request-second; scales to zero | Yes, when scaled to zero (image and Python import time) | Stateless; needs external store or FUSE mount | Automatic, per request | Low | not deployed (chosen target) |
| AWS App Runner | Per running instance plus requests; can pause | Smaller when kept warm | Stateless | Automatic | Low | not deployed |
| AWS ECS on Fargate | Per task-second; always-on tasks | None while tasks run | EFS volume possible (NFS, the same SQLite caveats apply) | Service auto-scaling rules you write | Medium (VPC, load balancer, task definitions) | not deployed |
| AWS Lambda | Per invocation and duration | Yes; large container image makes them worse | Stateless; 15-min max | Automatic | Medium (streaming responses and a 10 GB image limit need care) | not deployed |
| GKE | Per node, always on | None | Persistent volumes work properly | Pod autoscaling | High (cluster operations) | not deployed |
| Single VM (Compute Engine or EC2) | Per VM-hour, always on | None | Local persistent disk: Chroma works exactly as in dev | Manual (one box) | Medium (OS patching, TLS, process supervision) | not deployed |

For this app as written (local SQLite store, one process), a single VM is the only option where persistence works without changes. Cloud Run is the target because the planned fix (Chroma server or pgvector) makes the app stateless anyway.
