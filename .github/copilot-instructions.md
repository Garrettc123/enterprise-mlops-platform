# Enterprise MLOps Platform — Copilot Instructions

## Purpose
Complete MLOps lifecycle: experiment tracking, model versioning, A/B testing, monitoring, auto-retraining.

## Standards
- Python 3.11+, FastAPI, MLflow/custom tracking, scikit-learn, PyTorch
- All experiment data stored in S3 + DynamoDB
- All model artifacts use versioned S3 keys: `models/{name}/{version}/`
- GPU cluster calls must use `continue-on-error: true` in CI
- New pipeline stages must register in `pipeline_registry.py`

## Key Modules
- `experiment_tracker/` — logging, comparison, search
- `model_registry/` — versioning, lineage, artifact storage
- `serving/` — real-time + batch inference APIs
- `monitoring/` — drift detection, alerting, dashboards
- `gpu_manager/` — cluster orchestration, auto-scaling

## CI/CD
- GitHub Actions workflows are the source of truth for deployments
- All Docker images tagged with git SHA
- Canary deployments default to 10% traffic split
