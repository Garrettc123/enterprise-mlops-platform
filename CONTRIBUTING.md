# Contributing to Enterprise MLOps Platform

## Development Setup

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- GPU drivers (optional, for GPU acceleration)
- Kubernetes (optional, for production deployment)

### Local Environment Setup

```bash
# Clone the repository
git clone https://github.com/Garrettc123/enterprise-mlops-platform.git
cd enterprise-mlops-platform

# Start services with docker-compose
docker-compose up -d

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/ --cov

# Start the platform
python -m mlops_platform
```

### Environment Variables

Create `.env`:

```
MONGODB_URI=mongodb://localhost:27017/mlops_db
DATABASE_URL=postgresql://user:password@localhost:5432/mlops_db
REDIS_URL=redis://localhost:6379
S3_BUCKET=mlops-artifacts
MLFLOW_TRACKING_URI=http://localhost:5000
WANDB_API_KEY=your-api-key
```

## Code Style

- **Python**: Black + Flake8 + MyPy
- **Docstrings**: Google style
- **Commit Messages**: Conventional Commits

## Testing

Minimum requirements:
- Unit tests: 80%+ coverage
- Integration tests for ML pipelines
- GPU compatibility tests

## Pull Request Process

1. Fork and create feature branch
2. Write tests for new functionality
3. Ensure all tests pass locally
4. Submit PR with detailed description
5. CI/CD pipeline must pass
6. Code review approval
7. Merge triggers staging deployment

## Performance Guidelines

- Model deployment should complete in < 5 minutes
- Experiment tracking with < 100ms latency
- Support 50x faster deployments than manual processes

## Questions?

Open an issue or check existing documentation.
