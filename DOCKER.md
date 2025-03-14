# Noosphere Docker Setup

This document explains how to use the provided Docker Compose configuration to set up all the required dependencies for Noosphere.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- NVIDIA GPU with drivers installed (recommended for optimal performance)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)

## Services

The Docker Compose configuration includes the following services:

1. **Ollama** - Runs the LLM and embedding models locally
2. **Qdrant** - Vector database for storing and querying embeddings
3. **Initializers** - Helper services to pull models and create collections

## GPU Support

The Ollama service is configured to use GPU if available. If you don't have a GPU or don't want to use it, modify the `docker-compose.yml` file by removing the `deploy` section under the Ollama service.

## Quick Start

1. Copy the environment file:

```bash
cp .env.example .env
```

2. Adjust any settings in the `.env` file if needed (the defaults should work well for most users)

3. Start the services:

```bash
docker-compose up -d
```

4. Wait for the models to download (this can take 5-15 minutes depending on your internet connection and machine specifications):

```bash
docker-compose logs -f ollama-init
```

> **IMPORTANT**: The initial download of LLM models is a one-time operation but requires downloading several gigabytes of data. The Mistral model is approximately 4GB, LLaVA is around 4GB, and the embedding model is about 800MB. During this time, the health checks for the containers may show as "unhealthy" - this is normal and they will become healthy once initialization is complete.

5. Verify that the services are running:

```bash
docker-compose ps
```

6. Manually check service health to confirm they're ready for use:

```bash
# Check if Ollama is responding
curl http://localhost:11434
# You should see a simple response like "Ollama is running"

# Check if Qdrant is responding
curl http://localhost:6333/healthz
# Should return a 200 OK response with {"status":"ok"}

# Check for available models in Ollama
curl http://localhost:11434/api/tags
# Should show JSON with your models (mistral, llava, nomic-embed-text)
```

7. Verify that the Qdrant collections were created:

```bash
curl http://localhost:6333/collections
# Should show a list that includes telegram_text and telegram_images

curl http://localhost:6333/collections/telegram_text
curl http://localhost:6333/collections/telegram_images
# These should return details about each collection
```

## Configuration

The default Docker Compose setup matches the `conf.example.yaml` file. If you want to use the Docker services, make sure your configuration uses:

- Ollama URL: `http://localhost:11434`
- Qdrant URL: `http://localhost:6333`

## Managing Data

The configuration uses Docker volumes to persist data:

- `ollama_data` - Stores the downloaded models
- `qdrant_data` - Stores the vector database

You can inspect these volumes with:

```bash
docker volume ls | grep noosphere
```

## Stopping the Services

To stop the services:

```bash
docker-compose down
```

To completely remove the services and data:

```bash
docker-compose down -v
```

## Troubleshooting

### Health Checks Failing

If containers remain in "unhealthy" state after 15-20 minutes:

1. Check the logs for errors:
   ```bash
   docker-compose logs -f
   ```

2. Adjust the health check parameters in `docker-compose.yml` to be more lenient:
   ```yaml
   healthcheck:
     test: ["CMD", "curl", "-f", "http://localhost:11434"]
     interval: 60s
     timeout: 60s
     retries: 5
     start_period: 60s
   ```

3. If using the containers for testing, you can bypass health checks by setting conditions to `service_started` rather than `service_healthy`.

### Models Not Downloading

If the models fail to download, you can manually pull them:

```bash
docker exec -it noosphere-ollama ollama pull mistral:latest
docker exec -it noosphere-ollama ollama pull llava:latest
docker exec -it noosphere-ollama ollama pull nomic-embed-text:latest
```

### Qdrant Collections Not Created

If the Qdrant collections were not created during initialization:

1. Check if the Qdrant container is running:
   ```bash
   docker ps | grep qdrant
   ```

2. Manually create the collections:
   ```bash
   curl -X PUT http://localhost:6333/collections/telegram_text \
     -H 'Content-Type: application/json' \
     -d '{"vectors": {"size": 768, "distance": "Cosine"}}'
   
   curl -X PUT http://localhost:6333/collections/telegram_images \
     -H 'Content-Type: application/json' \
     -d '{"vectors": {"size": 768, "distance": "Cosine"}}'
   ```

3. Verify the collections were created:
   ```bash
   curl http://localhost:6333/collections
   ```

### GPU Not Detected

If Ollama isn't using your GPU:

1. Verify that the NVIDIA Container Toolkit is installed
2. Check if Docker can see your GPU:

```bash
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

3. Check Ollama's logs:

```bash
docker-compose logs ollama
```

## Using Docker for Tests

### Before Running Tests

Make sure all services are healthy before running integration tests:

```bash
# Verify Ollama is responding and has models loaded
curl http://localhost:11434/api/tags | grep mistral
curl http://localhost:11434/api/tags | grep llava
curl http://localhost:11434/api/tags | grep nomic-embed

# Verify Qdrant is ready and collections exist
curl http://localhost:6333/collections | grep telegram_text
curl http://localhost:6333/collections | grep telegram_images
```

If any of these commands fail, the services aren't fully initialized yet. Wait a bit longer or check the logs for errors.

### Running Tests With Docker

Once services are ready, run the tests:

```bash
# Run unit tests that don't require Docker services
poetry run pytest tests/a_unit/

# Run integration tests (requires healthy Docker services)
poetry run pytest tests/b_integration/

# Run all tests (both unit and integration)
poetry run pytest
```

### Running Tests Without Docker

If you're running tests and want to skip the Docker-dependent tests:

```bash
# For running unit tests only
poetry run pytest tests/a_unit/

# For excluding tests that require Docker services
poetry run pytest -k "not integration"
```