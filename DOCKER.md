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

1. Start the services:

```bash
docker-compose up -d
```

2. Wait for the models to download (this may take a while depending on your internet connection):

```bash
docker-compose logs -f ollama-init
```

3. Verify that the services are running:

```bash
docker-compose ps
```

4. Verify that the models are available:

```bash
curl http://localhost:11434/api/tags
```

5. Verify that the Qdrant collections were created:

```bash
curl http://localhost:6333/collections/telegram_text
curl http://localhost:6333/collections/telegram_images
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

### Models Not Downloading

If the models fail to download, you can manually pull them:

```bash
docker exec -it noosphere-ollama ollama pull mistral:latest
docker exec -it noosphere-ollama ollama pull llava:latest
docker exec -it noosphere-ollama ollama pull nomic-embed-text:latest
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