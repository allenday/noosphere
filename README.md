# Noosphere

An internet of knowledge graphs with progressive summarization for text and media.

Noosphere is a toolkit for building personal knowledge graphs through progressive summarization of digital communications. It processes chat data from various sources, extracts insights, and builds a connected knowledge graph that grows more valuable over time.

## Vision

Inspired by Tiago Forte's "Building a Second Brain" methodology, Noosphere employs progressive summarization techniques to distill valuable insights from your digital conversations. The long-term vision is to create a collaborative network of knowledge graphs - an "internet of knowledge" - enabling serendipitous connections between ideas and people.

## Components

- `noosphere.telegram.batch` - Process Telegram chat history exports
- `noosphere.telegram.stream` - Coming soon: Real-time processing of Telegram messages
- `noosphere.discord` - Coming soon: Discord message processing
- `noosphere.twitter` - Coming soon: Twitter/X data processing

## Features

- Extract meaningful summaries from text conversations
- Analyze and summarize images using multimodal language models
- Create searchable vector embeddings of content
- Store and query using vector databases
- Process data in windowed segments (by time or message count)
- Highly configurable pipeline with pluggable components

## Quick Start

Get up and running in minutes:

```bash
# Clone the repository
git clone https://github.com/allenday/noosphere.git
cd noosphere

# Set up the Docker environment (see DOCKER.md for details)
cp .env.example .env
docker-compose up -d

# Note: Docker services may take several minutes to fully initialize
# as they need to download LLM models (1-4GB each)

# Install the Python package
pip install -e .

# Create a configuration file
cp conf.example.yaml conf.yaml

# Process a Telegram export
noo-telegram-batch --input-dir /path/to/telegram_export --output-dir ./output --config conf.yaml
```

For more detailed instructions, see the [Docker Setup](DOCKER.md) guide.

## Installation

### Requirements

- Python 3.10 or higher
- For image summarization: Ollama with a multimodal model like LLaVA
- For vector storage: Qdrant

```bash
# Install from the repository
pip install -e .

# Or install directly using pip
pip install git+https://github.com/allenday/noosphere.git
```

## Usage

### Processing Telegram Exports

1. Export your Telegram chat history using the official Telegram Desktop app
2. Create a configuration file (see `conf.yaml` example)
3. Run the processor:

```bash
# Using the CLI tool
noo-telegram-batch --input-dir /path/to/telegram_export --output-dir /path/to/output --config conf.yaml

# Or programmatically
from noosphere.telegram.batch.pipelines.process import TelegramExportProcessor
from noosphere.telegram.batch.config import load_config

config = load_config("conf.yaml")
processor = TelegramExportProcessor(config)
processor.process(
    input_path="/path/to/telegram_export",
    output_path="/path/to/output",
    window_type="session"  # or "size" for sliding windows
)
```

### Configuration

Noosphere uses YAML configuration files. See example below:

```yaml
logging:
  level: INFO
  log_file: logs/noosphere.log

llm_services:
  - name: summarizer
    type: ollama
    model: llama3
    url: http://localhost:11434

embedding_services:
  - name: embedder
    type: ollama
    model: llama3
    url: http://localhost:11434

# Additional config options omitted for brevity
```

## Development

```bash
# Clone the repository
git clone https://github.com/allenday/noosphere.git
cd noosphere

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Create required directories
mkdir -p logs
mkdir -p tests/a_unit  # Required by pytest configuration

# Run tests
pytest
```

> **Note**: The test directory structure is currently missing. The repository is configured to expect tests in the `tests/a_unit` directory as specified in `pytest.ini`.

## License

MIT