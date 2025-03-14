"""Command-line entry point for image summarization."""
from noosphere.telegram.batch.transforms.image.summarize import main

if __name__ == "__main__":
    exit(main())