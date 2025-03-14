"""Command-line entry point for noosphere.telegram.batch module."""
import sys
import argparse
from pathlib import Path

from noosphere.telegram.batch.logging import setup_logging, get_logger
from noosphere.telegram.batch.config import load_config

logger = get_logger("noosphere.telegram.batch.main")

def main():
    """Main entry point for telegram batch export processing."""
    parser = argparse.ArgumentParser(description="Noosphere Telegram Batch Export Processor")
    parser.add_argument("--config", "-c", help="Path to config file", required=False)
    parser.add_argument("--input-dir", "-i", help="Path to Telegram export directory", required=True)
    parser.add_argument("--output-dir", "-o", help="Path to output directory", required=True)
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--window-type", choices=["session", "size"], default="session",
                        help="Window type (session or size based)")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = Path(args.output_dir) / "noosphere_telegram_batch.log"
    setup_logging(level=log_level, log_file=str(log_file))
    
    logger.info("Starting Telegram export processing", 
                context={
                    "input_dir": args.input_dir,
                    "output_dir": args.output_dir,
                    "window_type": args.window_type
                })
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Import here to avoid circular imports
        from noosphere.telegram.batch.pipelines.process import TelegramExportProcessor
        
        # Process the export
        processor = TelegramExportProcessor(config)
        processor.process(
            input_path=args.input_dir,
            output_path=args.output_dir,
            window_type=args.window_type
        )
        
        logger.info("Processing completed successfully")
        return 0
    except Exception as e:
        logger.exception(f"Error processing Telegram export: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())