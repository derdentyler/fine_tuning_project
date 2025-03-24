import argparse
import subprocess
from src.utils.config_loader import ConfigLoader
from src.utils.logger_loader import LoggerLoader
from src.youtube_dataset_builder import YouTubeDatasetBuilder
from src.fine_tune import fine_tune_model

logger = LoggerLoader().get_logger()

def run_eda():
    logger.info("Launching EDA notebook...")
    subprocess.run(["jupyter", "notebook", "eda.ipynb"], check=True)

def main():
    parser = argparse.ArgumentParser(description="Run pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, help="Override model name from config")
    parser.add_argument("--task", type=str, choices=["scrape", "eda", "train"], required=True, help="Choose task to run")
    args = parser.parse_args()

    # Загружаем конфиг
    config = ConfigLoader(args.config).config
    model_name = args.model if args.model else config["model_name"]

    logger.info(f"Loaded config from {args.config}")
    logger.info(f"Using model: {model_name}")

    # Выбираем задачу
    if args.task == "scrape":
        logger.info("Running data scraper...")
        builder = YouTubeDatasetBuilder(args.config)
        builder.build_dataset()
    elif args.task == "eda":
        run_eda()
    elif args.task == "train":
        logger.info("Starting fine-tuning...")
        fine_tune_model(args.config, model_name)
    else:
        logger.error("Unknown task")

if __name__ == "__main__":
    main()
