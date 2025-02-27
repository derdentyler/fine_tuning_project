from src.utils.logger_loader import LoggerLoader

if __name__ == "__main__":
    logger_loader = LoggerLoader()
    logger = logger_loader.get_logger()

    logger.debug("✅ Это DEBUG сообщение")
    logger.info("✅ Это INFO сообщение")
    logger.warning("⚠️  Это WARNING сообщение")
    logger.error("❌ Это ERROR сообщение")
    logger.critical("🔥 Это CRITICAL сообщение")
