import logging
import sys

logger = logging.getLogger("server")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

# Avoid adding multiple handlers if the module is already imported
if not logger.hasHandlers():
    logger.addHandler(console_handler)