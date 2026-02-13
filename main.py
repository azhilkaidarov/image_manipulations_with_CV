import logging
import numpy as np
from pathlib import Path
from PIL import Image

from src.utils import Picture, PIPELINE_CONFIG
from src.PipelineRunner import PipelineRunner

# Setting up I/O
input_dir = Path.cwd() / "data" / "raw"
output_dir = Path.cwd() / "data" / "output"

log_file = Path("logs/app.log")
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(name)-25s [%(funcName)s:%(lineno)d] %(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("="*50)
logger.info(f"Логирование запущено. Файл лога: {log_file}")


def main():
    choose_pic = Picture.ISLAND
    inp = Image.open(f"{input_dir}/{choose_pic.value}")
    img = np.array(inp, dtype='uint8')

    orchestrator = PipelineRunner(PIPELINE_CONFIG)
    nimg = orchestrator.run(img, choose_pic.value)
    logger.info(f"Результат сохранен в ../data/output/{choose_pic.value} - through orchestrator.jpg")
    Image.fromarray(nimg).save(f"{output_dir}/{choose_pic.value} - through orchestrator.jpg")


if __name__ == "__main__":
    main()
