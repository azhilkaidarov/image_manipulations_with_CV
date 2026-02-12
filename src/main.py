import numpy as np
from pathlib import Path
from PIL import Image

from utils import Picture, PIPELINE_CONFIG
from PipelineRunner import PipelineRunner

# Setting up I/O
input_dir = Path.cwd() / "data" / "raw"
output_dir = Path.cwd() / "data" / "output"


def main():
    choose_pic = Picture.MOUNTAINS
    inp = Image.open(f"{input_dir}/{choose_pic.value}")
    img = np.array(inp, dtype='uint8')
    
    orchestrator = PipelineRunner(PIPELINE_CONFIG)
    nimg = orchestrator.run(img)

    Image.fromarray(nimg).save(f"{output_dir}/{choose_pic.value} - through orchestrator.jpg")

if __name__ == "__main__":
    main()
