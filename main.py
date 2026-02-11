import numpy as np
from pathlib import Path
from PIL import Image

from src.utils import Picture

from src.manipulations import Manipulation, ImageManipulation

# Setting up I/O
input_dir = Path.cwd() / "data" / "raw"
output_dir = Path.cwd() / "data" / "output"


def main():
    choose_pic = Picture.BOATS
    inp = Image.open(f"{input_dir}/{choose_pic.value}")
    img = np.array(inp, dtype='uint8')
    h, w, _ = img.shape
    im = ImageManipulation()

    mirrored = im.apply(Manipulation.MIRROR, img, axis=0)
    double_mirrored = im.apply(Manipulation.MIRROR, mirrored, axis=1)
    also_inverted = im.apply(Manipulation.INVERT, double_mirrored)
    and_croped = im.apply(Manipulation.CROP, also_inverted, bbox=(h // 4, w // 4, h - (h // 4), w - (w // 4)))
    later_rotated = im.apply(Manipulation.ROTATE, and_croped, angle=123)
    later_rotated = later_rotated.astype(np.uint8)
    Image.fromarray(later_rotated).save(f"{output_dir}/{choose_pic.value} - later_rotated.jpg")


if __name__ == "__main__":
    main()
