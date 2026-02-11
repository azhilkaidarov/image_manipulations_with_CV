from tokenize import maybe
import numpy as np
from pathlib import Path
from PIL import Image

from src.utils import Picture

from src.manipulations import Manipulation, ImageManipulation

# Setting up I/O
input_dir = Path.cwd() / "data" / "raw"
output_dir = Path.cwd() / "data" / "output"


def main():
    choose_pic = Picture.MOUNTAINS
    inp = Image.open(f"{input_dir}/{choose_pic.value}")
    img = np.array(inp, dtype='uint8')
    h, w, _ = img.shape
    im = ImageManipulation()

    mirrored = im.apply(Manipulation.MIRROR, img, axis=0)
    double_mirrored = im.apply(Manipulation.MIRROR, mirrored, axis=1)
    also_inverted = im.apply(Manipulation.INVERT, double_mirrored)
    and_croped = im.apply(Manipulation.CROP, also_inverted, bbox=(h // 4, w // 4, h - (h // 4), w - (w // 4)))
    later_rotated = im.apply(Manipulation.ROTATE, and_croped, angle=10)
    but_blurred = im.apply(Manipulation.BLUR, later_rotated)
    maybe_compressed = im.apply(Manipulation.COMPRESSION, but_blurred, size=224)
    end_with_permutation = im.apply(Manipulation.PERMUTATION, maybe_compressed, size = 8)
    Image.fromarray(end_with_permutation).save(f"{output_dir}/{choose_pic.value} - end_with_permutation.jpg")


if __name__ == "__main__":
    main()
