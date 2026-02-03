# загрузка / сохранение (PIL, matplotlib)
import os

from PIL import Image
from core.utils import Picture

SOURCE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw") + "/"
print(type(Picture.BOATS))

def get_image(image: Picture):
    try:
        return Image.open(f"{SOURCE_PATH}{image.value}")
    except FileNotFoundError:
        print("Error: The image file was not found. Please check the file path.")
    except IOError:
        print("Error: Cannot open the image file.")
