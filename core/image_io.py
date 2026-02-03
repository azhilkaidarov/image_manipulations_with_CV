# загрузка / сохранение (PIL, matplotlib)
import os

from PIL import Image
from utils import Picture

SOURCE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw") + "/"


try:
    img = Image.open(f"{SOURCE_PATH}{Picture.MOUNTAINS.value}")
    img.show()
except FileNotFoundError:
    print("Error: The image file was not found. Please check the file path.")
except IOError:
    print("Error: Cannot open the image file.")
