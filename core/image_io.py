# загрузка / сохранение (PIL, matplotlib)
import os
from PIL import Image
from enum import Enum

SOURCE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw") + "/"


class Picture(Enum):
    BOATS = "the_boats.jpg"
    DOOR = "the_door.jpg"
    ISLAND = "the_island.jpg"
    MOUNTAINS = "the_mountains.jpg"
    TREES = "the_trees.png"


try:
    img = Image.open(f"{SOURCE_PATH}{Picture.ISLAND.value}")
    img.show()
except FileNotFoundError:
    print("Error: The image file was not found. Please check the file path.")
except IOError:
    print("Error: Cannot open the image file.")
