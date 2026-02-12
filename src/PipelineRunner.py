import yaml
import numpy as np
from pathlib import Path
from manipulations import Manipulation, ImageManipulation

root_dir = Path.cwd()


class PipelineRunner:
    def __init__(self, config_path: str) -> None:
        self.config_path = f"{root_dir}/{config_path}"
        self.steps = None

    def run(self, image: np.ndarray):
        conf = self._read_config(self.config_path)

        q = []
        try:
            for step in conf['pipeline']:
                name = step['name']
                params = step.get('params', {})
                q.append((name, params))
        except Exception as e:
            print(e)

        im = ImageManipulation()
        nimg = image.copy()
        try:
            for op in q:
                name = op[0]
                params = op[1]
                manip = self._op_chooser(name)
                print(params)
                nimg = im.apply(manip, nimg, **params)
        except Exception as e:
            print(f"Problems in pipeline! {e}")

        return nimg

    def _op_chooser(self, op: str) -> Manipulation:
        if not op:
            raise ValueError("Invalid input!")

        match op:
            case 'rotate':
                return Manipulation.ROTATE
            case 'crop':
                return Manipulation.CROP
            case 'blur':
                return Manipulation.BLUR
            case 'mirror':
                return Manipulation.MIRROR
            case 'compression':
                return Manipulation.COMPRESSION
            case 'permutation':
                return Manipulation.PERMUTATION
            case 'invert':
                return Manipulation.INVERT
            case _:
                raise ValueError("Invalid manipulation name!")

    def _read_config(self, config_path: str):
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Config is not found!{e}")
