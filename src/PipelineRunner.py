import logging
import yaml
import numpy as np
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any
from src.manipulations import Manipulation, ImageManipulation


logger = logging.getLogger(__name__)


class PipelineStep(BaseModel):
    name: str = Field(min_length=3)
    enabled: bool = True
    p: float = Field(default=1.0, ge=0.0, le=1.0)
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('name')
    @classmethod
    def check_name(cls, v: str):
        if v not in Manipulation.__members__:
            logger.error(f"Manipulation {v} is not supported!")
            raise ValueError(f"Manipulation {v} is not supported!")
        return v


class PipelineConfig(BaseModel):
    pipeline: List[PipelineStep]


class PipelineRunner:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)

        try:
            self.config = PipelineConfig(**raw_config)
            self.steps = self.config.pipeline
        except Exception as e:
            logger.error(f"Config validation error:\n{e}")
            raise

    def run(self, image: np.ndarray, image_name: str):
        logger.info("Starting image processing pipeline")

        im = ImageManipulation()
        nimg = image.copy()

        logger.info(f"Config loaded ({len(self.steps)} step(s))")
        logger.info(f"Processing image {image_name}...")
        op_names = ""

        for op in self.steps:
            name = op.name
            params = op.params

            mp = self._op_chooser(name)  # mp - manipulation
            n_img = im.apply(mp, nimg, **params)

            op_names += f"{op.name}, "

        logger.info(f"Applied operations: {op_names}")

        return n_img

    def _op_chooser(self, op: str) -> Manipulation:
        if not op:
            raise ValueError("Invalid input!")

        match op:
            case 'ROTATE':
                return Manipulation.ROTATE
            case 'CROP':
                return Manipulation.CROP
            case 'BLUR':
                return Manipulation.BLUR
            case 'MIRROR':
                return Manipulation.MIRROR
            case 'COMPERSSION':
                return Manipulation.COMPRESSION
            case 'PERMUTATION':
                return Manipulation.PERMUTATION
            case 'INVERT':
                return Manipulation.INVERT
            case _:
                raise ValueError("Invalid manipulation name!")
