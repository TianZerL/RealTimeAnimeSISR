import random, functools
import numpy as np
from PIL import Image
from typing import Union, Tuple

from . import compress, resize


def simple_degrade(image: Union[Image.Image, np.ndarray], scale: int = 2, bgr_to_rgb: bool = False) -> Union[Image.Image, np.ndarray]:
    resize_factor = 1.0 / scale
    image = resize.resize(image, factor=resize_factor)
    image = compress.random_compress(image, quality_range=(80, 95))
    return image

def random_resize_degrade(image: Union[Image.Image, np.ndarray], scale: int = 2, bgr_to_rgb: bool = False) -> Union[Image.Image, np.ndarray]:
    resize_factor = 1.0 / scale
    return resize.resize(image, factor=resize_factor)

def bicubic_degrade(image: Union[Image.Image, np.ndarray], scale: int = 2, bgr_to_rgb: bool = False) -> Union[Image.Image, np.ndarray]:
    resize_factor = 1.0 / scale
    return resize.resize(image, factor=resize_factor, resample=Image.Resampling.BICUBIC)

def box_degrade(image: Union[Image.Image, np.ndarray], scale: int = 2, bgr_to_rgb: bool = False) -> Union[Image.Image, np.ndarray]:
    resize_factor = 1.0 / scale
    return resize.resize(image, factor=resize_factor, resample=Image.Resampling.BOX)
