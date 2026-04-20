import random
import numpy as np
from PIL import Image
from typing import Union, Tuple

def get_width_pil(image: Image.Image) -> int:
    return image.width

def get_width_numpy(image: np.ndarray) -> int:
    return get_width_pil(Image.fromarray(image))

def get_width(image: Union[Image.Image, np.ndarray]) -> int:
    if isinstance(image, Image.Image):
        return get_width_pil(image)
    if isinstance(image, np.ndarray):
        return get_width_numpy(image)
    raise TypeError('image must be a PIL Image or numpy array')

def get_height_pil(image: Image.Image) -> int:
    return image.height

def get_height_numpy(image: np.ndarray) -> int:
    return get_height_pil(Image.fromarray(image))

def get_height(image: Union[Image.Image, np.ndarray]) -> int:
    if isinstance(image, Image.Image):
        return get_height_pil(image)
    if isinstance(image, np.ndarray):
        return get_height_numpy(image)
    raise TypeError('image must be a PIL Image or numpy array')

def get_size_pil(image: Image.Image) -> Tuple[int, int]:
    return image.size

def get_size_numpy(image: np.ndarray) -> Tuple[int, int]:
    return get_size_pil(Image.fromarray(image))

def get_size(image: Union[Image.Image, np.ndarray]) -> Tuple[int, int]:
    if isinstance(image, Image.Image):
        return get_size_pil(image)
    if isinstance(image, np.ndarray):
        return get_size_numpy(image)
    raise TypeError('image must be a PIL Image or numpy array')

def get_downscaled_size(image: Union[Image.Image, np.ndarray], scale: int) -> Tuple[int, int]:
    w, h = get_size(image)
    return w // scale, h // scale

def resize_pil(image: Image.Image, factor: Union[float, None], size: Union[Tuple[int, int], None], resample: Union[int, None]) -> Image.Image:
    if size is None:
        if factor is None:
            return image
        size = (int(image.width * factor), int(image.height * factor))

    if resample is None:
        resample = random.choices(
            population=(
                Image.Resampling.BICUBIC,
                Image.Resampling.BILINEAR,
                Image.Resampling.LANCZOS,
            ),
            weights=(0.5, 0.3, 0.2),
            k=1
        )[0]

    return image.resize(size, resample)

def resize_numpy(image: np.ndarray, factor: Union[float, None], size: Union[Tuple[int, int], None], resample: Union[int, None]) -> np.ndarray:
    return np.asarray(resize_pil(Image.fromarray(image), factor, size, resample))

def resize(image: Union[Image.Image, np.ndarray], *, factor: Union[float, None] = None, size: Union[Tuple[int, int], None] = None, resample: Union[int, None] = None) -> Union[Image.Image, np.ndarray]:
    if isinstance(image, Image.Image):
        return resize_pil(image, factor, size, resample)
    if isinstance(image, np.ndarray):
        return resize_numpy(image, factor, size, resample)
    raise TypeError('image must be a PIL Image or numpy array')

def random_resize(image: Union[Image.Image, np.ndarray], factor_range: Tuple[float, float] = (360 / 1080, 1080 / 720), up_down_keep_prob: Union[Tuple[float, float, float], None] = (0.2, 0.7, 0.1)) -> Union[Image.Image, np.ndarray]:
    if up_down_keep_prob is None or factor_range[0] >= 1 or factor_range[1] <= 1:
        factor = random.uniform(*factor_range)
        return resize(image, factor=factor)

    up_down_keep_factor = (
        random.uniform(1.0, factor_range[1]),
        random.uniform(factor_range[0], 1.0),
        1.0
    )

    factor = random.choices(up_down_keep_factor, up_down_keep_prob)[0]
    return resize(image, factor=factor)
