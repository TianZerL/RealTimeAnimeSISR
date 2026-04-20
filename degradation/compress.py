import random, io
import numpy as np
from PIL import Image
from typing import Union, Tuple, Any
from pillow_heif import register_heif_opener
register_heif_opener()


def compress_image_pil(image: Image.Image, format: str, **params: Any) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format=format, **params)
    output = Image.open(buffer)
    if output.mode != image.mode:
        output = output.convert(image.mode)
    return output

def compress_image_numpy(image: np.ndarray, format: str, **params: Any)-> np.ndarray:
    return np.asarray(compress_image_pil(Image.fromarray(image), format, **params))

def compress_image(image: Union[Image.Image, np.ndarray], format: str, **params: Any) -> Union[Image.Image, np.ndarray]:
    if isinstance(image, Image.Image):
        return compress_image_pil(image, format = format, **params)
    if isinstance(image, np.ndarray):
        return compress_image_numpy(image, format = format, **params)
    raise TypeError('image must be a PIL Image or numpy array')

def jpeg(image: Union[Image.Image, np.ndarray], quality_range: Tuple[int, int] = (70, 95)) -> Union[Image.Image, np.ndarray]:
    quality = random.randint(*quality_range)
    return compress_image(image, 'JPEG', quality = quality)

def webp(image: Union[Image.Image, np.ndarray], quality_range: Tuple[int, int] = (70, 95), method_range: Tuple[int, int] = (2, 5)) -> Union[Image.Image, np.ndarray]:
    quality = random.randint(*quality_range)
    method = random.randint(*method_range)
    return compress_image(image, 'WebP', quality = quality, method = method)

def avif(image: Union[Image.Image, np.ndarray], quality_range: Tuple[int, int] = (70, 95), speed_range: Tuple[int, int] = (3, 7)) -> Union[Image.Image, np.ndarray]:
    quality = random.randint(*quality_range)
    speed = random.randint(*speed_range)
    return compress_image(image, 'AVIF', quality = quality, speed = speed)

def heif(image: Union[Image.Image, np.ndarray], quality_range: Tuple[int, int] = (70, 95)) -> Union[Image.Image, np.ndarray]:
    quality = random.randint(*quality_range)
    return compress_image(image, 'HEIF', quality = quality)

def random_compress(image: Union[Image.Image, np.ndarray], quality_range: Tuple[int, int] = (75, 95)) -> Union[Image.Image, np.ndarray]:
    """
    Apply random compression to an image using various codecs (jpeg, webp, avif, heif).

    This function simulates real-world compression artifacts commonly found in anime content
    from different sources including streaming platforms, fan encodes, and archival formats.

    Args:
        image: Input image in PIL Image or numpy array format
        quality_range: Tuple of (min_quality, max_quality) for compression

    Returns:
        Compressed image in same format as input, or None if compression fails

    Weight distribution rationale:
        - JPEG (40%): Represents traditional codecs (H.264, MPEG4, MPEG2) that are still
          widely used in streaming, legacy content, and general web distribution.

        - WebP (20%): VP9-based format commonly used by YouTube and other modern streaming
          platforms, especially for international anime distribution.

        - AVIF (10%): AV1-based next-gen format with excellent compression efficiency but
          limited current adoption in anime circles due to encoding complexity.

        - HEIF (30%): HEVC-based format representing modern fan encodes and high-quality
          releases dominate current high-quality anime distribution.
    """
    compress_func = random.choices((jpeg, webp, avif, heif), (0.4, 0.2, 0.1, 0.3))[0]
    return compress_func(image, quality_range)
