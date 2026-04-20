import numpy as np
from PIL import Image
from typing import Union

class Processor():
    def __init__(self):
        super(Processor, self).__init__()

    def process_numpy(self, image: np.ndarray, *args, **kwds) -> np.ndarray:
        return np.asarray(self.process_pil(Image.fromarray(image), *args, **kwds))

    def process_pil(self, image: Image.Image, *args, **kwds) -> Image.Image:
        return Image.fromarray(self.process_numpy(np.asarray(image), *args, **kwds))

    def process(self, image: Union[Image.Image, np.ndarray], *args, **kwds) -> Union[Image.Image, np.ndarray]:
        if isinstance(image, Image.Image):
            return self.process_pil(image, *args, **kwds)
        if isinstance(image, np.ndarray):
            return self.process_numpy(image, *args, **kwds)
        raise TypeError('image must be a PIL Image or numpy array')

    def __call__(self, image: Union[Image.Image, np.ndarray], *args, **kwds) -> Union[Image.Image, np.ndarray]:
        return self.process(image, *args, **kwds)
    