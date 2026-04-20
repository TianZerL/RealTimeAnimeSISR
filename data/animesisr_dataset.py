import random, collections
from pathlib import Path
from PIL import Image
from torch.utils import data
from torchvision.transforms import RandomCrop
from torchvision.transforms.functional import to_tensor
from basicsr.utils.registry import DATASET_REGISTRY

import degradation, enhancement

def crop_for_scale(images, scale, random_crop_size=None):
    if not isinstance(scale, int) or scale <= 0:
        raise ValueError("scale must be a positive integer.")

    image_sequence_flag = isinstance(images, collections.abc.Sequence)

    if image_sequence_flag:
        if not images:
            raise ValueError("Empty image sequence provided.")
        original_width, original_height = images[0].size
    else:
        original_width, original_height = images.size

    left = 0
    upper = 0

    if random_crop_size:
        if isinstance(random_crop_size, int):
            target_height = target_width = random_crop_size
        else:
            if len(random_crop_size) != 2:
                raise ValueError("random_crop_size must be int or a sequence of two ints.")
            target_height, target_width = random_crop_size

        if target_height > original_height or target_width > original_width:
            raise ValueError("Invalid random crop size: larger than original image.")

        left = random.randint(left, original_width - target_width)
        upper = random.randint(upper, original_height - target_height)

        original_width, original_height = target_width, target_height

    new_width = (original_width // scale) * scale
    new_height = (original_height // scale) * scale

    if new_width == 0 or new_height == 0:
        raise ValueError(
            f"Image dimensions ({original_width}x{original_height}) are too small to be cropped to a size divisible by {scale}."
        )

    if not random_crop_size and new_width == original_width and new_height == original_height:
        return images

    left = left + (original_width - new_width) // 2
    upper = upper + (original_height - new_height) // 2
    right = left + new_width
    lower = upper + new_height

    if image_sequence_flag:
        return type(images)(image.crop((left, upper, right, lower)) for image in images)
    return images.crop((left, upper, right, lower))

@DATASET_REGISTRY.register()
class AnimeSISRPairDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.is_training = self.opt['phase'] == 'train'
        self.scale = self.opt['scale']
        self.image_mode = self.opt.get('image_mode', 'L')

        image_suffix = ['.png', '.webp', '.bmp', '.jpg', '.jpeg']

        gt_folder = self.opt['dataroot_gt']
        self.gt_path_list = [p for p in Path(gt_folder).rglob('*') if p.suffix in image_suffix]
        self.gt_path_list.sort()

        lq_folder = self.opt['dataroot_lq']
        self.lq_path_list = [p for p in Path(lq_folder).rglob('*') if p.suffix in image_suffix]
        self.lq_path_list.sort()

        assert len(self.gt_path_list) == len(self.lq_path_list)

    def transpose(self, images):
        if (transpose_method := random.choice([
            Image.Transpose.ROTATE_90,
            Image.Transpose.ROTATE_180,
            Image.Transpose.ROTATE_270,
            Image.Transpose.FLIP_TOP_BOTTOM,
            Image.Transpose.FLIP_LEFT_RIGHT,
            None
        ])) is not None:
            if isinstance(images, collections.abc.Sequence):
                return type(images)(image.transpose(transpose_method) for image in images)
            return images.transpose(transpose_method)
        else:
            return images

    def load_image(self, image_path):
        image = Image.open(str(image_path))
        if image.mode != self.image_mode:
            image = image.convert(self.image_mode)
        return image

    def __getitem__(self, index):
        gt_path = str(self.gt_path_list[index])
        lq_path = str(self.lq_path_list[index])
        gt_image = self.load_image(gt_path)
        lq_image = self.load_image(lq_path)

        if self.is_training:
            gt_image, lq_image = self.transpose((gt_image, lq_image))

        gt_tensor = to_tensor(gt_image)
        lq_tensor = to_tensor(lq_image)

        return {'lq': lq_tensor, 'gt': gt_tensor, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.gt_path_list)

@DATASET_REGISTRY.register()
class AnimeSISRDataset(data.Dataset):
    def __init__(self, opt: dict):
        super().__init__()
        self.opt = opt

        self.rcas_sharpener = None
        self.usm_sharpener = None

        self.degrade_type = self.opt['degrade_type']

        self.degrade_func = {
            "bicubic": degradation.degrade.bicubic_degrade,
            "box": degradation.degrade.box_degrade,
            "random_resize": degradation.degrade.random_resize_degrade,
            "simple": degradation.degrade.simple_degrade,
        }[self.degrade_type]

        self.is_training = self.opt['phase'] == 'train'
        self.scale = self.opt['scale']
        self.image_mode = self.opt.get('image_mode', 'L')

        image_suffix = ['.png', '.webp', '.bmp', '.jpg', '.jpeg']

        gt_folder = self.opt['dataroot_gt']
        self.gt_path_list = [p for p in Path(gt_folder).rglob('*') if p.suffix in image_suffix]
        self.gt_path_list.sort()

        self.use_enhanced_image = 'dataroot_en' in self.opt
        if self.use_enhanced_image:
            en_folder = self.opt['dataroot_en']
            self.en_path_list = [p for p in Path(en_folder).rglob('*') if p.suffix in image_suffix]
            self.en_path_list.sort()

            assert len(self.gt_path_list) == len(self.en_path_list)

        self.local_init_flag = False

    def local_init(self):
        if self.local_init_flag:
            return

        if self.rcas_sharpener is None and self.opt.get('enable_rcas_sharpening'):
            self.rcas_sharpener = enhancement.sharpen.RCASSharpener((self.opt['opencl_pid'], self.opt['opencl_did']))

        if self.usm_sharpener is None and self.opt.get('enable_usm_sharpening'):
            self.usm_sharpener = enhancement.sharpen.USMSharpener()

        self.local_init_flag = True

    def degrade(self, image):
        return self.degrade_func(image, self.scale)

    def transpose(self, images):
        if (transpose_method := random.choice([
            Image.Transpose.ROTATE_90,
            Image.Transpose.ROTATE_180,
            Image.Transpose.ROTATE_270,
            Image.Transpose.FLIP_TOP_BOTTOM,
            Image.Transpose.FLIP_LEFT_RIGHT,
            None
        ])) is not None:
            if isinstance(images, collections.abc.Sequence):
                return type(images)(image.transpose(transpose_method) for image in images)
            return images.transpose(transpose_method)
        else:
            return images

    def preprocess(self, images):
        return crop_for_scale(images=images, scale=self.scale, random_crop_size=self.opt.get('random_crop_size'))

    def load_image(self, image_path):
        image = Image.open(str(image_path))
        if image.mode != self.image_mode:
            image = image.convert(self.image_mode)
        return image

    def load_gt_data(self, index):
        gt_path = str(self.gt_path_list[index])
        gt_image = self.load_image(gt_path)

        gt_image = self.preprocess(gt_image)
        gt_image = self.transpose(gt_image)
        lq_image = self.degrade(gt_image)

        if self.usm_sharpener:
            gt_image = self.usm_sharpener(gt_image)

        if self.rcas_sharpener:
            gt_image = self.rcas_sharpener(gt_image)

        lq_tensor = to_tensor(lq_image)
        gt_tensor = to_tensor(gt_image)
        return {'lq': lq_tensor, 'gt': gt_tensor, 'lq_path': gt_path, 'gt_path': gt_path}

    def load_gt_en_data(self, index):
        gt_path = str(self.gt_path_list[index])
        en_path = str(self.en_path_list[index])
        gt_image = self.load_image(gt_path)
        en_image = self.load_image(en_path)

        gt_image, en_image = self.preprocess((gt_image, en_image))
        gt_image, en_image = self.transpose((gt_image, en_image))
        lq_image = self.degrade(gt_image)

        if self.usm_sharpener:
            en_image = self.usm_sharpener(en_image)

        if self.rcas_sharpener:
            en_image = self.rcas_sharpener(en_image)

        lq_tensor = to_tensor(lq_image)
        gt_tensor = to_tensor(gt_image)
        en_tensor = to_tensor(en_image)
        return {'lq': lq_tensor, 'gt': gt_tensor, 'en': en_tensor, 'lq_path': gt_path, 'gt_path': gt_path, 'en_path': en_path}

    def load_training_data(self, index):
        if self.use_enhanced_image:
            return self.load_gt_en_data(index)
        return self.load_gt_data(index)

    def load_validation_data(self, index):
        gt_path = str(self.gt_path_list[index])

        gt_image = self.load_image(gt_path)

        gt_image = self.preprocess(gt_image)
        lq_image = self.degrade(gt_image)

        lq_tensor = to_tensor(lq_image)
        gt_tensor = to_tensor(gt_image)

        return {'lq': lq_tensor, 'gt': gt_tensor, 'lq_path': gt_path, 'gt_path': gt_path}

    def __getitem__(self, index):
        if not self.local_init_flag:
            self.local_init()

        if self.is_training:
            return self.load_training_data(index)
        return self.load_validation_data(index)

    def __len__(self):
        return len(self.gt_path_list)
