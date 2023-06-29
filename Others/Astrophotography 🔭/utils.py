from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import Dataset

from typing import Iterable, Optional, Callable
import glob


def get_manifold_image(images, im_size=(64, 64), manifold_size=(5, 5), mode='RGB'):
    """
    Creates a grid of images from a given set of images.

    Args:
        images (torch.Tensor): A tensor of images of shape (num_images, channels, height, width).
        im_size (tuple): Size of each individual image in the grid.
        manifold_size (tuple): Size of the grid.
        mode (str): Color mode of the images.

    Returns:
        PIL.Image: A PIL image containing the grid of images.
    """
    assert images.shape[0] == manifold_size[0] * manifold_size[1]

    to_pil = ToPILImage()
    pil_images = [to_pil(image) for image in images]

    dst = Image.new(mode, (im_size[0] * manifold_size[0], im_size[1] * manifold_size[1]))
    for i in range(manifold_size[0]):
        dst_line = Image.new(mode, (im_size[0], im_size[1] * manifold_size[1]))
        for j in range(manifold_size[1]):
            dst_line.paste(pil_images[i * manifold_size[0] + j], (0, im_size[1] * j))
        dst.paste(dst_line, (im_size[0] * i, 0))
    return dst


def square(im):
    """
    Converts an image to a square shape by padding or cropping.

    Args:
        im (PIL.Image): Input image.

    Returns:
        PIL.Image: Square-shaped image.
    """
    max_size = max(im.size)
    dst = Image.new('RGB', (max_size, max_size))
    dst.paste(im, (0, 0))
    return dst


def resize(im, size=Iterable):
    """
    Resizes an image to a specific size.

    Args:
        im (PIL.Image): Input image.
        size (tuple): Target size (width, height).

    Returns:
        PIL.Image: Resized image.
    """
    return im.resize(size)


class AstroDataset(Dataset):
    def __init__(self, data_dir: str, transform: Optional[Callable] = None):
        """
        Custom dataset for loading astro images.

        Args:
            data_dir (str): Path to the data directory.
            transform (Optional[Callable]): Image transformation function (default: None).
        """
        self.image_paths = glob.glob(f"{data_dir}/*.jpg")
        self.transform = transform or (lambda x: x)
        self.to_tensor = ToTensor()

    def __getitem__(self, item):
        im = Image.open(self.image_paths[item])
        im = self.transform(im)
        return self.to_tensor(im)

    def __len__(self):
        return len(self.image_paths)


def logging(log, path_to_save_dir):
    """
    Logs the given message to a file.

    Args:
        log (str): Log message.
        path_to_save_dir (str): Path to the directory where the log file will be saved.
    """
    with open(f"{path_to_save_dir}/loss.log", 'a') as logger:
        logger.write(log + "\n")
