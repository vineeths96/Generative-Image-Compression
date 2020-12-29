import os
import numpy as np
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def plot_image(images):
    """
    Plots the image provided
    :param images: Image as Torch Tensor
    :return: None
    """

    grid = torchvision.utils.make_grid(images.clamp(min=-1, max=1), scale_each=True, normalize=True)
    grid_image = grid.permute(1, 2, 0).cpu().numpy()
    plt.imshow(grid_image)
    plt.show()


def save_image(images, save_path, mode, iteration=None):
    """
    Save the image provided
    :param images: Image as a Torch tensor
    :param save_path: Path to which image is to be saved
    :param mode: Folder to which image has to be saved
    :param iteration: Optional iteration count
    :return: None
    """

    PATH = f"{save_path}/{mode}"
    os.makedirs(PATH, exist_ok=True)

    grid = torchvision.utils.make_grid(images.clamp(min=-1, max=1), scale_each=True, normalize=True)
    grid_image = grid.permute(1, 2, 0).cpu().numpy()
    plt.imshow(grid_image)

    if iteration:
        plt.imsave(f"{PATH}/image_{iteration}.png", grid_image)
    else:
        plt.imsave(f"{PATH}/original_image.png", grid_image)


def RGBA2RGB(image):
    """
    Converts an 4 channel RGBA image to 3 channel RGB image
    :param image: Image to be converted to RGB
    :return: RGB image
    """

    if image.shape[-1] == 3:
        return image

    rgba_image = Image.fromarray(image)
    rgba_image.load()
    rgb_image = Image.new("RGB", rgba_image.size, (255, 255, 255))
    rgb_image.paste(rgba_image, mask=rgba_image.split()[3])

    return np.array(rgb_image)


def metrics(firstImage, secondImage):
    """
    Calculates and returns SSIM and PSNR for a pair of images
    :param firstImage: First Image
    :param secondImage: Second Image
    :return: Metrics Dictionary
    """

    firstImage = firstImage
    secondImage = secondImage

    firstImage = RGBA2RGB(firstImage)
    secondImage = RGBA2RGB(secondImage)

    ssim = structural_similarity(
        firstImage, secondImage, data_range=firstImage.max() - firstImage.min(), multichannel=True
    )
    psnr = peak_signal_noise_ratio(firstImage, secondImage, data_range=firstImage.max() - firstImage.min())

    image_metrics = {"SSIM": ssim, "PSNR": psnr}

    return image_metrics
