import os
import torchvision
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def plot_image(images):
    grid = torchvision.utils.make_grid(images.clamp(min=-1, max=1), scale_each=True, normalize=True)
    grid_image = grid.permute(1, 2, 0).cpu().numpy()
    plt.imshow(grid_image)
    plt.show()


def save_image(images, path, mode, iteration=None):
    GAN_PATH = f'{path}/{mode}'
    os.makedirs(GAN_PATH, exist_ok=True)

    grid = torchvision.utils.make_grid(images.clamp(min=-1, max=1), scale_each=True, normalize=True)
    grid_image = grid.permute(1, 2, 0).cpu().numpy()
    plt.imshow(grid_image)

    if iteration:
        plt.imsave(f'{GAN_PATH}/image_{iteration}.png', grid_image)
    else:
        plt.imsave(f'{GAN_PATH}/original_image.png', grid_image)


def metrics(firstImage, secondImage):
    firstImage = firstImage.numpy()[0]
    secondImage = secondImage.numpy()[0]

    firstImage = np.moveaxis(firstImage, 0, 2)
    secondImage = np.moveaxis(secondImage, 0, 2)

    ssim = structural_similarity(firstImage, secondImage, data_range=firstImage.max() - firstImage.min(), multichannel=True)
    psnr = peak_signal_noise_ratio(firstImage, secondImage, data_range=firstImage.max() - firstImage.min())

    image_metrics = {'SSIM': ssim, 'PSNR': psnr}

    return image_metrics
