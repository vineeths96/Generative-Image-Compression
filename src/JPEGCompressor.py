import os
import cv2
import numpy as np
from PIL import Image


def jpeg(image, path, quality=10):
    JPEG_PATH = f'../results/{path}/JPEG'
    os.makedirs(JPEG_PATH, exist_ok=True)

    image_normalized = cv2.normalize(src=image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    image_IM_Object = Image.fromarray(image_normalized, 'RGB')
    image_IM_Object.save(f'{JPEG_PATH}/image_{quality}.jpeg', "JPEG", quality=quality)


def jpeg_analysis(image, PATH):
    image_numpy = image.cpu().numpy()[0]
    image_numpy = np.moveaxis(image_numpy, 0, 2)

    for quality in [100, 80, 60, 40, 20, 10, 5, 1]:
        jpeg(image_numpy, PATH, quality)
