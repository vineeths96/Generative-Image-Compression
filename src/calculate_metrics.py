import os
from skimage import io
from utils import metrics


def calculate_metrics():
    """
    Calculates the evaluation metrics for the original and reconstructed images
    :return: None
    """

    folders = os.listdir("../results")

    modes = [
        "GAN",
        "QSGD_1",
        "QSGD_2",
        "QSGD_4",
        "QSGD_6",
        "QSGD_8",
        "JPEG_1",
        "JPEG_5",
        "JPEG_10",
        "JPEG_20",
        "JPEG_40",
    ]

    SSIM = {mode: [] for mode in modes}
    PSNR = {mode: [] for mode in modes}

    # Read the first image and second image (as per the mode) and calculate
    for folder in folders:
        if folder.endswith("txt"):
            continue

        firstImage = io.imread(f"../results/{folder}/original_image.png")

        for mode in modes:
            if mode == "GAN":
                # 2976 is the index of last saved image of GAN output
                secondImage = io.imread(f"../results/{folder}/GAN/image_2976.png")
            else:
                method, ID = mode.split("_")
                if method == "QSGD":
                    secondImage = io.imread(f"../results/{folder}/QSGD/image_{ID}bits.png")
                elif method == "JPEG":
                    secondImage = io.imread(f"../results/{folder}/JPEG/image_{ID}.jpeg")

            image_metrics = metrics(firstImage, secondImage)
            SSIM[mode].append(image_metrics["SSIM"])
            PSNR[mode].append(image_metrics["PSNR"])

    with open("../results/avg_ssim.txt", "w") as file:
        for mode in SSIM.keys():
            file.write(f"SSIM for {mode} Mode : {sum(SSIM[mode]) / len(SSIM[mode])}\n")

    with open("../results/avg_psnr.txt", "w") as file:
        for mode in PSNR.keys():
            file.write(f"PSNR for {mode} Mode : {sum(PSNR[mode]) / len(PSNR[mode])}\n")
