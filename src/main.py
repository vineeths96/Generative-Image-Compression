import torch
import argparse
import datetime
from load_model import load_model
from networks import compressor, decompressor
from JPEGCompressor import jpeg_analysis
from QuantizedCompressor import quantization_analysis
from calculate_metrics import calculate_metrics
from utils import plot_image, save_image, metrics


def main():
    """
    Main function to train latent vector or run evaluation metrics
    :return: None
    """

    parser = argparse.ArgumentParser(description="Train model or evaluate model (default)")
    parser.add_argument("--train-model", action="store_true", default=False)

    arg_parser = parser.parse_args()

    if arg_parser.train_model:
        PATH = f"../results/{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"

        generator = load_model()
        inputLatent = torch.randn(1, 512).cuda()
        original_image = generator(inputLatent).detach().clone()
        plot_image(original_image)
        save_image(original_image, PATH, "")

        image_compressed_vector = compressor(generator, original_image, PATH)
        torch.save(image_compressed_vector, f"{PATH}/ICV.pt")

        quantization_analysis(generator, image_compressed_vector.clone(), PATH)
        jpeg_analysis(original_image, PATH)
    else:
        calculate_metrics()


if __name__ == "__main__":
    main()
