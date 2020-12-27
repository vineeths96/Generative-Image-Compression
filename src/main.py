import torch
import datetime
from load_model import load_model
from networks import compressor, decompressor
from JPEGCompressor import jpeg_analysis
from QuantizedCompressor import quantization_analysis
from utils import plot_image, save_image, metrics


def main():
    PATH = f"../results/{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"

    generator = load_model()
    inputLatent = torch.randn(1, 512).cuda()
    original_image = generator(inputLatent).detach().clone()
    plot_image(original_image)
    save_image(original_image, PATH, 'GAN')

    image_compressed_vector = compressor(generator, original_image, PATH)
    torch.save(image_compressed_vector, f"{PATH}/ICV.pt")

    quantization_analysis(generator, image_compressed_vector.clone(), PATH)
    jpeg_analysis(original_image, PATH)


if __name__ == '__main__':
    main()
