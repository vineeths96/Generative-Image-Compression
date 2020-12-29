import os
import torch
import struct
import numpy as np
from networks import compressor, decompressor
from utils import plot_image, save_image


class QuantizedCompressor:
    """
    Quantized Compressor with Elias coding.
    Code: Elias coded string is represented in 64 bit integers.
    """

    def __init__(self, device, quantization_level=8):
        self._device = device
        self._quantization_level = quantization_level
        self._sign_int_bit = 62
        self._encode_dict = self.elias_dict()

    def elias_dict(self):
        """Caching Elias codes"""

        s = 1 << self._quantization_level
        keys = set(np.arange(0, s))
        encode_dict = dict.fromkeys(keys)

        for key in encode_dict:
            encode_dict[key] = self.elias_encode(key)

        return encode_dict

    def compress(self, tensor):
        """Compress the tensors"""

        s = (1 << self._quantization_level) - 1

        norm = torch.norm(tensor)

        sign_array = torch.sign(tensor)
        sign_array *= -1
        sign_array[sign_array == -1] = 0
        sign_array = sign_array.to(dtype=torch.int8)

        l_array = torch.abs(tensor) / norm * s
        l_array_floored = l_array.to(dtype=torch.int)
        prob_array = l_array - l_array_floored
        prob_array = torch.clamp(prob_array, min=0.0, max=1.0)

        mask = torch.bernoulli(prob_array).to(torch.int)
        xi_array = l_array_floored + mask

        norm = norm / s
        code = ""
        code += self.float_to_bin(norm)

        for sign, xi in zip(sign_array, xi_array):
            code += str(sign.item())
            code += self._encode_dict[xi.item()]

        code_int_list = []
        for i in range(len(code) // self._sign_int_bit + 1):
            code_chunk = "1" + code[i * self._sign_int_bit : (i + 1) * self._sign_int_bit]
            code_int_list.append(int(code_chunk, 2))

        compressed_tensor = torch.tensor(code_int_list, dtype=torch.int64, device=self._device)
        compressed_tensor_size = torch.tensor(compressed_tensor.size(), device=self._device)

        return compressed_tensor, compressed_tensor_size

    def decompress(self, compressed_tensor, compressed_tensor_size):
        """Decompress the tensors"""

        s = (1 << self._quantization_level) - 1

        unpadded_compressed_tensor = compressed_tensor[:compressed_tensor_size]
        code_int_list = unpadded_compressed_tensor.tolist()

        code = ""
        for ind, code_int in enumerate(code_int_list):
            if ind == len(code_int_list) - 1:
                code += bin(code_int)[3:]
                continue
            code += bin(code_int)[3:].zfill(self._sign_int_bit)

        norm = self.bin_to_float(code[:32])
        code = code[32:]

        xi_list = []
        sign_list = []

        while code != "":
            sign = int(code[0])

            xi, code = self.elias_decode(code[1:])
            sign_list.append(sign)
            xi_list.append(xi)

        norm = torch.tensor(norm) / s
        sign_array = torch.tensor(sign_list)
        xi_array = torch.tensor(xi_list)

        sign_array[sign_array == 1] = -1
        sign_array[sign_array == 0] = 1

        return norm * sign_array * xi_array

    def float_to_bin(self, num):
        """Float to Binary representation"""

        return format(struct.unpack("!I", struct.pack("!f", num))[0], "032b")

    def bin_to_float(self, binary):
        """Binary to Float representation"""

        return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]

    def elias_encode(self, n):
        """Elias encoding"""

        elias_code = "0"

        while n > 1:
            binary = bin(n)[2:]
            elias_code = binary + elias_code
            n = len(binary) - 1

        return elias_code

    def elias_decode(self, elias_code):
        """Elias decoding"""

        n = 1

        while elias_code[0] != "0":
            m = int(elias_code[: n + 1], 2)
            elias_code = elias_code[n + 1 :]
            n = m

        elias_code = elias_code[1:]

        return n, elias_code


def quantization_analysis(generator, compressed_vector, PATH):
    """
    Quantization and lossy compression of a given vector. Save the reconstructed images.
    :param generator: Generator model
    :param compressed_vector: Latent vector
    :param PATH: Path where images has to be saved
    :return: None
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    QSGD_PATH = f"{PATH}/QSGD"
    os.makedirs(QSGD_PATH, exist_ok=True)

    compressed_vector = torch.tensor(compressed_vector[0], dtype=torch.float32)

    # Compress and save the vectors
    for q_bits in [1, 2, 4, 6, 8]:
        lossy_compressor = QuantizedCompressor(device=device, quantization_level=q_bits)
        quantized_compressed_vector, quantized_compressed_vector_len = lossy_compressor.compress(compressed_vector)
        torch.save(quantized_compressed_vector, f"{QSGD_PATH}/ICV_{q_bits}.pt")
        torch.save(quantized_compressed_vector_len, f"{QSGD_PATH}/ICVL_{q_bits}.pt")

    # Decompress, reconstruct and save the images
    for q_bits in [1, 2, 4, 6, 8]:
        lossy_compressor = QuantizedCompressor(device=device, quantization_level=q_bits)
        quantized_compressed_vector = torch.load(f"{QSGD_PATH}/ICV_{q_bits}.pt")
        quantized_compressed_vector_len = torch.load(f"{QSGD_PATH}/ICVL_{q_bits}.pt")

        latent_vector = lossy_compressor.decompress(quantized_compressed_vector, quantized_compressed_vector_len)
        latent_vector = latent_vector.reshape(1, -1)
        lossy_reconstructed_image = decompressor(generator, latent_vector, PATH)
        plot_image(lossy_reconstructed_image)
        save_image(lossy_reconstructed_image, PATH, "QSGD", f"{q_bits}bits")
