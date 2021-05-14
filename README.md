 ![Language](https://img.shields.io/badge/language-python--3.8.3-blue) [![Contributors][contributors-shield]][contributors-url] [![Forks][forks-shield]][forks-url] [![Stargazers][stars-shield]][stars-url] [![Issues][issues-shield]][issues-url] [![MIT License][license-shield]][license-url] [![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/vineeths96/Generative-Image-Compression">
    <img src="docs/readme/gan.png" alt="Logo" width="300" height="200">
  </a>
  <h3 align="center">Generative Image Compression</h3>
  <p align="center">
    Image Compression using GANs
    <br />
    <a href=https://github.com/vineeths96/Generative-Image-Compression><strong>Explore the repository»</strong></a>
    <br />
    <a href=https://github.com/vineeths96/Generative-Image-Compression/blob/master/docs/report.pdf>View Report</a>
  </p>

</p>

> tags : image compression, gans, generative networks, celeba, deep learning, pytorch 



<!-- ABOUT THE PROJECT -->

## About The Project

In the modern world, a huge amount of data is being sent across the internet every day. Efficient data compression and communication protocols are of great research interest today. We focus on the compression of images and video (sequence of image frames) using deep generative models and show
that they achieve a better performance in compression ratio and perceptual quality. We explore the use of GANs for this task. We know that generative models such as GANs and VAE can reproduce an image from its latent vector. We ask the question whether we can go the other direction — from an image to a latent vector. Research, though limited, has shown that these types of methods are quite effective and efficient. We further employ lossy compression and use Elias coding to reduce the vector size. A detailed description of quantization algorithms and analysis of the results are available in the [Report](./docs/report.pdf).



### Built With
This project was built with 

* python v3.8.5
* PyTorch v1.7
* The environment used for developing this project is available at [environment.yml](environment.yml).



<!-- GETTING STARTED -->

## Getting Started

Clone the repository into a local machine and enter the [src](src) directory using

```shell
git clone https://github.com/vineeths96/Generative-Image-Compression
cd Generative-Image-Compression/src
```

### Prerequisites

Create a new conda environment and install all the libraries by running the following command

```shell
conda env create -f environment.yml
```

The project uses a pretrained Progressive GAN trained on CelebA-HQ dataset from Facebook Research which gets automatically downloaded. 

### Instructions to run

To train (compress the images) the model run,

```sh
python main.py --train-model True
```

This generates a folder in the `results` directory for each run. The generated folder contains the compressed images using different schemes and bit rates.

To evaluate the model on the compressed images run,

```sh
python main.py 
```

This calculates the average PSNR and SSIM values across different runs, and generates `avg_psnr.txt` and `avg_ssim.txt` in the `results` directory.



## Model overview

The architecture of the model is shown below. We freeze the GAN model and optimize for the best latent vector using gradient descent.

![Transformer](./docs/readme/model.jpg)



<!-- RESULTS -->

## Results

We evaluate the models on the Structural Similarity Index (SSIM) and Peak Signal to Noise Ratio (PSNR) between the original image and reconstructed image. More detailed results and inferences are available in report [here](./docs/report.pdf).

| Compression method | SSIM | PSNR | CR |
| :------------------------------------------: | :-----------------: | :------------------------------------------: | :------------------------------------------: |
| GAN        | 0.79 | 26.48 | 176 × |
| Lossy compression 8bits | 0.77 | 25.06 | 412.5 × |
| Lossy compression 6bits | 0.67 | 25.06 |  495 ×  |
| Lossy compression 4bits | 0.46 | 25.06 |  559 ×  |
|     JPEG Quality 1%     | 0.51 | 19.96 | 99 × |

Figure below compares the image quality of reconstruction for a sample image for different schemes.

![Transformer](./docs/readme/results.jpg)

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Vineeth S - vs96codes@gmail.com

Project Link: [https://github.com/vineeths96/Generative-Image-Compression](https://github.com/vineeths96/Generative-Image-Compression)






<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/vineeths96/Generative-Image-Compression.svg?style=flat-square
[contributors-url]: https://github.com/vineeths96/Generative-Image-Compression/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/vineeths96/Generative-Image-Compression.svg?style=flat-square
[forks-url]: https://github.com/vineeths96/Generative-Image-Compression/network/members
[stars-shield]: https://img.shields.io/github/stars/vineeths96/Generative-Image-Compression.svg?style=flat-square
[stars-url]: https://github.com/vineeths96/Generative-Image-Compression/stargazers
[issues-shield]: https://img.shields.io/github/issues/vineeths96/Generative-Image-Compression.svg?style=flat-square
[issues-url]: https://github.com/vineeths96/Generative-Image-Compression/issues
[license-shield]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://github.com/vineeths96/Generative-Image-Compression/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/vineeths

