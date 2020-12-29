import torch


def load_model(GAN="PGAN"):
    """
    Download and caches the GAN model into models directory
    :param GAN: Type of GAN to be downloaded
    :return: Freezed Generator of the GAN
    """

    torch.hub.set_dir("../models")

    use_gpu = True if torch.cuda.is_available() else False
    model = torch.hub.load(
        "facebookresearch/pytorch_GAN_zoo:hub", GAN, model_name="celebAHQ-512", pretrained=True, useGPU=use_gpu
    )
    generator = model.netG

    # Freeze the generator parameters
    for name, parameter in generator.named_parameters():
        parameter.requires_grad = False

    return generator
