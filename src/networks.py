import torch
import torchvision
import torch.nn as nn
from utils import plot_image, save_image


def compressor(model, image, save_path, image_latent=None, iterations=3000, log_freq=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    latent_vector = torch.randn(1, 512, device=device)
    latent_vector = nn.Parameter(latent_vector)

    optimizer = torch.optim.SGD([latent_vector], lr=1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    loss_fn = torch.nn.MSELoss()

    for iteration in range(iterations):
        optimizer.zero_grad()
        output = model(latent_vector)
        loss = loss_fn(output, image)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if not iteration % log_freq:
            if isinstance(image_latent, torch.Tensor):
                print(iteration, "Loss:", loss, "MSE (Latent): ", torch.mean(torch.square(latent_vector - image_latent)))
            generated_img = output.clone().detach().cpu()
            plot_image(generated_img)
            save_image(generated_img, save_path, iteration + 1)

    return latent_vector.cpu().detach()


def decompressor(model, image_latent, save_path):
    output = model(image_latent)

    generated_img = output.clone().detach().cpu()
    plot_image(generated_img)
    save_image(generated_img, save_path, 9999)

    return generated_img.cpu().detach()
