import torch
import torch.nn as nn

from data.dataloaders_related import get_mnist_dataloaders


def entropy_of_batch_of_probs(probs):
    return -torch.sum(probs * torch.log(probs), axis=1)

def get_inception_score_for_model(model, batchsize, std=False): 
    samples = model.sample(batchsize)
    predictions = fid_feature_extractor(samples, extract_layer=20)#fid_feature_extractor=ImprovedCNN().to(device) : look into it later
    predictions = nn.functional.softmax(predictions, dim=1)
    if std:
        return torch.mean(entropy_of_batch_of_probs(predictions)).cpu().detach().numpy(), torch.std(entropy_of_batch_of_probs(predictions)).cpu().detach().numpy()
    else: return torch.mean(entropy_of_batch_of_probs(predictions)).cpu().detach().numpy()
    
# function for checking density across different digits
def check_acfff_classification_output(model, batchsize=500, device='cpu'):
  digit_probs = []
  for digit in range(10):
    digit_loader, _ = get_mnist_dataloaders(batchsize=batchsize, pixelwidth=28, digit=digit) # check line 13 under cell function for dataloaders
    batch = next(iter(digit_loader))[0].to(device)
    cond = torch.zeros(batchsize, 10).to(device)
    output = model(batch, cond)
    classification_probs = output[..., -10:]
    classification_probs = torch.softmax(classification_probs, dim=-1)
    digit_prob = torch.mean(classification_probs[:, digit])
    digit_probs.append(digit_prob.item())
  return digit_probs

#  Function for checking density across different  digits
def class_density_comparison(model, pixelwidth, verbose=False, zero_one_range=False, decoder=False, pca=None, device='cpu'):
    class_densities = []
    if verbose:
        class_latents, class_ljds = [], []
    for digit in range(10):
        _, test_loader = get_mnist_dataloaders(batchsize=10,
                                               pixelwidth=pixelwidth,
                                               digit=digit,
                                               zero_one_range=zero_one_range)
        five_images = next(iter(test_loader))[0]
        if pca is not None:
            five_images = pca.transform(five_images)
            five_images = torch.Tensor(five_images).to(device)
        five_images = five_images.to(device)
        if verbose:
          if decoder:
            densities, latents, ljds = model.logprob(five_images, verbose=True, jac_of_enc=False)
          else:
            densities, latents, ljds = model.logprob(five_images, verbose=True)
          class_densities.append(np.mean(densities.detach().cpu().numpy()))
          class_latents.append(np.mean(latents.detach().cpu().numpy()))
          class_ljds.append(np.mean(ljds.detach().cpu().numpy()))
        else:
          if decoder:
            densities, latents, ljds = model.logprob(five_images, verbose=False, jac_of_enc=False)
          else:
            densities, latents, ljds = model.logprob(five_images, verbose=False)
          class_densities.append(np.mean(densities.detach().cpu().numpy()))
    if verbose: return class_densities, class_latents, class_ljds
    return class_densities

def latent_ljd_bar_plots(model, ax, title=None, verbose=True, decoder=False, zero_one_range=False, pca=None):
    if verbose:
        densities, latents, ljds = class_density_comparison(model, 28,
                                                            verbose=True,
                                                            zero_one_range=zero_one_range,
                                                            decoder=decoder,
                                                            pca=pca)
        densities = np.array(densities)
        latents = np.array(latents)
        ljds = np.array(ljds)
        print(densities.shape, latents.shape, ljds.shape)
    else:
        densities, latents, ljd = class_density_comparison(model, 28,
                                                           verbose=False,
                                                           zero_one_range=zero_one_range,
                                                           decoder=decoder,
                                                           pca=pca)

    digits = range(10)

    if verbose:
        ax.bar(digits, latents, color='orange', label='Latents')  # Bottom bar for latents
        ax.bar(digits, ljds, bottom=latents, color='blue', label='LJDs')
    else:
        ax.bar(digits, densities, color='orange', label='Densities')  # Bottom bar for latents
    ax.set_xticks(digits)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Log-Density")
    if title is not None:
        ax.set_title(title)
    ax.legend()