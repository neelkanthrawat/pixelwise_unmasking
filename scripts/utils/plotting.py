import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from data.dataloaders_related  import get_mnist_dataloaders

# function for simple plotting
def plot_sample(model_in, pixelwidth=28, zero_one_range = True):
    plt.figure(figsize=(15,3))
    axs = axs.flatten()
    sample = model_in.sample(1)
    if zero_one_range: vmin, vamx = 0., 1.
    else: vmin, vmax = -0.447, 2.852
    plt.matshow(sample.detach().numpy().reshape((pixelwidth,pixelwidth)), vmin=vmin, vmax=vmax)
    


# function for plotting samples during training
def five_samples(model_snapshots, pixelwidth=28, zero_one_range=True):
    fig, axs = plt.subplots(1,5,figsize=(15,3))
    axs = axs.flatten()
    for i in range(5):
        sample = model_snapshots[i].sample(1)
        if zero_one_range: vmin, vamx = 0., 1.
        else: vmin, vmax = -0.447, 2.852
        axs[i].matshow(sample.detach().numpy().reshape((pixelwidth,pixelwidth)), vmin=vmin, vmax=vmax)

# Functions for five samples from same model (unconditional)
def five_samples_same_model(model, pixelwidth=28, save_path=None, zero_one_range=False):
    fig, axs = plt.subplots(1,5,figsize=(15,3))
    axs = axs.flatten()
    if zero_one_range: vmin, vmax = 0., 1.
    else: vmin, vmax = -0.447, 2.852
    for i in range(5):
        sample = model.sample(1)
        axs[i].matshow(sample.detach().cpu().numpy().reshape((pixelwidth,pixelwidth)), vmin=vmin, vmax=vmax)
        axs[i].grid(False)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=1000)

def five_samples_same_model_during_training(model_list, n_epochs, pixelwidth=28, save_path=None, vmin=-0.447, vmax=2.852):
    fig, axs = plt.subplots(5,5,figsize=(15,15))
    for i in range(5):
        for j in range(len(model_list)):
            if i == 0:
                axs[i,j].set_title(f"Epoch {(j+1)*n_epochs//5}")
            model = model_list[j]
            sample = model.sample(1)
            axs[i,j].matshow(sample.detach().cpu().numpy().reshape((pixelwidth,pixelwidth)), vmin=vmin, vmax=vmax)
            axs[i,j].grid(False)
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=1000)

# Function for plotting all digits (conditional FFF)
def plot_all_classes(model, device='cpu'):
  fig, axs = plt.subplots(3, 10, figsize=(20,6))
  for digit in range(10):
    cond = torch.nn.functional.one_hot(torch.Tensor([digit,]).long(), num_classes=10).to(device)
    for i in range(3):
      sample = model.sample(1, cond=cond).cpu().detach().numpy().reshape((28,28))
      axs[i][digit].imshow(sample, vmin=-0.447, vmax=2.852)
      axs[i][digit].set_xticks([])
      axs[i][digit].set_yticks([])

# function to evaluate  acFFF output for all digits
def check_acfff_classification_output(model, batchsize=500, device='cpu'):
  digit_probs = []
  for digit in range(10):
    digit_loader, _ = get_mnist_dataloaders(batchsize=batchsize, pixelwidth=28, digit=digit)
    batch = next(iter(digit_loader))[0].to(device)
    cond = torch.zeros(batchsize, 10).to(device)
    output = model(batch, cond)
    classification_probs = output[..., -10:]
    classification_probs = torch.softmax(classification_probs, dim=-1)
    digit_prob = torch.mean(classification_probs[:, digit])
    digit_probs.append(digit_prob.item())
  return digit_probs


