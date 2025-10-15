import torch
import torch.nn as nn

import numpy as np

import copy
import tqdm

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from IPython.display import clear_output 

from data.dataloaders_related import get_mnist_dataloaders

#losses
from loss_and_metrics.loss import fff_loss, mmd_inverse_multi_quadratic
from loss_and_metrics.metrics import get_fid_for_model, get_inception_score_for_model

def test_model(model, test_loader, device='cpu'):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device), extract_layer=20)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

def train_model_mnist(model,
                      model_name,
                      digit=None,
                      conditional_training=False,
                      pixelwidth=28,
                      batchsize=1000,
                      epochs=20,
                      lr=0.001,
                      device='cpu',
                      calculate_mmd=False,
                      calculate_fid=False,
                      calculate_is=False,
                      calculate_logprobs=False,
                      trial=None,
                      beta_r=None,
                      beta_a=None,
                      dynamic_plot=True,
                      dequantization=None,
                      ridge=None,
                      weight_decay=0.,
                      lr_schedule=False,
                      zero_one_range=False,
                      #pca_instance=None,
                      pca_instance:PCA=None,
                      reconstruction_error_from_pca=False,
                      check_acfff_classification_quality=True):

    assert model_name in ["inn", "fff", "afff", "freia"]
    if model_name in ["fff", "afff"]:
        assert beta_r is not None
        if model_name == "afff":
            assert beta_a is not None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_schedule:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, patience=50)


    train_loader, test_loader = get_mnist_dataloaders(batchsize=batchsize,
                                                      pixelwidth=pixelwidth,
                                                      digit=digit,
                                                      zero_one_range=zero_one_range)

    metrics = {}
    loss_history = []
    mmd_history = []
    mmd_epochs = []
    fid_history = []
    fid_epochs = []
    is_history = []
    is_epochs = []
    lp_history = []
    lp_epochs = []
    acfff_quality_history = []
    acfff_quality_epochs = []
    model_screenshots = []

    for epoch in tqdm.tqdm(range(epochs)):
        optimizer.zero_grad()
        model.train()

        batch, labels = next(iter(train_loader))

        if pca_instance is not None:
            original_images = copy.deepcopy(batch)
            batch = pca_instance.transform(batch)
            batch = torch.Tensor(batch)

        batch, labels = batch.to(device), labels.to(device)
        if conditional_training:
            cond = torch.nn.functional.one_hot(labels, num_classes=10).to(device)
        else:
            cond = None

        if dequantization is not None:
                batch = batch + torch.randn_like(batch) * dequantization

        if model_name == "inn":
            # z, ljd = model(batch)
            # loss = torch.mean((0.5*torch.sum(z**2, dim=-1)-ljd+0.5*z.shape[1]*np.log(2*math.pi)), dim=0)
            # loss = loss / batch.shape[1]
            logprobs = model.logprob(batch)
            loss = - logprobs.mean() / batch.shape[1]

        if model_name == "freia":
            # if digit is None:
            #     labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=10).to(device)
            # else:
            #     labels_one_hot = torch.zeros(batch.shape[0], 10).to(device)
            z, ljd = model(batch, c=[cond,])

            logprobs = model.logprob(batch)
            loss = - logprobs.mean() / batch.shape[1]

        elif model_name == "fff":
            if (pca_instance is not None) and (not reconstruction_error_from_pca):
                loss = fff_loss(batch, encode=model.encoder, decode=model.decoder, beta=beta_r, cond=cond).loss
                latents = model(batch)
                rec_pca = model.decoder(latents)
                #rec_images = pca.inverse_transform(rec_pca)
                rec_images = pca_instance.inverse_transform(rec_pca)
                rec_term = torch.sum((original_images - rec_images)**2, dim=-1)
                loss = loss + beta_r * rec_term
            else:
                loss = fff_loss(batch, encode=model.encoder, decode=model.decoder,
                                beta=beta_r, cond=cond).loss

            loss = loss.mean()

            if beta_a is not None:
                z = model.encoder(batch, cond=torch.zeros_like(cond))
                pred_lp = z[..., -10:]
                target_lp = acfff_prob_teacher(batch, extract_layer=20) # acfff_prob_teacher = ImprovedCNN().to(device) (WHERE DO I FIX THIS)
                target_lp = torch.softmax(target_lp, dim=-1)
                mse_loss = 0.5 * torch.sum((torch.exp(pred_lp)-torch.exp(target_lp))**2, dim=-1)
                loss = loss + beta_a * mse_loss.mean()

        loss.backward()
        loss_history.append(loss.item())

        optimizer.step()
        if lr_schedule:
            scheduler.step(loss.item())


        if (((epoch+1)%200 == 0) or (epoch == 5)):
            model.eval()
            if calculate_mmd:
                mmd_history.append(mmd_inverse_multi_quadratic(batch, model.sample(batchsize)[...,:batch.shape[-1]]).item())
                mmd_epochs.append(epoch)
            if calculate_fid:
                if model_name == "freia":
                    verbose = True
                else: verbose = False
                fid_history.append(get_fid_for_model(model, digit=digit, verbose=verbose, cond=cond, batchsize=batchsize))
                fid_epochs.append(epoch)
            if check_acfff_classification_quality:
                probs_per_class = check_acfff_classification_output(model)
                acfff_quality_history.append(np.mean(probs_per_class))
                acfff_quality_epochs.append(epoch)

        if (epoch+1)%10 == 0:
            model.eval()
            if calculate_is:
                is_history.append(get_inception_score_for_model(model, batchsize=batchsize))
                is_epochs.append(epoch)
            if calculate_logprobs:
                lp_history.append(model.logprob(batch).cpu().detach().numpy().mean())
                lp_epochs.append(epoch)

        if ((epoch+1) % (epochs//5))==0:
            model_screenshots.append(copy.deepcopy(model))
        if trial is not None: trial.report(loss.item(), epoch)

        if dynamic_plot and (epoch+1)%5 == 0:
            fig, axs = plt.subplots(1,2,figsize=(8,4))
            axs = axs.flatten()
            axs[0].set_xlabel("Epochs")
            axs[0].set_ylabel("Loss")
            clear_output(wait=True)

            lines = []

            axs[0].set_ylim([min(loss_history[1:]), max(loss_history[len(loss_history)//2:])])
            lines.append(axs[0].plot(loss_history, label="Loss", color="orange")[0])

            if calculate_mmd:
                mmd_ax = axs[0].twinx()
                lines.append(mmd_ax.plot(mmd_epochs, mmd_history, label="MMD", color="blue")[0])
                mmd_ax.set_ylabel("MMD")
                mmd_ax.set_yscale("log")

            if calculate_fid:
                fid_ax = axs[0].twinx()
                lines.append(fid_ax.plot(fid_epochs, fid_history, label="FID", color="red")[0])
                fid_ax.set_ylabel("FID")
                fid_ax.set_yscale("log")

            if calculate_is:
                is_ax = axs[0].twinx()
                lines.append(fid_ax.plot(is_epochs, is_history, label="IS", color="green")[0])
                is_ax.set_ylabel("IS")

            if calculate_logprobs:
                lp_ax = axs[0].twinx()
                lines.append(lp_ax.plot(lp_epochs, lp_history, label="Logprob", color="black")[0])
                lp_ax.set_ylabel("Logprob")

            if check_acfff_classification_quality:
                acfff_ax = axs[0].twinx()
                lines.append(acfff_ax.plot(acfff_quality_epochs, acfff_quality_history, label="ACFFF", color="purple")[0])
                acfff_ax.set_ylabel("ACFFF")

            axlabels = [line.get_label() for line in lines]
            axs[0].legend(lines, axlabels)
            if cond is not None:
              sample = model.sample(1, cond[0].reshape((1, -1))).detach().cpu().numpy()
            else:
              sample = model.sample(1).detach().cpu().numpy()
            if pca_instance is not None:
              sample = pca_instance.inverse_transform(sample)
            sample = sample.reshape((pixelwidth, pixelwidth))
            if not zero_one_range:
                axs[1].matshow(sample, vmin=-0.447, vmax=2.852)
            else:
                axs[1].matshow(sample, vmin=0, vmax=1)
            axs[1].set_xticks([])
            axs[1].set_yticks([])
            if conditional_training:
                axs[1].set_title(f"This should be a {labels[0]}")

            plt.tight_layout()

            plt.show()

    clear_output()
    metrics["loss"] = loss_history
    if calculate_mmd: metrics["mmd"] = (mmd_epochs, mmd_history)
    if calculate_fid: metrics["fid"] = (fid_epochs, fid_history)
    if check_acfff_classification_quality: metrics["acfff"] = (acfff_quality_epochs, acfff_quality_history)

    return model, metrics, model_screenshots