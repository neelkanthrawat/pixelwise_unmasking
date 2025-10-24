import torch
import torch.nn as nn

import numpy as np
import scipy
from data.dataloaders_related import get_mnist_dataloaders

# fid_feature_extractor and zero_fid_feature_extractor  is not defined below. we  need to fix that thing
# device is  also not defined

def calculate_fid(real_batch, synthetic_batch, feature_extractor, extract_layer=2, normalize_features=False, verbose=False):
    feature_extractor.eval()  # Set model to evaluation mode

    # Extract features for real and synthetic batches
    with torch.no_grad():
        real_features = feature_extractor(real_batch, extract_layer=extract_layer)
        synthetic_features = feature_extractor(synthetic_batch, extract_layer=extract_layer)

    if normalize_features:
        real_features = real_features / torch.std(real_features, dim=1, keepdim=True)
        synthetic_features = synthetic_features / torch.std(synthetic_features, dim=1, keepdim=True)

    assert not torch.isnan(real_features).any(), "NaNs found in real_features"
    assert not torch.isnan(synthetic_features).any(), "NaNs found in synthetic_features"
    assert not torch.isinf(real_features).any(), "Infs found in real_features"
    assert not torch.isinf(synthetic_features).any(), "Infs found in synthetic_features"

    # Calculate mean and covariance for both feature sets
    mu_real = real_features.mean(dim=0)
    mu_synthetic = synthetic_features.mean(dim=0)

    cov_real = torch.cov(real_features.T, correction=0)
    cov_synthetic = torch.cov(synthetic_features.T, correction=0)

    assert not torch.isnan(cov_real).any(), "NaNs found in real_cov"
    assert not torch.isnan(cov_synthetic).any(), "NaNs found in synthetic_cov"
    assert not torch.isinf(cov_real).any(), "Infs found in real_cov"
    assert not torch.isinf(cov_synthetic).any(), "Infs found in synthetic_cov"

    # Convert to numpy for FID calculation
    mu_real, mu_synthetic = mu_real.cpu().numpy(), mu_synthetic.cpu().numpy()
    cov_real, cov_synthetic = cov_real.cpu().numpy(), cov_synthetic.cpu().numpy()
    cov_real += 1e-6 * np.eye(cov_real.shape[0])
    cov_synthetic += 1e-6 * np.eye(cov_synthetic.shape[0])

    # Calculate Frechet Distance (FID)
    diff = mu_real - mu_synthetic
    sigma1sigma2 = cov_real.dot(cov_synthetic)
    covmean, _ = scipy.linalg.sqrtm(sigma1sigma2, disp=False)

    # Numerical stability: remove imaginary component if present
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    if verbose:
        print(f"Mean distance: {diff.dot(diff)}")
        print(f"Covariance distance: {np.trace(cov_real + cov_synthetic - 2 * covmean)}")
    fid = diff.dot(diff) + np.trace(cov_real + cov_synthetic - 2 * covmean)
    return fid

def get_fid_for_model(model,device, batchsize=1000, digit=4, extract_layer=2, verbose=False, cond=None, zero_one_range=False, zero_one_fid_feature_extractor=None, fid_feature_extractor=None):
    if zero_one_range:
      _, fid_test_loader = get_mnist_dataloaders(batchsize=batchsize, pixelwidth=28, digit=digit, zero_one_range=True)
    else:
      _, fid_test_loader = get_mnist_dataloaders(batchsize=batchsize, pixelwidth=28, digit=digit)
    real_batch, _ = next(iter(fid_test_loader))
    real_batch = real_batch.to(device)

    model_batch = model.sample(batchsize, cond=cond)  # Replace with generated synthetic data

    # Calculate FID score
    if zero_one_range: #Z.B. zero_one_fid_feature_extractor = ImprovedCNN().to(device): THIS IS HOW HE IMPLEMENTED IT
      assert zero_one_fid_feature_extractor.is_trained, "Feature extractor is not trained"
      fid_score = calculate_fid(real_batch, model_batch, zero_one_fid_feature_extractor, extract_layer=extract_layer, verbose=verbose)
    else: 
      assert fid_feature_extractor.is_trained, "Feature extractor is not trained"
      fid_score = calculate_fid(real_batch, model_batch, fid_feature_extractor, extract_layer=extract_layer, verbose=verbose)
    return fid_score

def entropy_of_batch_of_probs(probs):
    return -torch.sum(probs * torch.log(probs), axis=1)

def get_inception_score_for_model(model, batchsize, fid_feature_extractor, std=False):
    samples = model.sample(batchsize)
    predictions = fid_feature_extractor(samples, extract_layer=20)
    predictions = nn.functional.softmax(predictions, dim=1)
    if std:
        return torch.mean(entropy_of_batch_of_probs(predictions)).cpu().detach().numpy(), torch.std(entropy_of_batch_of_probs(predictions)).cpu().detach().numpy()
    else: return torch.mean(entropy_of_batch_of_probs(predictions)).cpu().detach().numpy()
