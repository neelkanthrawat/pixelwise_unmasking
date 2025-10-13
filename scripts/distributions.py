import torch
from sklearn.datasets import make_moons
from sklearn.mixture import GaussianMixture

class MoonsGMM:
    """A Gaussian mixture model fit to the two-moons dataset."""
    def __init__(self, components=20, noise=0.1):
        x, _ = make_moons(n_samples=2000, noise=noise)
        self.gmm = GaussianMixture(n_components=components, covariance_type="full").fit(x)

    def log_prob(self, x):
        """Return log probability of each point x (torch tensor)."""
        x_np = x.detach().cpu().numpy()
        logp = self.gmm.score_samples(x_np)
        return torch.tensor(logp, dtype=torch.float32)

def moons_gmm(components=20, noise=0.1):
    return MoonsGMM(components, noise)
