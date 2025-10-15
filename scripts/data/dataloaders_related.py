import torch
from torchvision import transforms, datasets


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'

def get_mnist_dataloaders(batchsize, pixelwidth, digit=None, zero_one_range=False):
    if not zero_one_range:
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Resize((pixelwidth, pixelwidth)),
            transforms.Normalize((0.1307,), (0.3081,)),  # Normalize the images
            AddGaussianNoise(mean=0.0, std=0.01),
            transforms.Lambda(lambda x: x.view(-1))
        ])
    if zero_one_range:
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Resize((pixelwidth, pixelwidth)),
            transforms.Lambda(lambda x: x.view(-1))
        ])


    # Download and load the training data
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Download and load the test data
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    if digit is not None:
        # Filter training dataset for the specific digit
        train_indices = [i for i, label in enumerate(train_dataset.targets) if label == digit]
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)

        # Filter test dataset for the specific digit
        test_indices = [i for i, label in enumerate(test_dataset.targets) if label == digit]
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    # Create data loaders for batching and shuffling
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=True)

    return train_loader, test_loader