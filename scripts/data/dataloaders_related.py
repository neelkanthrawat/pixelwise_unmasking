import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import numpy as np
from typing import Literal
import math
import torch
import matplotlib.pyplot as plt

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


### for conditional_encoder/decoder related thing
#NOTE: FLATTEN THE MASK before concatenation in the encoder:
# cond = mask.view(mask.shape[0], -1)
# x = encoder(masked_image, cond)

class MaskedMNIST(Dataset):
    def __init__(self, base_dataset, train=True, 
                 mask_type:Literal['square', 'random', 'checkerboard_pattern','mostly_masked']='square',
                 num_pixel_masked_random_case:int = 100,
                 num_pixel_left_unmasked:int = 3,
                 ):
        """
        Initialize MNIST with randomply masked areas in rectengular format

        Parameter:
            root (str): Root directory of dataset where ``MNIST/processed/training.pt``
                        and  ``MNIST/processed/test.pt`` will be saved.
            train (bool):   If True, creates dataset from ``training.pt``,
                            otherwise from ``test.pt``.
        """
        if base_dataset is None:
            raise ValueError("base_dataset must be provided")
        self.base_dataset = base_dataset
        self.mask_fns = {
                        "square": self.apply_square_mask,
                        "random": self.apply_random_mask,
                        "mostly_masked":self.apply_maximal_mask,
                        "checkerboard_pattern": ...,
                    }
        self.mask_type = mask_type
        if self.mask_type == 'random':
            self.num_pixel_masked_random = num_pixel_masked_random_case
        elif self.mask_type == 'mostly_masked':
            self.num_pixel_left_unmasked = num_pixel_left_unmasked


    def __len__(self):
        """
        Return the number of samples in dataset.
        """
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        Fetch a sample for a given index, apply a random squared mask, and return both masked and original images.
        """
        image, label = self.base_dataset[idx] 
        #print(f'shape of the image is: {image.shape}')


        # Generate a random square mask
        mask_fn = self.mask_fns[self.mask_type]
        masked_image, mask = mask_fn(image)

        # Return the masked image, original image, and label AS WELL AS THE INDICES OF THE MASK POSITION ( we need it while sending in the condition)
        #return image, masked_image, mask, label # orignal_image_which_decoder_should_reconstruct, masked_image_fed_as_input_to_encoder, condition (used in encoder and decoder)
        return masked_image, label

    def apply_square_mask(self, image):
        """
        Apply a random square mask to the image.
        """
        #print(f'image.size is {image.size(0)}')
        im_size = int(math.sqrt(image.size(0)))
        # Random mask size between 1/4 to 1/2 of image size
        mask_size = int(im_size//4) #np.random.randint(im_size // 4, im_size // 2)
        top = np.random.randint(0, im_size - mask_size)
        left = np.random.randint(0, im_size - mask_size)

        mask = torch.ones(size=(im_size, im_size))#torch.ones_like(image)
        mask[top : top + mask_size, left : left + mask_size] = 0  # Apply mask

        # masked_image = image.clone()  # Clone to not modify the original image
        image_reshaped = torch.reshape(image, shape= (im_size, im_size))
        masked_image = image_reshaped.clone()
        masked_image[top : top + mask_size, left : left + mask_size] = torch.min(masked_image)-5

        # flatten back everything
        masked_image= masked_image.view(-1)
        mask = mask.view(-1)

        return masked_image, mask
    
    def apply_random_mask(self, image):
        """  
        Apply mask at random position of image
        """
        #print(f'shape of the image is: {image.shape}')
        im_size = int(math.sqrt(image.size(0)))

        mask = torch.ones(size=(im_size, im_size))#torch.ones_like(image)
        image_reshaped = torch.reshape(image, shape= (im_size, im_size))
        masked_image = image_reshaped.clone()

        ys = np.random.randint(0, im_size, size=self.num_pixel_masked_random)
        xs = np.random.randint(0, im_size, size=self.num_pixel_masked_random)

        mask[ys, xs] = 0
        masked_image[ys, xs] = torch.min(masked_image)-5 # some negative value (super low)

        # flatten everything back
        masked_image= masked_image.view(-1)
        mask = mask.view(-1)
        
        return masked_image, mask
    
    def apply_maximal_mask(self, image):
        """  
        Apply mask at random position of image
        """
        pass
    


def get_masked_mnist_dataloaders(batchsize, pixelwidth, digit=None, zero_one_range=False,
                                 mask_type:Literal['square', 'random', 'checkerboard_pattern','mostly_masked']='square'):

    if not zero_one_range:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((pixelwidth, pixelwidth)),
            transforms.Normalize((0.1307,), (0.3081,)),
            AddGaussianNoise(mean=0.0, std=0.01),
            transforms.Lambda(lambda x: x.view(-1))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((pixelwidth, pixelwidth)),
            transforms.Lambda(lambda x: x.view(-1))
        ])

    # Base MNIST
    train_base = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_base  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Digit filter
    if digit is not None:
        train_idx = [i for i, label in enumerate(train_base.targets) if label == digit]
        test_idx  = [i for i, label in enumerate(test_base.targets) if label == digit]

        train_base = torch.utils.data.Subset(train_base, train_idx)
        test_base  = torch.utils.data.Subset(test_base, test_idx)

    # Wrap in paired dataset
    train_dataset = MaskedMNIST(train_base, mask_type=mask_type)
    test_dataset  = MaskedMNIST(test_base, mask_type=mask_type)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=True)

    return train_loader, test_loader



def show_masked_samples(train_loader, num_images=16, pixelwidth=28):
    """
    Visualize masked images returned by MaskedMNIST.
    Dataset returns (masked_image, label).
    masked_image is flattened, so we reshape it back.
    """
    # Fetch one batch
    masked_batch, labels = next(iter(train_loader))

    # Pick the first N images
    masked_batch = masked_batch[:num_images]
    labels = labels[:num_images]

    # Reshape from (batch, 784) -> (batch, 1, 28, 28)
    masked_batch = masked_batch.view(-1, 1, pixelwidth, pixelwidth)

    # Plot grid
    rows = int(num_images**0.5)
    cols = rows

    plt.figure(figsize=(8, 8))
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(masked_batch[i].squeeze().cpu(), cmap='gray')
        plt.title(f"{labels[i].item()}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
