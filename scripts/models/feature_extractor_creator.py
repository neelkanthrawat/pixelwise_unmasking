from data.dataloaders_related import  get_mnist_dataloaders
import torch.nn as nn
import torch

def train_feature_extractor(
    model_class,
    train_loader,
    device,
    learning_rate=0.001,
    epochs=5,
    extract_layer=20,
    print_every=100
):
    """
    Train a feature extractor model on the given dataloader.

    Args:
        model_class: The model class to instantiate (e.g., ImprovedCNN)
        train_loader: DataLoader for training data
        device: Device to train on ('cpu' or 'cuda')
        learning_rate: Learning rate for the optimizer
        epochs: Number of epochs to train
        extract_layer: Layer index for feature extraction
        print_every: Number of batches after which to print training progress

    Returns:
        Trained model (with model.is_trained = True)
    """

    # Initialize model, loss function, optimizer
    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, extract_layer=extract_layer)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % print_every == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], "
                      f"Step [{batch_idx + 1}/{len(train_loader)}], "
                      f"Loss: {running_loss / print_every:.4f}")
                running_loss = 0.0

    model.is_trained = True
    return model
