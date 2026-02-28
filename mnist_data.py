import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple


def get_transforms(augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Return torchvision transforms for MNIST train and test sets.

    Parameters
    ----------
    augment : bool
        If True, apply simple data augmentation to the training data.

    Returns
    -------
    train_transform, test_transform : Tuple[Compose, Compose]
    """
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    if augment:
        train_transform = transforms.Compose(
            [
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_transform, test_transform


def get_dataloaders(
    batch_size: int = 64,
    augment: bool = True,
    num_workers: int = 2,
    root: str = "./data",
) -> Tuple[DataLoader, DataLoader]:
    """
    Create MNIST train and test DataLoaders with optional augmentation.

    Parameters
    ----------
    batch_size : int
        Batch size for both train and test loaders.
    augment : bool
        Whether to apply data augmentation to the training set.
    num_workers : int
        Number of workers for data loading.
    root : str
        Root directory where MNIST will be stored/downloaded.

    Returns
    -------
    train_loader, test_loader : Tuple[DataLoader, DataLoader]
    """
    train_transform, test_transform = get_transforms(augment=augment)

    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
) -> float:
    """
    Train the model for a single epoch on the given DataLoader.

    Returns
    -------
    avg_loss : float
        Average training loss over the epoch.
    """
    if isinstance(device, str):
        device = torch.device(device)

    model.to(device)
    model.train()

    running_loss = 0.0
    total_batches = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

    avg_loss = running_loss / max(total_batches, 1)
    return avg_loss


def evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    criterion: torch.nn.Module | None = None,
    device: torch.device | str = "cpu",
) -> Tuple[float | None, float]:
    """
    Evaluate the model on the test set.

    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate.
    test_loader : DataLoader
        DataLoader for the test set.
    criterion : nn.Module, optional
        Loss function. If provided, returns average loss as well.
    device : torch.device or str
        Device to run evaluation on.

    Returns
    -------
    avg_loss : float or None
        Average loss over the test set if criterion is provided, else None.
    accuracy : float
        Classification accuracy in range [0, 1].
    """
    if isinstance(device, str):
        device = torch.device(device)

    model.to(device)
    model.eval()

    correct = 0
    total = 0
    running_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                total_batches += 1

    accuracy = correct / max(total, 1)
    avg_loss = None
    if criterion is not None and total_batches > 0:
        avg_loss = running_loss / total_batches

    return avg_loss, accuracy

