"""
Main module where download and load the dataset and define all functions to preprocess the data.
"""
import os

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class MNISTDataset:

    dataset_directory = "./data"
    
    @classmethod
    def download_dataset(cls):
        """
        Function to download the MNIST dataset.

        Returns:
            Tuple[MNIST, MNIST, int, int, Dict[str, int]]: Train dataset, test dataset, image size, number of channels, label map.
        """
        # Create the data directory if it does not exist
        os.makedirs(name=cls.dataset_directory, exist_ok=True)

        # Define data transformations
        train_data_transform = transforms.Compose(transforms=[
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.0,), std=(1.0,))
        ])
        test_data_transform = transforms.Compose(transforms=[
            transforms.ToTensor()
        ])

        # Download the dataset
        train_set = MNIST(root=cls.dataset_directory, train=True, download=True, transform=train_data_transform)
        test_set = MNIST(root=cls.dataset_directory, train=False, download=True, transform=test_data_transform)

        # Get image size
        image_size = train_set.data.shape[-1]

        # Get number of channels
        num_channels = 1 if train_set.data.ndim == 3 else train_set.data.shape[1]

        # Get labels
        label_map = {lbl.split(sep=" - ")[-1]: idx for lbl, idx in train_set.class_to_idx.items()}

        return train_set, test_set, image_size, num_channels, label_map

    @staticmethod
    def get_data_loader(dataset, batch_size, shuffle = True):
        """
        Function to create a DataLoader from a dataset.

        Args:
            dataset (MNIST): Dataset to create the DataLoader from.
            batch_size (int): Batch size.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: DataLoader object.
        """
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
