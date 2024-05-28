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
            Tuple[MNIST, MNIST]: Train and test datasets.
        """
        # Create the data directory if it does not exist
        os.makedirs(name=cls.dataset_directory, exist_ok=True)

        # Define data transformations
        data_transform = transforms.Compose(transforms=[
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.0,), std=(1.0,))
        ])

        # Download the dataset
        train_set = MNIST(root=cls.dataset_directory, train=True, download=True, transform=data_transform)
        test_set = MNIST(root=cls.dataset_directory, train=False, download=True)

        return train_set, test_set

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
