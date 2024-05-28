"""
Main module where the model is defined and trained.
"""
import argparse

import torch

from py_model.block import Classifier
from py_model.data import MNISTDataset
from py_model.trainer import Trainer


def arg_parser():
    """
    Function to parse the arguments.

    Returns:
        argparse.Namespace: Arguments.
    """
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for the optimizer.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for the model.")
    parser.add_argument("--shuffle", type=bool, default=True, help="Whether to shuffle the data.")
    parser.add_argument("--feature-scale", type=int, default=16, help="Feature scale for the model.")
    parser.add_argument("--patience", type=int, default=5, help="Number of epochs to wait before early stopping.")
    parser.add_argument("--delta", type=float, default=0.2, help="Difference between the best score and the current score.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    return parser.parse_args()


def main():
    """
    Main function to train the model.
    """
    # Parse the arguments
    args = arg_parser()

    # Define seed for reproducibility
    torch.manual_seed(args.seed)

    # Load the data
    train_dataset, valid_dataset, img_size, num_channels, label_map = MNISTDataset.download_dataset()
    train_loader = MNISTDataset.get_data_loader(dataset=train_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    valid_loader = MNISTDataset.get_data_loader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    # Initialize the model
    model = Classifier(
        img_size=img_size,
        image_channels=num_channels,
        n_classes=len(label_map),
        feature_scale=args.feature_scale,
        dropout=args.dropout
    )

    # Initialize the trainer
    trainer = Trainer(model=model, train_loader=train_loader, valid_loader=valid_loader)

    # Compile the model
    trainer.compile(
        total_epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        delta=args.delta,
        weight_decay=args.weight_decay
    )

    # Train the model
    trainer.train()


if __name__=="__main__":
    main()
