"""
Main module to define the Trainer class and all useful functions to train the model.
"""
from warnings import WarningMessage

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


class EarlyStopping:

    def __init__(self, patience, delta):
        """
        Initialize the EarlyStopping class.

        Args:
            patience (int): Number of epochs to wait before early stopping.
            delta (int): Difference between the best score and the current score.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False


    def __call__(self, train_loss, valid_loss):
        """
        Function to call the EarlyStopping class.

        Args:
            train_loss (float): Training loss value.
            valid_loss (float): Validation loss value.

        Returns:
            bool: Whether to early stop or not.
        """
        if self.best_score is None or valid_loss < self.best_score:
            self.best_score = valid_loss
            self.counter = 0
        elif abs(train_loss - valid_loss) > self.delta:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


class Trainer:

    @staticmethod
    def compute_forward(model, x, no_grad = False):
        """
        Function to compute the forward pass of the model.

        Args:
            model (nn.Module): Model to compute the forward pass.
            x (Tensor): Input tensor.
            no_grad (bool): Whether to compute the forward pass without gradients. (Default: False)

        Returns:
            Tensor: Output tensor.
        """
        # Set the model to evaluation mode if no_grad is True
        if no_grad:
            model.eval()
            with torch.no_grad():
                return model(x)

        # Set the model to training mode if no_grad is False
        model.train()
        return model(x)


    @staticmethod
    def compute_accuracy(y_pred, y_true):
        """
        Function to compute the accuracy of the model.

        Args:
            y_pred (Tensor): Predicted values.
            y_true (Tensor): True values.

        Returns:
            torch.Tensor: Accuracy value.
        """
        return (y_pred.argmax(dim=1) == y_true).float().mean()


    @staticmethod
    def compute_loss(y_pred, y_true):
        """
        Function to compute the loss of the model.

        Args:
            y_pred (Tensor): Predicted values.
            y_true (Tensor): True values.

        Returns:
            torch.Tensor: Loss value.
        """
        return F.cross_entropy(input=y_pred, target=y_true)


    @staticmethod
    def run_one_iter(model, x, y, device, optim = None):
        """
        Function to run one iteration of the model.

        Args:
            model (nn.Module): Model to run the iteration.
            x (Tensor): Input tensor.
            y (Tensor): True values.
            device (torch.device): Device to run the model.
            optim (Optimizer): Optimizer to update the model. (Default: None)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Predicted values, loss value, accuracy value.
        """
        # Set data to the device
        x, y = x.to(device=device), y.to(device=device)

        # Compute the forward pass
        no_grad = False if optim is not None else True
        y_pred = Trainer.compute_forward(model=model, x=x, no_grad=no_grad)

        # Compute the loss
        loss = Trainer.compute_loss(y_pred=y_pred, y_true=y)

        # Compute the accuracy
        accuracy = Trainer.compute_accuracy(y_pred=y_pred, y_true=y)

        # Compute the backward pass
        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()

        return y_pred, loss, accuracy


    @staticmethod
    def save_model(model, path):
        """
        Function to save the model.

        Args:
            model (nn.Module): Model to save.
            path (str): Path to save the model.
        """
        torch.save(obj=model.state_dict(), f=path)


    @staticmethod
    def get_device():
        """
        Function to get the device.

        Returns:
            torch.device: Device to run the model.
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def __init__(self, model, train_loader, valid_loader):
        """
        Initialize the trainer class.

        Args:
            model (nn.Module): Instance of classifier model.
            train_loader (Dataloader): Training data loader.
            valid_loader (Dataloader): Validation data loader.
        """
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Training variables
        self.is_compiled = False

        self.device = None
        self.optim = None
        self.total_epochs = None
        self.early_stopping = None

        self.best_score = None


    def compile(self, total_epochs, lr = 1e-5, patience = 10, delta = 5):
        """
        Function to compile the model.

        Args:
            total_epochs (int): Total number of epochs to train the model.
            lr (float): Learning rate for the optimizer. (Default: 1e-5)
            patience (int): Number of epochs to wait before early stopping. (Default: 10)
            delta (int): Difference between the best score and the current score. (Default: 5)
        """
        # Define total number of epochs
        self.total_epochs = total_epochs

        # Initialize the optimizer
        self.optim = optim.Adam(params=self.model.parameters(), lr=lr)

        # Initialize the early stopping
        self.early_stopping = EarlyStopping(patience=patience, delta=delta)

        # Set the model to the device
        self.device = self.get_device()
        self.model.to(device=self.device)

        # Set the model to compiled
        self.is_compiled = True


    def train(self):
        """
        Function to train the model.
        """
        # Check if the model is compiled
        if not self.is_compiled:
            raise RuntimeError("Model is not compiled. Please compile the model before training.")

        # Start training
        for epoch in range(1, self.total_epochs + 1):

            # Display epoch
            print(f"Epoch: [{epoch}/{self.total_epochs}]")

            # Training loop
            train_loss = 0
            train_accuracy = 0
            train_tqdm = tqdm(iterable=self.train_loader)
            for x, y in train_tqdm:
                # Run one iteration
                _, loss, accuracy = self.run_one_iter(model=self.model, x=x, y=y, device=self.device, optim=self.optim)

                # Update the progress bar
                train_loss += loss.item()
                train_accuracy += accuracy.item()
                train_tqdm.set_postfix(loss=loss.item(), accuracy=accuracy.item())

            # Validation loop
            valid_loss = 0
            valid_accuracy = 0
            valid_tqdm = tqdm(iterable=self.valid_loader)
            for x, y in valid_tqdm:
                # Run one iteration
                _, loss, accuracy = self.run_one_iter(model=self.model, x=x, y=y, device=self.device)

                # Update the progress bar
                valid_loss += loss.item()
                valid_accuracy += accuracy.item()
                valid_tqdm.set_postfix(loss=loss.item(), accuracy=accuracy.item())

            # Compute the average loss and accuracy
            train_loss /= len(self.train_loader)
            train_accuracy /= len(self.train_loader)
            valid_loss /= len(self.valid_loader)
            valid_accuracy /= len(self.valid_loader)
            print(f"Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.6f} | Valid Loss: {valid_loss:.6f}, Valid Accuracy: {valid_accuracy:.6f}")

            # Early stopping
            if self.early_stopping(train_loss=train_loss, valid_loss=valid_loss):
                WarningMessage(message="Early stopping activated.")
                break

            # Check if the model is the best
            if self.best_score is None or valid_accuracy > self.best_score:
                self.best_score = valid_accuracy
                self.save_model(model=self.model, path="./py_model/weights.pth")
