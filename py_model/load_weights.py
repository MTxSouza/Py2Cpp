"""
This module is used to load the weights of the model and convert it to a compatible format for C++.
"""
import torch

from py_model.block import Classifier
from py_model.trainer import Trainer


def main():
    """
    Main function to load the weights of the model and save it in a compatible format for C++.
    """
    # Load the model.
    model = Classifier(img_size=28, image_channels=1, feature_scale=16, n_classes=10, dropout=0.5)

    # Load the weights.
    model.load_state_dict(state_dict=torch.load(f="./py_model/weights.pth", map_location=Trainer.get_device()))

    # Set the model to evaluation mode.
    model.eval()

    # Convert the model to TorchScript
    scripted_model = torch.jit.script(obj=model)

    # Save the model.
    scripted_model.save("./py_model/weights.pt")


if __name__ == "__main__":
    main()
