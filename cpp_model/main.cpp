/*
Main file used to load the model and run the inference.
*/
#include <filesystem>
#include <iostream>

#include <torch/torch.h>

#include "include/block.h"
#include "include/data.h"


int main(int argc, const char* argv[]) {

    // Check the number of arguments.
    std::clog << "Checking the number of arguments..." << std::endl;
    if (argc != 2) {
        std::cerr << "Usage: ./main <path-to-weights-file>" << std::endl;
        return -1;
    }

    // Get the path to the weights file.
    const char* weightPath = argv[1];

    // Check if the file exists.
    if (!checkFile(weightPath)) {
        std::cerr << "Error: File not found." << std::endl;
        return -1;
    }

    // Check device.
    torch::Device device = getDevice();

    // Load the model.
    Classifier model = Classifier(28, 1, 16, 10, 0.5f); // Default hyperparameters.
    model->to(device);

    // Display the model architecture.
    std::clog << "Summary:" << std::endl;
    std::clog << "---------------------------------------------------------------------------------------" << std::endl;
    std::clog << model << std::endl; // Model architecture.
    std::clog << "---------------------------------------------------------------------------------------" << std::endl;

    // Load weights.
    std::clog << "Loading weights..." << std::endl;
    try {
        loadWeights(model, weightPath);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    };
};
