/*
Main header to define all utility functions for data processing.
*/
#pragma once
#include <filesystem>

#include <torch/torch.h>

#include "block.h"


torch::Device getDevice(bool verbose = false) {
    // Check current device.

    // Check if CUDA is available.
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    // Display the device information.
    if (verbose) {
        if (device == torch::kCUDA)
            std::cout << "CUDA is available. Running on GPU." << std::endl;
        else
            std::cout << "CUDA is not available. Running on CPU." << std::endl;
    }

    return device;
}


bool checkFile(const char*& path) {
    // Check if the file exists.
    return std::filesystem::exists(path);
}


void loadWeights(Classifier& model, const char*& path) {
    // Load the weights of model.
    torch::load(model, path);
}
