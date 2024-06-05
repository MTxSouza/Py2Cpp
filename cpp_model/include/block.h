/*
Main header to define all the block classes for the model.
*/
#pragma once
#include <torch/torch.h>

using namespace torch;


class ConvBlockImpl : public nn::Module {

    public: // Definitions.

        nn::Conv2d conv{nullptr};
        nn::ReLU relu{nullptr};
        nn::Dropout dp{nullptr};
        nn::MaxPool2d pool{nullptr};

    public: // Methods.

        // Builder.
        ConvBlockImpl(int in_channels, int out_channels, float dropout) {

            // Layers.
            conv = register_module("conv", nn::Conv2d(nn::Conv2dOptions(in_channels, out_channels, 5).padding(2)));
            relu = register_module("relu", nn::ReLU());
            dp = register_module("dp", nn::Dropout(nn::DropoutOptions(dropout)));
            pool = register_module("pool", nn::MaxPool2d(nn::MaxPool2dOptions({2, 2}).stride(2)));

        }

        // Forward Pass.
        torch::Tensor forward(torch::Tensor x) {
            x = conv->forward(x);
            x = relu->forward(x);
            x = dp->forward(x);
            x = pool->forward(x);
            return x;
        }
};

// Define the ConvBlock module.
TORCH_MODULE(ConvBlock);

class ClassifierImpl : public nn::Module {

    public: // Definitions.
    
        ConvBlock conv1{nullptr}, conv2{nullptr};
        nn::Linear fc{nullptr};
        int imgFeatures;

    public:

        // Builder.
        ClassifierImpl(int img_size, int image_channels, int feature_scale, int n_classes, float dropout) {
                
                // Convolutional Blocks.
                conv1 = register_module("conv1", ConvBlock(image_channels, feature_scale, dropout));
                conv2 = register_module("conv2", ConvBlock(feature_scale, feature_scale * 2, dropout));
    
                // Fully Connected Layer.
                int imgFeatures = ((img_size / 4) * (img_size / 4)) * feature_scale * 2;
                fc = register_module("fc", nn::Linear(imgFeatures, n_classes));
        }

        // Forward method.
        torch::Tensor forward(torch::Tensor x) {
            x = conv1->forward(x);
            x = conv2->forward(x);
            x = x.view({x.size(0), -1});
            x = fc->forward(x);
            return x;
        }
};

// Define the Classifier module.
TORCH_MODULE(Classifier);
