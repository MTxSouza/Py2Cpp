#include <torch/torch.h>
#include <iostream>


class ConvBlock : public torch::nn::Module {
    public:
    ConvBlock(int64_t in_channels, int64_t out_channels, int64_t kernel_size) {
        conv = register_module("conv", torch::nn::Conv2d(in_channels, out_channels, kernel_size));
        bn = register_module("bn", torch::nn::BatchNorm2d(out_channels));
    }

    torch::Tensor forward(torch::Tensor x) {
        return torch::relu(bn(conv(x)));
    }

    torch::nn::Conv2d conv;
    torch::nn::BatchNorm2d bn;
};


int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}