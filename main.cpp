#include <torch/extension.h>

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Naive Cuda Flash Attention 1 Forward");
}
