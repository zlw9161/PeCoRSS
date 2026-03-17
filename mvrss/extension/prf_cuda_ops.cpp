/**
 * longrookie@202407009
 * cuda speed up
 */

#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/ATen.h>

torch::Tensor prf_forward(
    const at::Tensor &input,
    int64_t kernel_size,
    int64_t stride);

torch::Tensor prf_backward(
    at::Tensor &grad_output,
    int64_t height,
    int64_t width,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t N,
    int64_t kernel_size,
    int64_t stride);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("prf_forward", &prf_forward, "Prf forward(CUDA) longrookie@20240709");
    m.def("prf_backward", &prf_backward, "Prf backward(CUDA) longrookie@20240912");
}