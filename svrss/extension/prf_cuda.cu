/**
 * longrookie@202407009
 * cuda speed up
 */
#include <stdio.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <algorithm>
#include <cfloat>
#include <iostream>

template <typename scalar_t>
__global__ void prf_forward_kernel(
    const scalar_t *__restrict__ input,
    scalar_t *__restrict__ output,
    int batch_size,
    int n_channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int N,
    int kernel_size,
    int stride)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * n_channels * pooled_height * pooled_width)
    {
        return;
    }
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % n_channels;
    int n = index / pooled_width / pooled_height / n_channels;

    int h_start = ph * stride;
    int w_start = pw * stride;
    int h_end = min(h_start + kernel_size, height);
    int w_end = min(w_start + kernel_size, width);

    int h_center = h_start + kernel_size / 2;
    int w_center = w_start + kernel_size / 2;
    int idx_center = ((n * n_channels + c) * height + h_center) * width + w_center;

    for (int w = 0; w < kernel_size; ++w)
    {
        int idx = ((n * n_channels + c) * height + h_start) * width + w_start + w;
        int idx2 = ((n * n_channels + c) * height + h_end - 1) * width + w_start + w;
        output[index * N + w] = input[idx_center] - input[idx];
        output[index * N + w + 8] = input[idx_center] - input[idx2];
    }

    for (int h = 0; h < kernel_size - 2; ++h)
    {
        int idx = ((n * n_channels + c) * height + h_start + h + 1) * width + w_end - 1;
        int idx2 = ((n * n_channels + c) * height + h_start + h + 1) * width + w_start;
        output[index * N + h + 5] = input[idx_center] - input[idx];
        output[index * N + h + 13] = input[idx_center] - input[idx2];
    }
}

torch::Tensor prf_forward(
    const at::Tensor &input,
    int64_t kernel_size,
    int64_t stride)
{
    auto input_sizes = input.sizes();
    int64_t batch_size = input_sizes[0];
    int64_t n_channels = input_sizes[1];
    int64_t height = input_sizes[2];
    int64_t width = input_sizes[3];

    int64_t pooled_height = (height - kernel_size) / stride + 1;
    int64_t pooled_width = (width - kernel_size) / stride + 1;
    int N = 4 * (kernel_size - 1);
    auto output = torch::zeros({batch_size, n_channels, pooled_height, pooled_width, N}, input.options());
    int threads = 1024;
    int nums = batch_size * n_channels * pooled_height * pooled_width;
    int blocks = (nums + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "prf_forward_cuda", ([&]
                                                  { prf_forward_kernel<scalar_t><<<blocks, threads>>>(
                                                        input.data_ptr<scalar_t>(),
                                                        output.data_ptr<scalar_t>(),
                                                        batch_size,
                                                        n_channels,
                                                        height,
                                                        width,
                                                        pooled_height,
                                                        pooled_width,
                                                        N,
                                                        kernel_size,
                                                        stride); })

    );
    return output;
}

template <typename scalar_t>
__global__ void prf_backward_kernel(
    scalar_t *__restrict__ grad_output,
    scalar_t *__restrict__ grad_input,
    int batch_size,
    int n_channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int N,
    int kernel_size,
    int stride)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch_size * n_channels * height * width)
    {
        return;
    }
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % n_channels;
    int n = index / width / height / n_channels;
    scalar_t value = 0.0f;

    int padding = kernel_size / 2;
    int ph = h - padding;
    int pw = w - padding;
    if (ph >= 0 && ph < pooled_height && pw >= 0 && pw < pooled_width)
    {
        int grad_output_idx = (((n * n_channels + c) * pooled_height + ph) * pooled_width + pw) * N;
        for (int i = grad_output_idx; i < grad_output_idx + N; i++)
        {
            value += grad_output[i];
        }
    }

    if (h < pooled_height)
    {
        int min_w = pw - kernel_size / 2;
        int i_w = 0;
        for (int i = 0; i < kernel_size; i++)
        {
            i_w = min_w + i;
            int grad_output_idx = (((n * n_channels + c) * pooled_height + h - padding + kernel_size / 2) * pooled_width + i_w) * N; // 中心坐标

            if (i_w >= 0 && i_w < pooled_width)
            {
                value -= grad_output[grad_output_idx + kernel_size - 1 - i];
            }
        }
    }
    if (h >= padding + kernel_size / 2)
    {
        int min_w = pw - kernel_size / 2;
        int i_w = 0;
        for (int i = 0; i < kernel_size; i++)
        {
            i_w = min_w + i;
            int grad_output_idx = (((n * n_channels + c) * pooled_height + ph - kernel_size / 2) * pooled_width + i_w) * N;
            if (i_w >= 0 && i_w < pooled_width)
            {
                value -= grad_output[grad_output_idx + 12 - i];
            }
        }
    }

    if (w >= padding + kernel_size / 2)
    {
        int min_h = ph - kernel_size / 2 + 1;
        int i_h = 0;

        for (int i = 0; i < kernel_size - 2; i++)
        {
            i_h = min_h + i;
            int grad_output_idx = (((n * n_channels + c) * pooled_height + i_h) * pooled_width + w - padding - kernel_size / 2) * N;
            if (i_h >= 0 && i_h < pooled_height)
            {
                value -= grad_output[grad_output_idx + 7 - i];
            }
        }
    }

    if (w < pooled_width)
    {
        int min_h = ph - kernel_size / 2 + 1;
        int i_h = 0;
        for (int i = 0; i < kernel_size - 2; i++)
        {
            i_h = min_h + i;
            int grad_output_idx = (((n * n_channels + c) * pooled_height + i_h) * pooled_width + w - padding + kernel_size / 2) * N;
            if (i_h >= 0 && i_h < pooled_height)
            {
                value -= grad_output[grad_output_idx + 15 - i];
            }
        }
    }

    grad_input[index] = value;
}

torch::Tensor prf_backward(
    at::Tensor &grad_output,
    int64_t height,
    int64_t width,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t N,
    int64_t kernel_size,
    int64_t stride)
{
    auto grad_output_sizes = grad_output.sizes();
    int64_t batch_size = grad_output_sizes[0];
    int64_t n_channels = grad_output_sizes[1];
    auto grad_input = torch::zeros({batch_size, n_channels, height, width}, grad_output.options());

    int threads = 1024;
    int blocks = (grad_output.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        grad_output.scalar_type(), "prf_backward_cuda", ([&]
                                                         { prf_backward_kernel<scalar_t><<<blocks, threads>>>(
                                                               grad_output.data_ptr<scalar_t>(),
                                                               grad_input.data_ptr<scalar_t>(),
                                                               batch_size,
                                                               n_channels,
                                                               height,
                                                               width,
                                                               pooled_height,
                                                               pooled_width,
                                                               N,
                                                               kernel_size,
                                                               stride); })

    );
    return grad_input;
}