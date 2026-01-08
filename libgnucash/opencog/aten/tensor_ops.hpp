/*
 * opencog/aten/tensor_ops.hpp
 *
 * Advanced tensor operations for financial computations
 *
 * Copyright (C) 2024 GnuCash Developers
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef GNC_ATEN_TENSOR_OPS_HPP
#define GNC_ATEN_TENSOR_OPS_HPP

#include "tensor.hpp"
#include <random>

namespace gnc {
namespace aten {
namespace ops {

/**
 * Concatenate tensors along a dimension.
 */
template<typename T>
Tensor<T> cat(const std::vector<Tensor<T>>& tensors, size_t dim = 0)
{
    if (tensors.empty())
        throw std::invalid_argument("Cannot concatenate empty tensor list");

    // Calculate output shape
    Shape result_shape = tensors[0].shape();
    for (size_t i = 1; i < tensors.size(); ++i) {
        const auto& t = tensors[i];
        for (size_t d = 0; d < result_shape.size(); ++d) {
            if (d == dim) {
                result_shape[d] += t.shape()[d];
            } else if (result_shape[d] != t.shape()[d]) {
                throw std::invalid_argument("Shape mismatch in concatenation");
            }
        }
    }

    Tensor<T> result(result_shape);

    // Copy data (simplified for 1D/2D)
    size_t offset = 0;
    if (dim == 0 && result_shape.size() == 1) {
        for (const auto& t : tensors) {
            for (size_t i = 0; i < t.size(); ++i)
                result[offset + i] = t[i];
            offset += t.size();
        }
    }

    return result;
}

/**
 * Stack tensors along a new dimension.
 */
template<typename T>
Tensor<T> stack(const std::vector<Tensor<T>>& tensors, size_t dim = 0)
{
    if (tensors.empty())
        throw std::invalid_argument("Cannot stack empty tensor list");

    Shape result_shape = tensors[0].shape();
    result_shape.insert(result_shape.begin() + dim, tensors.size());

    Tensor<T> result(result_shape);

    // Copy data (simplified)
    size_t tensor_size = tensors[0].size();
    for (size_t i = 0; i < tensors.size(); ++i) {
        for (size_t j = 0; j < tensor_size; ++j) {
            result[i * tensor_size + j] = tensors[i][j];
        }
    }

    return result;
}

/**
 * Split a tensor into chunks.
 */
template<typename T>
std::vector<Tensor<T>> split(const Tensor<T>& tensor, size_t chunks, size_t dim = 0)
{
    std::vector<Tensor<T>> result;
    size_t chunk_size = tensor.shape()[dim] / chunks;

    // Simplified for 1D
    if (tensor.ndim() == 1) {
        for (size_t i = 0; i < chunks; ++i) {
            std::vector<T> data(chunk_size);
            for (size_t j = 0; j < chunk_size; ++j)
                data[j] = tensor[i * chunk_size + j];
            result.emplace_back(Shape{chunk_size}, std::move(data));
        }
    }

    return result;
}

/**
 * Apply softmax normalization.
 */
template<typename T>
Tensor<T> softmax(const Tensor<T>& tensor, size_t dim = 0)
{
    // Find max for numerical stability
    T max_val = tensor.max();

    Tensor<T> exp_tensor = (tensor - max_val).exp();
    T sum = exp_tensor.sum();

    return exp_tensor / sum;
}

/**
 * Apply sigmoid function.
 */
template<typename T>
Tensor<T> sigmoid(const Tensor<T>& tensor)
{
    Tensor<T> result(tensor.shape());
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = T{1} / (T{1} + std::exp(-tensor[i]));
    }
    return result;
}

/**
 * Apply ReLU function.
 */
template<typename T>
Tensor<T> relu(const Tensor<T>& tensor)
{
    return tensor.clamp(T{0}, std::numeric_limits<T>::max());
}

/**
 * Apply tanh function.
 */
template<typename T>
Tensor<T> tanh(const Tensor<T>& tensor)
{
    Tensor<T> result(tensor.shape());
    for (size_t i = 0; i < tensor.size(); ++i) {
        result[i] = std::tanh(tensor[i]);
    }
    return result;
}

/**
 * Normalize tensor to have zero mean and unit variance.
 */
template<typename T>
Tensor<T> normalize(const Tensor<T>& tensor)
{
    T m = tensor.mean();
    T s = tensor.std();
    if (s == T{0}) s = T{1};
    return (tensor - m) / s;
}

/**
 * Min-max scaling to [0, 1].
 */
template<typename T>
Tensor<T> min_max_scale(const Tensor<T>& tensor)
{
    T min_val = tensor.min();
    T max_val = tensor.max();
    T range = max_val - min_val;
    if (range == T{0}) range = T{1};
    return (tensor - min_val) / range;
}

/**
 * Batch matrix multiplication.
 */
template<typename T>
Tensor<T> bmm(const Tensor<T>& a, const Tensor<T>& b)
{
    if (a.ndim() != 3 || b.ndim() != 3)
        throw std::invalid_argument("bmm requires 3D tensors");
    if (a.shape()[0] != b.shape()[0])
        throw std::invalid_argument("Batch size mismatch");

    size_t batch = a.shape()[0];
    size_t m = a.shape()[1];
    size_t k = a.shape()[2];
    size_t n = b.shape()[2];

    Tensor<T> result({batch, m, n});

    for (size_t b_idx = 0; b_idx < batch; ++b_idx) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                T sum = T{0};
                for (size_t l = 0; l < k; ++l) {
                    sum += a.at({b_idx, i, l}) * b.at({b_idx, l, j});
                }
                result.at({b_idx, i, j}) = sum;
            }
        }
    }

    return result;
}

/**
 * Cosine similarity between tensors.
 */
template<typename T>
T cosine_similarity(const Tensor<T>& a, const Tensor<T>& b)
{
    T dot = a.flatten().dot(b.flatten());
    T norm_a = a.pow(2).sum();
    T norm_b = b.pow(2).sum();
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

/**
 * Euclidean distance between tensors.
 */
template<typename T>
T euclidean_distance(const Tensor<T>& a, const Tensor<T>& b)
{
    return (a - b).pow(2).sum();
}

/**
 * Moving average along a dimension.
 */
template<typename T>
Tensor<T> moving_average(const Tensor<T>& tensor, size_t window)
{
    if (tensor.ndim() != 1)
        throw std::invalid_argument("moving_average requires 1D tensor");

    size_t n = tensor.size();
    if (window > n) window = n;

    std::vector<T> result(n - window + 1);
    T sum = T{0};

    // Initial window
    for (size_t i = 0; i < window; ++i)
        sum += tensor[i];
    result[0] = sum / static_cast<T>(window);

    // Sliding window
    for (size_t i = window; i < n; ++i) {
        sum += tensor[i] - tensor[i - window];
        result[i - window + 1] = sum / static_cast<T>(window);
    }

    return Tensor<T>({result.size()}, std::move(result));
}

/**
 * Exponential moving average.
 */
template<typename T>
Tensor<T> exponential_moving_average(const Tensor<T>& tensor, T alpha)
{
    if (tensor.ndim() != 1)
        throw std::invalid_argument("ema requires 1D tensor");

    std::vector<T> result(tensor.size());
    result[0] = tensor[0];

    for (size_t i = 1; i < tensor.size(); ++i) {
        result[i] = alpha * tensor[i] + (T{1} - alpha) * result[i-1];
    }

    return Tensor<T>({result.size()}, std::move(result));
}

/**
 * Generate random tensor with uniform distribution.
 */
template<typename T>
Tensor<T> rand(const Shape& shape, T low = T{0}, T high = T{1})
{
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<T> dist(low, high);

    Tensor<T> result(shape);
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = dist(gen);
    return result;
}

/**
 * Generate random tensor with normal distribution.
 */
template<typename T>
Tensor<T> randn(const Shape& shape, T mean = T{0}, T stddev = T{1})
{
    static std::mt19937 gen(std::random_device{}());
    std::normal_distribution<T> dist(mean, stddev);

    Tensor<T> result(shape);
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = dist(gen);
    return result;
}

} // namespace ops
} // namespace aten
} // namespace gnc

#endif // GNC_ATEN_TENSOR_OPS_HPP
