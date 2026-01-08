/*
 * opencog/aten/tensor.hpp
 *
 * ATen-style Tensor library for GnuCash
 * Provides multi-dimensional tensor operations for financial computations
 *
 * Based on PyTorch ATen design principles
 *
 * Copyright (C) 2024 GnuCash Developers
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef GNC_ATEN_TENSOR_HPP
#define GNC_ATEN_TENSOR_HPP

#include <vector>
#include <memory>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <initializer_list>
#include <functional>
#include <string>
#include <sstream>

namespace gnc {
namespace aten {

/**
 * Data types supported by tensors.
 */
enum class DType
{
    Float32,
    Float64,
    Int32,
    Int64,
    Bool
};

/**
 * Device type for tensor storage.
 */
enum class DeviceType
{
    CPU,
    // Future: CUDA, OpenCL
};

/**
 * Shape of a tensor (dimensions).
 */
using Shape = std::vector<size_t>;

/**
 * Strides for tensor indexing.
 */
using Strides = std::vector<size_t>;

/**
 * TensorStorage - underlying data storage for tensors.
 * Implements reference counting for efficient memory sharing.
 */
template<typename T>
class TensorStorage
{
public:
    explicit TensorStorage(size_t size)
        : m_data(size, T{0})
    {}

    TensorStorage(size_t size, T fill_value)
        : m_data(size, fill_value)
    {}

    TensorStorage(std::vector<T>&& data)
        : m_data(std::move(data))
    {}

    TensorStorage(const std::vector<T>& data)
        : m_data(data)
    {}

    T* data() { return m_data.data(); }
    const T* data() const { return m_data.data(); }
    size_t size() const { return m_data.size(); }

    T& operator[](size_t idx) { return m_data[idx]; }
    const T& operator[](size_t idx) const { return m_data[idx]; }

private:
    std::vector<T> m_data;
};

/**
 * Tensor - multi-dimensional array for numerical computations.
 *
 * This is a simplified ATen-compatible tensor implementation optimized
 * for financial computations in GnuCash. Supports:
 * - Arbitrary dimensions
 * - Element-wise operations
 * - Matrix operations
 * - Aggregations (sum, mean, etc.)
 * - Broadcasting
 */
template<typename T = double>
class Tensor
{
public:
    using value_type = T;
    using storage_type = TensorStorage<T>;
    using storage_ptr = std::shared_ptr<storage_type>;

    // =========================================
    // Constructors
    // =========================================

    /**
     * Create an empty tensor.
     */
    Tensor() : m_shape{}, m_strides{}, m_offset(0) {}

    /**
     * Create a tensor with given shape, initialized to zeros.
     */
    explicit Tensor(const Shape& shape)
        : m_shape(shape)
        , m_strides(compute_strides(shape))
        , m_offset(0)
        , m_storage(std::make_shared<storage_type>(compute_size(shape)))
    {}

    /**
     * Create a tensor with given shape and fill value.
     */
    Tensor(const Shape& shape, T fill_value)
        : m_shape(shape)
        , m_strides(compute_strides(shape))
        , m_offset(0)
        , m_storage(std::make_shared<storage_type>(compute_size(shape), fill_value))
    {}

    /**
     * Create a tensor from data.
     */
    Tensor(const Shape& shape, std::vector<T>&& data)
        : m_shape(shape)
        , m_strides(compute_strides(shape))
        , m_offset(0)
        , m_storage(std::make_shared<storage_type>(std::move(data)))
    {
        if (m_storage->size() != compute_size(shape))
            throw std::invalid_argument("Data size doesn't match shape");
    }

    /**
     * Create a 1D tensor from initializer list.
     */
    Tensor(std::initializer_list<T> data)
        : m_shape{data.size()}
        , m_strides{1}
        , m_offset(0)
        , m_storage(std::make_shared<storage_type>(std::vector<T>(data)))
    {}

    /**
     * Create a 2D tensor from nested initializer list.
     */
    Tensor(std::initializer_list<std::initializer_list<T>> data)
    {
        size_t rows = data.size();
        size_t cols = rows > 0 ? data.begin()->size() : 0;

        m_shape = {rows, cols};
        m_strides = compute_strides(m_shape);
        m_offset = 0;

        std::vector<T> flat_data;
        flat_data.reserve(rows * cols);
        for (const auto& row : data) {
            if (row.size() != cols)
                throw std::invalid_argument("Inconsistent row sizes");
            flat_data.insert(flat_data.end(), row.begin(), row.end());
        }
        m_storage = std::make_shared<storage_type>(std::move(flat_data));
    }

    // =========================================
    // Static Factory Methods
    // =========================================

    /**
     * Create a tensor filled with zeros.
     */
    static Tensor zeros(const Shape& shape)
    {
        return Tensor(shape, T{0});
    }

    /**
     * Create a tensor filled with ones.
     */
    static Tensor ones(const Shape& shape)
    {
        return Tensor(shape, T{1});
    }

    /**
     * Create a tensor filled with a specific value.
     */
    static Tensor full(const Shape& shape, T value)
    {
        return Tensor(shape, value);
    }

    /**
     * Create an identity matrix.
     */
    static Tensor eye(size_t n)
    {
        Tensor result({n, n}, T{0});
        for (size_t i = 0; i < n; ++i)
            result.at({i, i}) = T{1};
        return result;
    }

    /**
     * Create a tensor with values from start to end.
     */
    static Tensor arange(T start, T end, T step = T{1})
    {
        std::vector<T> data;
        for (T v = start; v < end; v += step)
            data.push_back(v);
        return Tensor({data.size()}, std::move(data));
    }

    /**
     * Create a tensor with n evenly spaced values.
     */
    static Tensor linspace(T start, T end, size_t n)
    {
        std::vector<T> data(n);
        T step = (n > 1) ? (end - start) / (n - 1) : T{0};
        for (size_t i = 0; i < n; ++i)
            data[i] = start + i * step;
        return Tensor({n}, std::move(data));
    }

    // =========================================
    // Properties
    // =========================================

    const Shape& shape() const { return m_shape; }
    const Strides& strides() const { return m_strides; }
    size_t ndim() const { return m_shape.size(); }
    size_t size() const { return compute_size(m_shape); }
    size_t size(size_t dim) const { return m_shape.at(dim); }
    bool empty() const { return size() == 0; }

    T* data() { return m_storage ? m_storage->data() + m_offset : nullptr; }
    const T* data() const { return m_storage ? m_storage->data() + m_offset : nullptr; }

    // =========================================
    // Element Access
    // =========================================

    /**
     * Access element by multi-dimensional index.
     */
    T& at(const std::vector<size_t>& indices)
    {
        return (*m_storage)[compute_offset(indices)];
    }

    const T& at(const std::vector<size_t>& indices) const
    {
        return (*m_storage)[compute_offset(indices)];
    }

    /**
     * Access element by flat index.
     */
    T& operator[](size_t idx)
    {
        return (*m_storage)[m_offset + idx];
    }

    const T& operator[](size_t idx) const
    {
        return (*m_storage)[m_offset + idx];
    }

    /**
     * Access 2D element.
     */
    T& operator()(size_t i, size_t j)
    {
        return at({i, j});
    }

    const T& operator()(size_t i, size_t j) const
    {
        return at({i, j});
    }

    // =========================================
    // Shape Operations
    // =========================================

    /**
     * Reshape tensor to new dimensions.
     */
    Tensor reshape(const Shape& new_shape) const
    {
        if (compute_size(new_shape) != size())
            throw std::invalid_argument("Cannot reshape: size mismatch");

        Tensor result;
        result.m_shape = new_shape;
        result.m_strides = compute_strides(new_shape);
        result.m_offset = m_offset;
        result.m_storage = m_storage;
        return result;
    }

    /**
     * Flatten to 1D tensor.
     */
    Tensor flatten() const
    {
        return reshape({size()});
    }

    /**
     * Transpose (swap last two dimensions).
     */
    Tensor transpose() const
    {
        if (ndim() < 2)
            throw std::invalid_argument("Cannot transpose tensor with less than 2 dimensions");

        Shape new_shape = m_shape;
        std::swap(new_shape[ndim()-2], new_shape[ndim()-1]);

        Tensor result(new_shape);
        size_t rows = m_shape[ndim()-2];
        size_t cols = m_shape[ndim()-1];

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.at({j, i}) = at({i, j});
            }
        }
        return result;
    }

    /**
     * Unsqueeze - add dimension of size 1.
     */
    Tensor unsqueeze(size_t dim) const
    {
        Shape new_shape = m_shape;
        new_shape.insert(new_shape.begin() + dim, 1);
        return reshape(new_shape);
    }

    /**
     * Squeeze - remove dimensions of size 1.
     */
    Tensor squeeze() const
    {
        Shape new_shape;
        for (size_t s : m_shape) {
            if (s != 1) new_shape.push_back(s);
        }
        if (new_shape.empty()) new_shape.push_back(1);
        return reshape(new_shape);
    }

    // =========================================
    // Element-wise Operations
    // =========================================

    Tensor operator+(const Tensor& other) const { return binary_op(other, std::plus<T>()); }
    Tensor operator-(const Tensor& other) const { return binary_op(other, std::minus<T>()); }
    Tensor operator*(const Tensor& other) const { return binary_op(other, std::multiplies<T>()); }
    Tensor operator/(const Tensor& other) const { return binary_op(other, std::divides<T>()); }

    Tensor operator+(T scalar) const { return unary_op([scalar](T x) { return x + scalar; }); }
    Tensor operator-(T scalar) const { return unary_op([scalar](T x) { return x - scalar; }); }
    Tensor operator*(T scalar) const { return unary_op([scalar](T x) { return x * scalar; }); }
    Tensor operator/(T scalar) const { return unary_op([scalar](T x) { return x / scalar; }); }

    Tensor operator-() const { return unary_op([](T x) { return -x; }); }

    Tensor& operator+=(const Tensor& other) { return inplace_binary_op(other, std::plus<T>()); }
    Tensor& operator-=(const Tensor& other) { return inplace_binary_op(other, std::minus<T>()); }
    Tensor& operator*=(const Tensor& other) { return inplace_binary_op(other, std::multiplies<T>()); }
    Tensor& operator/=(const Tensor& other) { return inplace_binary_op(other, std::divides<T>()); }

    // =========================================
    // Mathematical Functions
    // =========================================

    Tensor abs() const { return unary_op([](T x) { return std::abs(x); }); }
    Tensor sqrt() const { return unary_op([](T x) { return std::sqrt(x); }); }
    Tensor exp() const { return unary_op([](T x) { return std::exp(x); }); }
    Tensor log() const { return unary_op([](T x) { return std::log(x); }); }
    Tensor pow(T exponent) const { return unary_op([exponent](T x) { return std::pow(x, exponent); }); }

    Tensor clamp(T min_val, T max_val) const
    {
        return unary_op([min_val, max_val](T x) { return std::max(min_val, std::min(max_val, x)); });
    }

    // =========================================
    // Aggregations
    // =========================================

    T sum() const
    {
        T result = T{0};
        for (size_t i = 0; i < size(); ++i)
            result += (*this)[i];
        return result;
    }

    T mean() const
    {
        return sum() / static_cast<T>(size());
    }

    T min() const
    {
        T result = (*this)[0];
        for (size_t i = 1; i < size(); ++i)
            result = std::min(result, (*this)[i]);
        return result;
    }

    T max() const
    {
        T result = (*this)[0];
        for (size_t i = 1; i < size(); ++i)
            result = std::max(result, (*this)[i]);
        return result;
    }

    T var() const
    {
        T m = mean();
        T result = T{0};
        for (size_t i = 0; i < size(); ++i) {
            T diff = (*this)[i] - m;
            result += diff * diff;
        }
        return result / static_cast<T>(size());
    }

    T std() const
    {
        return std::sqrt(var());
    }

    /**
     * Sum along a dimension.
     */
    Tensor sum(size_t dim) const
    {
        return reduce_op(dim, T{0}, std::plus<T>());
    }

    /**
     * Mean along a dimension.
     */
    Tensor mean(size_t dim) const
    {
        Tensor s = sum(dim);
        return s / static_cast<T>(m_shape[dim]);
    }

    // =========================================
    // Matrix Operations
    // =========================================

    /**
     * Matrix multiplication.
     */
    Tensor matmul(const Tensor& other) const
    {
        if (ndim() != 2 || other.ndim() != 2)
            throw std::invalid_argument("matmul requires 2D tensors");
        if (m_shape[1] != other.m_shape[0])
            throw std::invalid_argument("matmul shape mismatch");

        size_t m = m_shape[0];
        size_t k = m_shape[1];
        size_t n = other.m_shape[1];

        Tensor result({m, n});

        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                T sum = T{0};
                for (size_t l = 0; l < k; ++l) {
                    sum += at({i, l}) * other.at({l, j});
                }
                result.at({i, j}) = sum;
            }
        }

        return result;
    }

    /**
     * Dot product (for 1D tensors).
     */
    T dot(const Tensor& other) const
    {
        if (ndim() != 1 || other.ndim() != 1)
            throw std::invalid_argument("dot requires 1D tensors");
        if (size() != other.size())
            throw std::invalid_argument("dot size mismatch");

        T result = T{0};
        for (size_t i = 0; i < size(); ++i)
            result += (*this)[i] * other[i];
        return result;
    }

    /**
     * Outer product.
     */
    Tensor outer(const Tensor& other) const
    {
        if (ndim() != 1 || other.ndim() != 1)
            throw std::invalid_argument("outer requires 1D tensors");

        Tensor result({size(), other.size()});
        for (size_t i = 0; i < size(); ++i) {
            for (size_t j = 0; j < other.size(); ++j) {
                result.at({i, j}) = (*this)[i] * other[j];
            }
        }
        return result;
    }

    // =========================================
    // Comparison Operations
    // =========================================

    Tensor<bool> operator==(const Tensor& other) const { return compare_op(other, std::equal_to<T>()); }
    Tensor<bool> operator!=(const Tensor& other) const { return compare_op(other, std::not_equal_to<T>()); }
    Tensor<bool> operator<(const Tensor& other) const { return compare_op(other, std::less<T>()); }
    Tensor<bool> operator<=(const Tensor& other) const { return compare_op(other, std::less_equal<T>()); }
    Tensor<bool> operator>(const Tensor& other) const { return compare_op(other, std::greater<T>()); }
    Tensor<bool> operator>=(const Tensor& other) const { return compare_op(other, std::greater_equal<T>()); }

    // =========================================
    // String Representation
    // =========================================

    std::string to_string() const
    {
        std::ostringstream oss;
        oss << "Tensor(shape=[";
        for (size_t i = 0; i < m_shape.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << m_shape[i];
        }
        oss << "], data=[";

        size_t max_display = std::min(size(), size_t{10});
        for (size_t i = 0; i < max_display; ++i) {
            if (i > 0) oss << ", ";
            oss << (*this)[i];
        }
        if (size() > max_display) oss << ", ...";
        oss << "])";

        return oss.str();
    }

private:
    Shape m_shape;
    Strides m_strides;
    size_t m_offset;
    storage_ptr m_storage;

    static size_t compute_size(const Shape& shape)
    {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
    }

    static Strides compute_strides(const Shape& shape)
    {
        Strides strides(shape.size());
        size_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    size_t compute_offset(const std::vector<size_t>& indices) const
    {
        size_t offset = m_offset;
        for (size_t i = 0; i < indices.size(); ++i)
            offset += indices[i] * m_strides[i];
        return offset;
    }

    template<typename Op>
    Tensor unary_op(Op op) const
    {
        Tensor result(m_shape);
        for (size_t i = 0; i < size(); ++i)
            result[i] = op((*this)[i]);
        return result;
    }

    template<typename Op>
    Tensor binary_op(const Tensor& other, Op op) const
    {
        if (m_shape != other.m_shape)
            throw std::invalid_argument("Shape mismatch in binary operation");

        Tensor result(m_shape);
        for (size_t i = 0; i < size(); ++i)
            result[i] = op((*this)[i], other[i]);
        return result;
    }

    template<typename Op>
    Tensor& inplace_binary_op(const Tensor& other, Op op)
    {
        if (m_shape != other.m_shape)
            throw std::invalid_argument("Shape mismatch in binary operation");

        for (size_t i = 0; i < size(); ++i)
            (*this)[i] = op((*this)[i], other[i]);
        return *this;
    }

    template<typename Op>
    Tensor<bool> compare_op(const Tensor& other, Op op) const
    {
        if (m_shape != other.m_shape)
            throw std::invalid_argument("Shape mismatch in comparison");

        Tensor<bool> result(m_shape);
        for (size_t i = 0; i < size(); ++i)
            result[i] = op((*this)[i], other[i]);
        return result;
    }

    template<typename Op>
    Tensor reduce_op(size_t dim, T init, Op op) const
    {
        if (dim >= ndim())
            throw std::invalid_argument("Invalid dimension for reduction");

        Shape new_shape = m_shape;
        new_shape.erase(new_shape.begin() + dim);
        if (new_shape.empty()) new_shape.push_back(1);

        Tensor result(new_shape, init);

        // Simplified reduction - works for 2D case
        if (ndim() == 2) {
            if (dim == 0) {
                for (size_t j = 0; j < m_shape[1]; ++j) {
                    T acc = init;
                    for (size_t i = 0; i < m_shape[0]; ++i)
                        acc = op(acc, at({i, j}));
                    result[j] = acc;
                }
            } else {
                for (size_t i = 0; i < m_shape[0]; ++i) {
                    T acc = init;
                    for (size_t j = 0; j < m_shape[1]; ++j)
                        acc = op(acc, at({i, j}));
                    result[i] = acc;
                }
            }
        }

        return result;
    }
};

// Type aliases
using FloatTensor = Tensor<float>;
using DoubleTensor = Tensor<double>;
using IntTensor = Tensor<int>;
using LongTensor = Tensor<int64_t>;

} // namespace aten
} // namespace gnc

#endif // GNC_ATEN_TENSOR_HPP
