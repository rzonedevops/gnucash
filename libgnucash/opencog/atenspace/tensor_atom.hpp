/*
 * opencog/atenspace/tensor_atom.hpp
 *
 * TensorAtom - Atoms with embedded tensor representations
 * Bridges symbolic AtomSpace with neural tensor embeddings
 *
 * Based on ATenSpace design
 *
 * Copyright (C) 2024 GnuCash Developers
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef GNC_ATENSPACE_TENSOR_ATOM_HPP
#define GNC_ATENSPACE_TENSOR_ATOM_HPP

#include "../atomspace/atom.hpp"
#include "../aten/tensor.hpp"
#include <optional>

namespace gnc {
namespace atenspace {

using namespace gnc::opencog;
using namespace gnc::aten;

/**
 * Embedding type for different kinds of tensor representations.
 */
enum class EmbeddingType
{
    SEMANTIC,       // Semantic meaning embedding
    TEMPORAL,       // Time-series embedding
    STRUCTURAL,     // Graph structure embedding
    FINANCIAL,      // Financial metrics embedding
    ATTENTION       // Attention/importance embedding
};

/**
 * TensorEmbedding - tensor representation attached to an atom.
 */
class TensorEmbedding
{
public:
    TensorEmbedding() = default;

    TensorEmbedding(EmbeddingType type, const DoubleTensor& tensor)
        : m_type(type)
        , m_tensor(tensor)
    {}

    EmbeddingType type() const { return m_type; }
    const DoubleTensor& tensor() const { return m_tensor; }
    DoubleTensor& tensor() { return m_tensor; }

    size_t dimension() const { return m_tensor.size(); }

    /**
     * Compute similarity with another embedding.
     */
    double similarity(const TensorEmbedding& other) const
    {
        if (m_tensor.size() != other.m_tensor.size())
            return 0.0;

        // Cosine similarity
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < m_tensor.size(); ++i) {
            dot += m_tensor[i] * other.m_tensor[i];
            norm_a += m_tensor[i] * m_tensor[i];
            norm_b += other.m_tensor[i] * other.m_tensor[i];
        }

        if (norm_a == 0 || norm_b == 0) return 0.0;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

private:
    EmbeddingType m_type = EmbeddingType::SEMANTIC;
    DoubleTensor m_tensor;
};

/**
 * TensorNode - a Node with tensor embeddings.
 */
class TensorNode : public Node
{
public:
    TensorNode(AtomType type, const std::string& name)
        : Node(type, name)
    {}

    /**
     * Set an embedding.
     */
    void set_embedding(EmbeddingType type, const DoubleTensor& tensor)
    {
        m_embeddings[type] = TensorEmbedding(type, tensor);
    }

    /**
     * Get an embedding.
     */
    std::optional<TensorEmbedding> get_embedding(EmbeddingType type) const
    {
        auto it = m_embeddings.find(type);
        if (it != m_embeddings.end())
            return it->second;
        return std::nullopt;
    }

    /**
     * Check if has embedding.
     */
    bool has_embedding(EmbeddingType type) const
    {
        return m_embeddings.find(type) != m_embeddings.end();
    }

    /**
     * Get all embeddings.
     */
    const std::unordered_map<EmbeddingType, TensorEmbedding>& embeddings() const
    {
        return m_embeddings;
    }

    /**
     * Compute combined embedding vector.
     */
    DoubleTensor combined_embedding() const
    {
        if (m_embeddings.empty())
            return DoubleTensor({1}, 0.0);

        // Concatenate all embeddings
        size_t total_size = 0;
        for (const auto& [type, emb] : m_embeddings)
            total_size += emb.dimension();

        std::vector<double> data;
        data.reserve(total_size);

        for (const auto& [type, emb] : m_embeddings) {
            const auto& t = emb.tensor();
            for (size_t i = 0; i < t.size(); ++i)
                data.push_back(t[i]);
        }

        return DoubleTensor({data.size()}, std::move(data));
    }

private:
    std::unordered_map<EmbeddingType, TensorEmbedding> m_embeddings;
};

/**
 * TensorLink - a Link with tensor-based relationship weights.
 */
class TensorLink : public Link
{
public:
    TensorLink(AtomType type, const HandleSeq& outgoing)
        : Link(type, outgoing)
    {}

    /**
     * Set weight tensor (for multi-dimensional relationships).
     */
    void set_weight_tensor(const DoubleTensor& weights)
    {
        m_weights = weights;
    }

    /**
     * Get weight tensor.
     */
    const DoubleTensor& weight_tensor() const
    {
        return m_weights;
    }

    /**
     * Set attention tensor.
     */
    void set_attention(const DoubleTensor& attention)
    {
        m_attention = attention;
    }

    /**
     * Get attention tensor.
     */
    const DoubleTensor& attention() const
    {
        return m_attention;
    }

    /**
     * Compute weighted combination of outgoing embeddings.
     */
    DoubleTensor weighted_embedding() const
    {
        if (outgoing().empty())
            return DoubleTensor({1}, 0.0);

        // Get first embedding size
        size_t emb_size = 0;
        for (const auto& h : outgoing()) {
            auto* tn = dynamic_cast<TensorNode*>(h.get());
            if (tn && tn->has_embedding(EmbeddingType::SEMANTIC)) {
                emb_size = tn->get_embedding(EmbeddingType::SEMANTIC)->dimension();
                break;
            }
        }

        if (emb_size == 0)
            return DoubleTensor({1}, 0.0);

        DoubleTensor result({emb_size}, 0.0);
        double total_weight = 0.0;

        for (size_t i = 0; i < outgoing().size(); ++i) {
            auto* tn = dynamic_cast<TensorNode*>(outgoing()[i].get());
            if (!tn) continue;

            auto emb = tn->get_embedding(EmbeddingType::SEMANTIC);
            if (!emb) continue;

            double weight = (i < m_weights.size()) ? m_weights[i] : 1.0;
            for (size_t j = 0; j < emb_size; ++j) {
                result[j] += weight * emb->tensor()[j];
            }
            total_weight += weight;
        }

        if (total_weight > 0) {
            for (size_t j = 0; j < emb_size; ++j)
                result[j] /= total_weight;
        }

        return result;
    }

private:
    DoubleTensor m_weights;     // Relationship weights
    DoubleTensor m_attention;   // Attention scores
};

/**
 * Factory functions for tensor atoms.
 */
inline std::shared_ptr<TensorNode> create_tensor_node(AtomType type, const std::string& name)
{
    return std::make_shared<TensorNode>(type, name);
}

inline std::shared_ptr<TensorLink> create_tensor_link(AtomType type, const HandleSeq& outgoing)
{
    return std::make_shared<TensorLink>(type, outgoing);
}

} // namespace atenspace
} // namespace gnc

namespace std {
    template<>
    struct hash<gnc::atenspace::EmbeddingType> {
        size_t operator()(gnc::atenspace::EmbeddingType t) const {
            return static_cast<size_t>(t);
        }
    };
}

#endif // GNC_ATENSPACE_TENSOR_ATOM_HPP
