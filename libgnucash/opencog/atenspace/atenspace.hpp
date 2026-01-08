/*
 * opencog/atenspace/atenspace.hpp
 *
 * ATenSpace - AtomSpace with tensor embedding support
 * Hybrid symbolic-neural knowledge representation
 *
 * Copyright (C) 2024 GnuCash Developers
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef GNC_ATENSPACE_HPP
#define GNC_ATENSPACE_HPP

#include "../atomspace/atomspace.hpp"
#include "tensor_atom.hpp"
#include "../aten/tensor_ops.hpp"

namespace gnc {
namespace atenspace {

using namespace gnc::opencog;
using namespace gnc::aten;

/**
 * EmbeddingConfig - configuration for embedding generation.
 */
struct EmbeddingConfig
{
    size_t semantic_dim = 64;       // Semantic embedding dimension
    size_t temporal_dim = 32;       // Temporal embedding dimension
    size_t structural_dim = 16;     // Structural embedding dimension
    size_t financial_dim = 32;      // Financial metrics dimension
    double learning_rate = 0.01;
};

/**
 * ATenSpace - AtomSpace with integrated tensor operations.
 *
 * Extends the symbolic AtomSpace with:
 * - Tensor embeddings for atoms
 * - Semantic similarity search
 * - Neural attention mechanisms
 * - Embedding-based inference
 */
class ATenSpace : public AtomSpace
{
public:
    ATenSpace() = default;
    explicit ATenSpace(const EmbeddingConfig& config)
        : m_config(config)
    {}

    /**
     * Add a tensor-enabled node.
     */
    std::shared_ptr<TensorNode> add_tensor_node(AtomType type, const std::string& name)
    {
        auto node = create_tensor_node(type, name);
        add_atom(node);
        return node;
    }

    /**
     * Add a tensor-enabled link.
     */
    std::shared_ptr<TensorLink> add_tensor_link(AtomType type, const HandleSeq& outgoing)
    {
        auto link = create_tensor_link(type, outgoing);
        add_atom(link);
        return link;
    }

    /**
     * Set embedding for a node.
     */
    void set_embedding(const Handle& atom, EmbeddingType type, const DoubleTensor& tensor)
    {
        auto* tn = dynamic_cast<TensorNode*>(atom.get());
        if (tn) {
            tn->set_embedding(type, tensor);
            update_embedding_index(atom, type);
        }
    }

    /**
     * Generate semantic embedding for a node (based on name/context).
     */
    DoubleTensor generate_semantic_embedding(const std::string& text)
    {
        // Simple hash-based embedding (in practice, use word vectors)
        DoubleTensor embedding({m_config.semantic_dim}, 0.0);

        std::hash<std::string> hasher;
        size_t hash = hasher(text);

        // Generate pseudo-random embedding from hash
        std::mt19937 gen(hash);
        std::normal_distribution<double> dist(0.0, 1.0);

        for (size_t i = 0; i < m_config.semantic_dim; ++i) {
            embedding[i] = dist(gen);
        }

        // Normalize
        double norm = std::sqrt(embedding.pow(2).sum());
        if (norm > 0)
            return embedding / norm;
        return embedding;
    }

    /**
     * Generate financial embedding for an account/transaction.
     */
    DoubleTensor generate_financial_embedding(double amount, const std::string& account_type,
                                               bool is_debit, int month, int day_of_week)
    {
        std::vector<double> data(m_config.financial_dim, 0.0);

        // Amount features (log-scaled)
        data[0] = std::log1p(std::abs(amount));
        data[1] = amount > 0 ? 1.0 : -1.0;

        // Account type encoding
        size_t type_hash = std::hash<std::string>{}(account_type) % 10;
        data[2 + type_hash] = 1.0;

        // Debit/credit
        data[12] = is_debit ? 1.0 : 0.0;

        // Temporal features (cyclical encoding)
        data[13] = std::sin(2.0 * M_PI * month / 12.0);
        data[14] = std::cos(2.0 * M_PI * month / 12.0);
        data[15] = std::sin(2.0 * M_PI * day_of_week / 7.0);
        data[16] = std::cos(2.0 * M_PI * day_of_week / 7.0);

        return DoubleTensor({m_config.financial_dim}, std::move(data));
    }

    /**
     * Find atoms similar to a query embedding.
     */
    std::vector<std::pair<Handle, double>> find_similar(
        const DoubleTensor& query_embedding,
        EmbeddingType type,
        size_t top_k = 10,
        double threshold = 0.0)
    {
        std::vector<std::pair<Handle, double>> results;

        for_each([&](const Handle& atom) {
            auto* tn = dynamic_cast<TensorNode*>(atom.get());
            if (!tn) return;

            auto emb = tn->get_embedding(type);
            if (!emb) return;

            double sim = compute_similarity(query_embedding, emb->tensor());
            if (sim >= threshold)
                results.emplace_back(atom, sim);
        });

        // Sort by similarity
        std::sort(results.begin(), results.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        if (results.size() > top_k)
            results.resize(top_k);

        return results;
    }

    /**
     * Find semantically similar atoms by text query.
     */
    std::vector<std::pair<Handle, double>> semantic_search(
        const std::string& query,
        size_t top_k = 10)
    {
        auto query_emb = generate_semantic_embedding(query);
        return find_similar(query_emb, EmbeddingType::SEMANTIC, top_k);
    }

    /**
     * Compute attention scores between atoms.
     */
    DoubleTensor compute_attention(const HandleSeq& atoms, const DoubleTensor& query)
    {
        std::vector<double> scores;

        for (const auto& atom : atoms) {
            auto* tn = dynamic_cast<TensorNode*>(atom.get());
            if (!tn) {
                scores.push_back(0.0);
                continue;
            }

            auto emb = tn->get_embedding(EmbeddingType::SEMANTIC);
            if (!emb) {
                scores.push_back(0.0);
                continue;
            }

            double score = query.flatten().dot(emb->tensor().flatten());
            scores.push_back(score);
        }

        // Softmax
        auto scores_tensor = DoubleTensor({scores.size()}, std::move(scores));
        return ops::softmax(scores_tensor);
    }

    /**
     * Aggregate embeddings using attention.
     */
    DoubleTensor attention_aggregate(const HandleSeq& atoms, const DoubleTensor& query)
    {
        auto attention = compute_attention(atoms, query);

        // Get embedding dimension
        size_t emb_dim = m_config.semantic_dim;
        for (const auto& atom : atoms) {
            auto* tn = dynamic_cast<TensorNode*>(atom.get());
            if (tn && tn->has_embedding(EmbeddingType::SEMANTIC)) {
                emb_dim = tn->get_embedding(EmbeddingType::SEMANTIC)->dimension();
                break;
            }
        }

        DoubleTensor result({emb_dim}, 0.0);

        for (size_t i = 0; i < atoms.size(); ++i) {
            auto* tn = dynamic_cast<TensorNode*>(atoms[i].get());
            if (!tn) continue;

            auto emb = tn->get_embedding(EmbeddingType::SEMANTIC);
            if (!emb) continue;

            for (size_t j = 0; j < emb_dim; ++j) {
                result[j] += attention[i] * emb->tensor()[j];
            }
        }

        return result;
    }

    /**
     * Learn embedding from context (simplified skip-gram style).
     */
    void learn_embedding(const Handle& target, const HandleSeq& context)
    {
        auto* tn = dynamic_cast<TensorNode*>(target.get());
        if (!tn) return;

        // Initialize embedding if not exists
        if (!tn->has_embedding(EmbeddingType::SEMANTIC)) {
            tn->set_embedding(EmbeddingType::SEMANTIC,
                generate_semantic_embedding(tn->name()));
        }

        // Average context embeddings
        DoubleTensor context_sum({m_config.semantic_dim}, 0.0);
        size_t count = 0;

        for (const auto& ctx : context) {
            auto* ctx_tn = dynamic_cast<TensorNode*>(ctx.get());
            if (!ctx_tn) continue;

            auto emb = ctx_tn->get_embedding(EmbeddingType::SEMANTIC);
            if (!emb) continue;

            context_sum += emb->tensor();
            ++count;
        }

        if (count == 0) return;
        auto context_avg = context_sum / static_cast<double>(count);

        // Update target embedding (gradient step)
        auto target_emb = tn->get_embedding(EmbeddingType::SEMANTIC)->tensor();
        auto update = (context_avg - target_emb) * m_config.learning_rate;
        tn->set_embedding(EmbeddingType::SEMANTIC, target_emb + update);
    }

    /**
     * Get embedding configuration.
     */
    const EmbeddingConfig& config() const { return m_config; }

    /**
     * Set embedding configuration.
     */
    void set_config(const EmbeddingConfig& config) { m_config = config; }

private:
    EmbeddingConfig m_config;

    // Embedding index for fast similarity search
    std::unordered_map<EmbeddingType, std::vector<Handle>> m_embedding_index;

    void update_embedding_index(const Handle& atom, EmbeddingType type)
    {
        auto& index = m_embedding_index[type];
        if (std::find(index.begin(), index.end(), atom) == index.end())
            index.push_back(atom);
    }

    double compute_similarity(const DoubleTensor& a, const DoubleTensor& b) const
    {
        if (a.size() != b.size()) return 0.0;

        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        if (norm_a == 0 || norm_b == 0) return 0.0;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }
};

} // namespace atenspace
} // namespace gnc

#endif // GNC_ATENSPACE_HPP
