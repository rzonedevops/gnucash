/*
 * opencog/tensor-logic/tensor_logic_engine.hpp
 *
 * Tensor Logic Engine
 * Unified interface for Multi-Entity, Multi-Scale, Network-Aware
 * Tensor-Enhanced Accounting
 *
 * Copyright (C) 2024 GnuCash Developers
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef GNC_TENSOR_LOGIC_ENGINE_HPP
#define GNC_TENSOR_LOGIC_ENGINE_HPP

#include "tensor_account.hpp"
#include "tensor_network.hpp"
#include "../atenspace/atenspace.hpp"
#include "../gnc-cognitive/cognitive_engine.hpp"

namespace gnc {
namespace tensor_logic {

using namespace gnc::aten;
using namespace gnc::atenspace;
using namespace gnc::opencog;

/**
 * Analysis result from tensor logic operations.
 */
struct TensorAnalysisResult
{
    std::string analysis_type;
    std::string description;
    DoubleTensor data;
    double confidence;
    std::vector<std::string> related_accounts;
    std::vector<std::pair<std::string, double>> insights;
};

/**
 * Forecast result.
 */
struct TensorForecast
{
    std::string account_guid;
    DoubleTensor predicted_values;
    DoubleTensor confidence_intervals;
    size_t horizon;
    std::string method;
};

/**
 * TensorLogicEngine - Main interface for tensor-enhanced accounting.
 *
 * Provides a unified API for:
 * - Multi-Entity Accounting: Manage and analyze multiple business entities
 * - Multi-Scale Analysis: View data at different time granularities
 * - Network-Aware: Understand money flows between accounts
 * - Tensor Operations: Advanced mathematical analysis of financial data
 *
 * Integrates with:
 * - ATen tensor library for computations
 * - ATenSpace for hybrid symbolic-neural representation
 * - CognitiveEngine for AI-powered insights
 */
class TensorLogicEngine
{
public:
    TensorLogicEngine()
        : m_network(12)  // Default 12 periods (monthly)
    {}

    /**
     * Initialize the tensor logic engine.
     */
    bool initialize()
    {
        m_initialized = true;
        return true;
    }

    /**
     * Shutdown the engine.
     */
    void shutdown()
    {
        m_accounts.clear();
        m_initialized = false;
    }

    bool is_initialized() const { return m_initialized; }

    // =========================================
    // Account Management
    // =========================================

    /**
     * Create a tensor-enhanced account.
     */
    std::shared_ptr<TensorAccount> create_account(
        const std::string& guid,
        const std::string& name,
        size_t num_entities = 1,
        size_t num_periods = 12,
        size_t num_currencies = 1)
    {
        auto account = std::make_shared<TensorAccount>(
            guid, name, num_entities, num_periods, num_currencies);
        m_account_set.add_account(account);
        m_accounts[guid] = account;
        m_network.add_node(guid, name);

        // Create corresponding atom in ATenSpace
        auto atom = m_atenspace.add_tensor_node(AtomTypes::ACCOUNT_NODE, guid);
        atom->set_embedding(EmbeddingType::SEMANTIC,
            m_atenspace.generate_semantic_embedding(name));

        return account;
    }

    /**
     * Get an account by GUID.
     */
    std::shared_ptr<TensorAccount> get_account(const std::string& guid) const
    {
        auto it = m_accounts.find(guid);
        return (it != m_accounts.end()) ? it->second : nullptr;
    }

    /**
     * Import account data from GnuCash.
     */
    void import_account_data(const std::string& guid, size_t entity, size_t period,
                            size_t currency, double balance, double debit_flow,
                            double credit_flow)
    {
        auto account = get_account(guid);
        if (!account) return;

        account->set_metric(entity, period, currency, TensorAccount::Metrics::BALANCE, balance);
        account->set_metric(entity, period, currency, TensorAccount::Metrics::DEBIT_FLOW, debit_flow);
        account->set_metric(entity, period, currency, TensorAccount::Metrics::CREDIT_FLOW, credit_flow);
        account->set_metric(entity, period, currency, TensorAccount::Metrics::NET_FLOW,
                           credit_flow - debit_flow);
    }

    // =========================================
    // Transaction Recording
    // =========================================

    /**
     * Record a transaction in the network.
     */
    void record_transaction(const std::string& from_account, const std::string& to_account,
                           double amount, size_t period)
    {
        m_network.record_transaction(from_account, to_account, amount, period);

        // Update account flows
        auto from_acc = get_account(from_account);
        auto to_acc = get_account(to_account);

        if (from_acc) {
            double current = from_acc->get_metric(0, period, 0, TensorAccount::Metrics::DEBIT_FLOW);
            from_acc->set_metric(0, period, 0, TensorAccount::Metrics::DEBIT_FLOW, current + amount);
        }

        if (to_acc) {
            double current = to_acc->get_metric(0, period, 0, TensorAccount::Metrics::CREDIT_FLOW);
            to_acc->set_metric(0, period, 0, TensorAccount::Metrics::CREDIT_FLOW, current + amount);
        }
    }

    // =========================================
    // Multi-Entity Operations
    // =========================================

    /**
     * Get consolidated view across all entities.
     */
    TensorAnalysisResult consolidate_entities(const std::string& guid)
    {
        TensorAnalysisResult result;
        result.analysis_type = "entity_consolidation";

        auto account = get_account(guid);
        if (!account) {
            result.description = "Account not found";
            result.confidence = 0.0;
            return result;
        }

        auto consolidated = account->consolidate();
        result.description = "Consolidated view of " + account->name();
        result.data = consolidated.get_balances().flatten();
        result.confidence = 1.0;

        // Calculate entity contributions
        for (size_t e = 0; e < account->num_entities(); ++e) {
            auto contrib = account->entity_contribution_matrix(account->num_periods() - 1, 0);
            result.insights.emplace_back("Entity " + std::to_string(e) + " contribution",
                                        contrib[e]);
        }

        return result;
    }

    /**
     * Compare entities within an account.
     */
    DoubleTensor compare_entities(const std::string& guid, size_t metric)
    {
        auto account = get_account(guid);
        if (!account) return DoubleTensor({1}, 0.0);

        // Create comparison matrix (entities x periods)
        DoubleTensor comparison({account->num_entities(), account->num_periods()});

        for (size_t e = 0; e < account->num_entities(); ++e) {
            auto series = account->get_time_series(e, 0, metric);
            for (size_t p = 0; p < account->num_periods(); ++p) {
                comparison.at({e, p}) = series[p];
            }
        }

        return comparison;
    }

    // =========================================
    // Multi-Scale Analysis
    // =========================================

    /**
     * Analyze account at multiple time scales.
     */
    TensorAnalysisResult multi_scale_analysis(const std::string& guid)
    {
        TensorAnalysisResult result;
        result.analysis_type = "multi_scale";

        auto account = get_account(guid);
        if (!account) {
            result.description = "Account not found";
            return result;
        }

        result.description = "Multi-scale analysis of " + account->name();

        auto scales = account->multi_scale_decomposition(0, 0);

        // Combine scales into result
        size_t total_size = 0;
        for (const auto& s : scales) total_size += s.size();

        std::vector<double> combined;
        for (const auto& scale : scales) {
            for (size_t i = 0; i < scale.size(); ++i)
                combined.push_back(scale[i]);
        }

        result.data = DoubleTensor({combined.size()}, std::move(combined));
        result.confidence = 0.9;

        // Add insights about different scales
        for (size_t i = 0; i < scales.size(); ++i) {
            result.insights.emplace_back("Scale " + std::to_string(i) + " trend",
                                        scales[i].mean());
        }

        return result;
    }

    /**
     * Aggregate to specific time scale.
     */
    std::shared_ptr<TensorAccount> aggregate_to_scale(const std::string& guid, TimeScale scale)
    {
        auto account = get_account(guid);
        if (!account) return nullptr;

        auto aggregated = account->aggregate_to_scale(scale);
        return std::make_shared<TensorAccount>(aggregated);
    }

    // =========================================
    // Network Analysis
    // =========================================

    /**
     * Get network flow analysis.
     */
    TensorAnalysisResult analyze_flow_network()
    {
        TensorAnalysisResult result;
        result.analysis_type = "network_flow";
        result.description = "Account network flow analysis";

        // Get network statistics
        auto centrality = m_network.flow_centrality();
        auto pagerank = m_network.pagerank();

        result.data = centrality;
        result.confidence = 0.85;

        // Add top accounts by centrality
        auto nodes = m_network.get_nodes();
        std::vector<std::pair<size_t, double>> ranked;
        for (size_t i = 0; i < nodes.size(); ++i) {
            ranked.emplace_back(i, centrality[i]);
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        for (size_t i = 0; i < std::min(size_t{5}, ranked.size()); ++i) {
            result.insights.emplace_back(nodes[ranked[i].first], ranked[i].second);
            result.related_accounts.push_back(nodes[ranked[i].first]);
        }

        return result;
    }

    /**
     * Find money flow path between accounts.
     */
    std::vector<std::string> find_flow_path(const std::string& from, const std::string& to)
    {
        return m_network.shortest_path(from, to);
    }

    /**
     * Detect circular money flows.
     */
    std::vector<std::vector<std::string>> detect_circular_flows()
    {
        return m_network.detect_circular_flows();
    }

    /**
     * Detect flow anomalies.
     */
    std::vector<std::pair<std::string, double>> detect_flow_anomalies(double threshold = 2.0)
    {
        return m_network.detect_flow_anomalies(threshold);
    }

    // =========================================
    // Forecasting
    // =========================================

    /**
     * Forecast account balances.
     */
    TensorForecast forecast_balance(const std::string& guid, size_t horizon = 3)
    {
        TensorForecast forecast;
        forecast.account_guid = guid;
        forecast.horizon = horizon;
        forecast.method = "exponential_smoothing";

        auto account = get_account(guid);
        if (!account) {
            forecast.predicted_values = DoubleTensor({horizon}, 0.0);
            forecast.confidence_intervals = DoubleTensor({horizon, 2}, 0.0);
            return forecast;
        }

        auto history = account->get_time_series(0, 0, TensorAccount::Metrics::BALANCE);

        // Simple exponential smoothing forecast
        double alpha = 0.3;
        double last_value = history[history.size() - 1];
        double smoothed = last_value;

        // Calculate historical smoothed values
        for (size_t i = 0; i < history.size(); ++i) {
            smoothed = alpha * history[i] + (1 - alpha) * smoothed;
        }

        // Forecast
        std::vector<double> predictions(horizon, smoothed);

        // Calculate confidence intervals based on historical variance
        double variance = history.var();
        double std_error = std::sqrt(variance);

        std::vector<double> ci(horizon * 2);
        for (size_t i = 0; i < horizon; ++i) {
            double width = 1.96 * std_error * std::sqrt(i + 1);
            ci[i * 2] = predictions[i] - width;      // Lower bound
            ci[i * 2 + 1] = predictions[i] + width;  // Upper bound
        }

        forecast.predicted_values = DoubleTensor({horizon}, std::move(predictions));
        forecast.confidence_intervals = DoubleTensor({horizon, 2}, std::move(ci));

        return forecast;
    }

    /**
     * Forecast cash flow.
     */
    TensorForecast forecast_cash_flow(const std::string& guid, size_t horizon = 3)
    {
        TensorForecast forecast;
        forecast.account_guid = guid;
        forecast.horizon = horizon;
        forecast.method = "trend_projection";

        auto account = get_account(guid);
        if (!account) {
            forecast.predicted_values = DoubleTensor({horizon}, 0.0);
            return forecast;
        }

        // Get net flow history
        auto history = account->get_time_series(0, 0, TensorAccount::Metrics::NET_FLOW);

        // Calculate trend
        account->compute_trend(0, 0);
        double trend = account->get_metric(0, 0, 0, TensorAccount::Metrics::TREND);

        double last_flow = history[history.size() - 1];
        double avg_flow = history.mean();

        std::vector<double> predictions(horizon);
        for (size_t i = 0; i < horizon; ++i) {
            predictions[i] = avg_flow * (1 + trend * (i + 1));
        }

        forecast.predicted_values = DoubleTensor({horizon}, std::move(predictions));

        return forecast;
    }

    // =========================================
    // Similarity & Clustering
    // =========================================

    /**
     * Find accounts similar to a given account.
     */
    std::vector<std::pair<std::string, double>> find_similar_accounts(
        const std::string& guid, size_t top_k = 5)
    {
        auto account = get_account(guid);
        if (!account) return {};

        return m_account_set.find_similar(*account, top_k);
    }

    /**
     * Cluster accounts by behavior.
     */
    std::vector<std::vector<std::string>> cluster_accounts(size_t num_clusters)
    {
        return m_account_set.cluster_accounts(num_clusters);
    }

    /**
     * Semantic search for accounts.
     */
    std::vector<std::pair<Handle, double>> semantic_account_search(
        const std::string& query, size_t top_k = 10)
    {
        return m_atenspace.semantic_search(query, top_k);
    }

    // =========================================
    // Statistics
    // =========================================

    struct Stats
    {
        size_t num_accounts;
        size_t num_network_nodes;
        size_t num_network_edges;
        size_t num_atoms;
        double total_flow;
    };

    Stats get_stats() const
    {
        Stats stats;
        stats.num_accounts = m_accounts.size();
        stats.num_network_nodes = m_network.num_nodes();
        stats.num_network_edges = m_network.num_edges();
        stats.num_atoms = m_atenspace.size();

        // Calculate total flow
        stats.total_flow = 0.0;
        // Would need to iterate through network edges

        return stats;
    }

    // =========================================
    // Accessors
    // =========================================

    ATenSpace& atenspace() { return m_atenspace; }
    const ATenSpace& atenspace() const { return m_atenspace; }

    TensorNetwork& network() { return m_network; }
    const TensorNetwork& network() const { return m_network; }

    TensorAccountSet& account_set() { return m_account_set; }
    const TensorAccountSet& account_set() const { return m_account_set; }

private:
    bool m_initialized = false;

    ATenSpace m_atenspace;
    TensorNetwork m_network;
    TensorAccountSet m_account_set;
    std::unordered_map<std::string, std::shared_ptr<TensorAccount>> m_accounts;
};

/**
 * Global tensor logic engine instance.
 */
inline TensorLogicEngine& tensor_logic_engine()
{
    static TensorLogicEngine engine;
    return engine;
}

} // namespace tensor_logic
} // namespace gnc

#endif // GNC_TENSOR_LOGIC_ENGINE_HPP
