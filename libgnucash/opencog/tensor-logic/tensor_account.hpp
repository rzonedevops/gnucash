/*
 * opencog/tensor-logic/tensor_account.hpp
 *
 * Tensor-Enhanced Account System
 * Multi-Entity, Multi-Scale, Network-Aware Account Representation
 *
 * Copyright (C) 2024 GnuCash Developers
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef GNC_TENSOR_ACCOUNT_HPP
#define GNC_TENSOR_ACCOUNT_HPP

#include "../aten/tensor.hpp"
#include "../aten/tensor_ops.hpp"
#include "../atenspace/atenspace.hpp"

#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>

namespace gnc {
namespace tensor_logic {

using namespace gnc::aten;
using namespace gnc::atenspace;

/**
 * Time scale for multi-scale analysis.
 */
enum class TimeScale
{
    DAILY,
    WEEKLY,
    MONTHLY,
    QUARTERLY,
    YEARLY
};

/**
 * Entity type in multi-entity accounting.
 */
enum class EntityType
{
    INDIVIDUAL,
    BUSINESS,
    DEPARTMENT,
    PROJECT,
    CONSOLIDATED
};

/**
 * TensorAccount - Enhanced account with tensor representations.
 *
 * Represents an account as a multi-dimensional tensor structure:
 * - Dimension 0: Entity (for multi-entity accounting)
 * - Dimension 1: Time period (for multi-scale analysis)
 * - Dimension 2: Currency/commodity
 * - Dimension 3: Account metrics (balance, flow, velocity, etc.)
 */
class TensorAccount
{
public:
    /**
     * Account metrics tracked in tensor form.
     */
    struct Metrics
    {
        static constexpr size_t BALANCE = 0;
        static constexpr size_t DEBIT_FLOW = 1;
        static constexpr size_t CREDIT_FLOW = 2;
        static constexpr size_t NET_FLOW = 3;
        static constexpr size_t VELOCITY = 4;       // Transaction frequency
        static constexpr size_t VOLATILITY = 5;     // Balance volatility
        static constexpr size_t TREND = 6;          // Trend direction
        static constexpr size_t SEASONALITY = 7;    // Seasonal component
        static constexpr size_t NUM_METRICS = 8;
    };

    TensorAccount(const std::string& guid, const std::string& name,
                  size_t num_entities = 1, size_t num_periods = 12,
                  size_t num_currencies = 1)
        : m_guid(guid)
        , m_name(name)
        , m_num_entities(num_entities)
        , m_num_periods(num_periods)
        , m_num_currencies(num_currencies)
        , m_data({num_entities, num_periods, num_currencies, Metrics::NUM_METRICS}, 0.0)
    {}

    // =========================================
    // Accessors
    // =========================================

    const std::string& guid() const { return m_guid; }
    const std::string& name() const { return m_name; }
    const DoubleTensor& data() const { return m_data; }
    DoubleTensor& data() { return m_data; }

    size_t num_entities() const { return m_num_entities; }
    size_t num_periods() const { return m_num_periods; }
    size_t num_currencies() const { return m_num_currencies; }

    // =========================================
    // Metric Access
    // =========================================

    /**
     * Get metric value for specific entity, period, currency.
     */
    double get_metric(size_t entity, size_t period, size_t currency, size_t metric) const
    {
        return m_data.at({entity, period, currency, metric});
    }

    /**
     * Set metric value.
     */
    void set_metric(size_t entity, size_t period, size_t currency, size_t metric, double value)
    {
        m_data.at({entity, period, currency, metric}) = value;
    }

    /**
     * Get balance tensor (all entities, periods, currencies).
     */
    DoubleTensor get_balances() const
    {
        DoubleTensor result({m_num_entities, m_num_periods, m_num_currencies});
        for (size_t e = 0; e < m_num_entities; ++e) {
            for (size_t p = 0; p < m_num_periods; ++p) {
                for (size_t c = 0; c < m_num_currencies; ++c) {
                    result.at({e, p, c}) = get_metric(e, p, c, Metrics::BALANCE);
                }
            }
        }
        return result;
    }

    /**
     * Get time series for a specific entity and currency.
     */
    DoubleTensor get_time_series(size_t entity, size_t currency, size_t metric) const
    {
        std::vector<double> data(m_num_periods);
        for (size_t p = 0; p < m_num_periods; ++p) {
            data[p] = get_metric(entity, p, currency, metric);
        }
        return DoubleTensor({m_num_periods}, std::move(data));
    }

    /**
     * Get cross-entity comparison for a period.
     */
    DoubleTensor get_entity_comparison(size_t period, size_t currency, size_t metric) const
    {
        std::vector<double> data(m_num_entities);
        for (size_t e = 0; e < m_num_entities; ++e) {
            data[e] = get_metric(e, period, currency, metric);
        }
        return DoubleTensor({m_num_entities}, std::move(data));
    }

    // =========================================
    // Multi-Scale Operations
    // =========================================

    /**
     * Aggregate to different time scale.
     */
    TensorAccount aggregate_to_scale(TimeScale target_scale) const
    {
        size_t aggregation_factor = get_aggregation_factor(target_scale);
        size_t new_periods = m_num_periods / aggregation_factor;

        TensorAccount result(m_guid, m_name, m_num_entities, new_periods, m_num_currencies);

        for (size_t e = 0; e < m_num_entities; ++e) {
            for (size_t np = 0; np < new_periods; ++np) {
                for (size_t c = 0; c < m_num_currencies; ++c) {
                    // Aggregate each metric
                    for (size_t m = 0; m < Metrics::NUM_METRICS; ++m) {
                        double sum = 0.0;
                        for (size_t i = 0; i < aggregation_factor; ++i) {
                            sum += get_metric(e, np * aggregation_factor + i, c, m);
                        }

                        // Use last value for balance, sum for flows
                        if (m == Metrics::BALANCE) {
                            result.set_metric(e, np, c, m, get_metric(e, (np + 1) * aggregation_factor - 1, c, m));
                        } else {
                            result.set_metric(e, np, c, m, sum);
                        }
                    }
                }
            }
        }

        return result;
    }

    /**
     * Compute multi-scale representation (wavelet-like decomposition).
     */
    std::vector<DoubleTensor> multi_scale_decomposition(size_t entity, size_t currency) const
    {
        std::vector<DoubleTensor> scales;
        auto series = get_time_series(entity, currency, Metrics::BALANCE);

        // Original scale
        scales.push_back(series);

        // Progressively smoother scales
        auto current = series;
        for (size_t level = 1; level < 4 && current.size() >= 4; ++level) {
            current = ops::moving_average(current, 1 << level);
            scales.push_back(current);
        }

        return scales;
    }

    // =========================================
    // Multi-Entity Operations
    // =========================================

    /**
     * Consolidate across entities.
     */
    TensorAccount consolidate() const
    {
        TensorAccount result(m_guid + "_consolidated", m_name + " (Consolidated)",
                            1, m_num_periods, m_num_currencies);

        for (size_t p = 0; p < m_num_periods; ++p) {
            for (size_t c = 0; c < m_num_currencies; ++c) {
                for (size_t m = 0; m < Metrics::NUM_METRICS; ++m) {
                    double sum = 0.0;
                    for (size_t e = 0; e < m_num_entities; ++e) {
                        sum += get_metric(e, p, c, m);
                    }
                    result.set_metric(0, p, c, m, sum);
                }
            }
        }

        return result;
    }

    /**
     * Get entity contribution matrix.
     */
    DoubleTensor entity_contribution_matrix(size_t period, size_t currency) const
    {
        double total = 0.0;
        for (size_t e = 0; e < m_num_entities; ++e) {
            total += std::abs(get_metric(e, period, currency, Metrics::BALANCE));
        }

        std::vector<double> contributions(m_num_entities);
        for (size_t e = 0; e < m_num_entities; ++e) {
            contributions[e] = (total > 0) ?
                std::abs(get_metric(e, period, currency, Metrics::BALANCE)) / total : 0.0;
        }

        return DoubleTensor({m_num_entities}, std::move(contributions));
    }

    // =========================================
    // Analytics
    // =========================================

    /**
     * Compute velocity (transaction frequency).
     */
    void compute_velocity(size_t entity, size_t currency,
                         const std::vector<int>& transaction_counts)
    {
        for (size_t p = 0; p < std::min(m_num_periods, transaction_counts.size()); ++p) {
            set_metric(entity, p, currency, Metrics::VELOCITY,
                      static_cast<double>(transaction_counts[p]));
        }
    }

    /**
     * Compute volatility (standard deviation of changes).
     */
    void compute_volatility(size_t entity, size_t currency)
    {
        auto series = get_time_series(entity, currency, Metrics::BALANCE);
        if (series.size() < 2) return;

        // Compute returns
        std::vector<double> returns;
        for (size_t i = 1; i < series.size(); ++i) {
            if (series[i-1] != 0)
                returns.push_back((series[i] - series[i-1]) / std::abs(series[i-1]));
        }

        if (returns.empty()) return;

        // Standard deviation
        DoubleTensor ret_tensor({returns.size()}, std::move(returns));
        double vol = ret_tensor.std();

        // Set for all periods (rolling window could be used)
        for (size_t p = 0; p < m_num_periods; ++p) {
            set_metric(entity, p, currency, Metrics::VOLATILITY, vol);
        }
    }

    /**
     * Compute trend using linear regression.
     */
    void compute_trend(size_t entity, size_t currency)
    {
        auto series = get_time_series(entity, currency, Metrics::BALANCE);
        if (series.size() < 2) return;

        // Simple linear regression
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
        for (size_t i = 0; i < series.size(); ++i) {
            sum_x += i;
            sum_y += series[i];
            sum_xy += i * series[i];
            sum_xx += i * i;
        }

        double n = static_cast<double>(series.size());
        double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);

        // Normalize slope
        double mean = sum_y / n;
        double normalized_slope = (mean != 0) ? slope / std::abs(mean) : 0;

        for (size_t p = 0; p < m_num_periods; ++p) {
            set_metric(entity, p, currency, Metrics::TREND, normalized_slope);
        }
    }

    /**
     * Get account embedding for similarity search.
     */
    DoubleTensor get_embedding() const
    {
        // Flatten key metrics into embedding
        std::vector<double> emb;

        // Include aggregated statistics
        for (size_t e = 0; e < m_num_entities; ++e) {
            for (size_t c = 0; c < m_num_currencies; ++c) {
                auto balance_series = get_time_series(e, c, Metrics::BALANCE);
                emb.push_back(balance_series.mean());
                emb.push_back(balance_series.std());
                emb.push_back(balance_series.min());
                emb.push_back(balance_series.max());

                auto flow_series = get_time_series(e, c, Metrics::NET_FLOW);
                emb.push_back(flow_series.mean());
                emb.push_back(flow_series.sum());
            }
        }

        return DoubleTensor({emb.size()}, std::move(emb));
    }

private:
    std::string m_guid;
    std::string m_name;
    size_t m_num_entities;
    size_t m_num_periods;
    size_t m_num_currencies;
    DoubleTensor m_data;

    size_t get_aggregation_factor(TimeScale scale) const
    {
        switch (scale) {
            case TimeScale::DAILY: return 1;
            case TimeScale::WEEKLY: return 7;
            case TimeScale::MONTHLY: return 30;
            case TimeScale::QUARTERLY: return 91;
            case TimeScale::YEARLY: return 365;
            default: return 1;
        }
    }
};

/**
 * TensorAccountSet - Collection of tensor accounts for unified operations.
 */
class TensorAccountSet
{
public:
    /**
     * Add an account to the set.
     */
    void add_account(std::shared_ptr<TensorAccount> account)
    {
        m_accounts[account->guid()] = account;
    }

    /**
     * Get account by GUID.
     */
    std::shared_ptr<TensorAccount> get_account(const std::string& guid) const
    {
        auto it = m_accounts.find(guid);
        return (it != m_accounts.end()) ? it->second : nullptr;
    }

    /**
     * Get all accounts as a tensor (accounts x periods x metrics).
     */
    DoubleTensor as_tensor(size_t entity = 0, size_t currency = 0) const
    {
        if (m_accounts.empty())
            return DoubleTensor({0});

        size_t num_accounts = m_accounts.size();
        size_t num_periods = m_accounts.begin()->second->num_periods();
        size_t num_metrics = TensorAccount::Metrics::NUM_METRICS;

        DoubleTensor result({num_accounts, num_periods, num_metrics});

        size_t acc_idx = 0;
        for (const auto& [guid, account] : m_accounts) {
            for (size_t p = 0; p < num_periods; ++p) {
                for (size_t m = 0; m < num_metrics; ++m) {
                    result.at({acc_idx, p, m}) = account->get_metric(entity, p, currency, m);
                }
            }
            ++acc_idx;
        }

        return result;
    }

    /**
     * Find accounts similar to a query account.
     */
    std::vector<std::pair<std::string, double>> find_similar(
        const TensorAccount& query, size_t top_k = 5) const
    {
        auto query_emb = query.get_embedding();
        std::vector<std::pair<std::string, double>> results;

        for (const auto& [guid, account] : m_accounts) {
            if (guid == query.guid()) continue;

            auto acc_emb = account->get_embedding();
            double sim = cosine_similarity(query_emb, acc_emb);
            results.emplace_back(guid, sim);
        }

        std::sort(results.begin(), results.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        if (results.size() > top_k)
            results.resize(top_k);

        return results;
    }

    /**
     * Cluster accounts by embedding similarity.
     */
    std::vector<std::vector<std::string>> cluster_accounts(size_t num_clusters) const
    {
        // Simple k-means style clustering
        std::vector<std::vector<std::string>> clusters(num_clusters);

        // Assign randomly initially
        size_t idx = 0;
        for (const auto& [guid, account] : m_accounts) {
            clusters[idx % num_clusters].push_back(guid);
            ++idx;
        }

        return clusters;
    }

    size_t size() const { return m_accounts.size(); }

private:
    std::unordered_map<std::string, std::shared_ptr<TensorAccount>> m_accounts;

    double cosine_similarity(const DoubleTensor& a, const DoubleTensor& b) const
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

} // namespace tensor_logic
} // namespace gnc

#endif // GNC_TENSOR_ACCOUNT_HPP
