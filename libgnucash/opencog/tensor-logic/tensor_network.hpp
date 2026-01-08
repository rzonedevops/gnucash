/*
 * opencog/tensor-logic/tensor_network.hpp
 *
 * Network-Aware Tensor Flow System
 * Models money flows as tensor networks between accounts
 *
 * Copyright (C) 2024 GnuCash Developers
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef GNC_TENSOR_NETWORK_HPP
#define GNC_TENSOR_NETWORK_HPP

#include "../aten/tensor.hpp"
#include "../aten/tensor_ops.hpp"
#include "tensor_account.hpp"

#include <unordered_map>
#include <unordered_set>
#include <queue>

namespace gnc {
namespace tensor_logic {

using namespace gnc::aten;

/**
 * Edge in the account network.
 */
struct NetworkEdge
{
    std::string source;
    std::string target;
    DoubleTensor flow_tensor;   // Flow amounts over time periods
    DoubleTensor weight_tensor; // Edge weights/importance
    double total_flow = 0.0;
    size_t transaction_count = 0;
};

/**
 * TensorNetwork - Graph representation of money flows between accounts.
 *
 * Models the financial system as a directed graph where:
 * - Nodes are accounts (with tensor representations)
 * - Edges are money flows (with temporal tensor data)
 *
 * Supports:
 * - Flow analysis and visualization
 * - Network centrality measures
 * - Flow prediction
 * - Anomaly detection in flow patterns
 */
class TensorNetwork
{
public:
    TensorNetwork(size_t num_periods = 12)
        : m_num_periods(num_periods)
    {}

    // =========================================
    // Network Construction
    // =========================================

    /**
     * Add an account node.
     */
    void add_node(const std::string& guid, const std::string& name = "")
    {
        if (m_nodes.find(guid) == m_nodes.end()) {
            m_nodes[guid] = name.empty() ? guid : name;
            m_adjacency[guid] = {};
            m_reverse_adjacency[guid] = {};
        }
    }

    /**
     * Add or update an edge (money flow).
     */
    void add_edge(const std::string& source, const std::string& target,
                  double amount, size_t period = 0)
    {
        add_node(source);
        add_node(target);

        std::string edge_key = source + "->" + target;

        if (m_edges.find(edge_key) == m_edges.end()) {
            NetworkEdge edge;
            edge.source = source;
            edge.target = target;
            edge.flow_tensor = DoubleTensor({m_num_periods}, 0.0);
            edge.weight_tensor = DoubleTensor({m_num_periods}, 0.0);
            m_edges[edge_key] = edge;

            m_adjacency[source].insert(target);
            m_reverse_adjacency[target].insert(source);
        }

        auto& edge = m_edges[edge_key];
        if (period < m_num_periods) {
            edge.flow_tensor[period] += amount;
            edge.weight_tensor[period] += 1.0;
        }
        edge.total_flow += amount;
        edge.transaction_count++;
    }

    /**
     * Record a transaction as edge.
     */
    void record_transaction(const std::string& from_account, const std::string& to_account,
                           double amount, size_t period)
    {
        add_edge(from_account, to_account, amount, period);
    }

    // =========================================
    // Network Properties
    // =========================================

    size_t num_nodes() const { return m_nodes.size(); }
    size_t num_edges() const { return m_edges.size(); }

    /**
     * Get all node GUIDs.
     */
    std::vector<std::string> get_nodes() const
    {
        std::vector<std::string> nodes;
        for (const auto& [guid, name] : m_nodes)
            nodes.push_back(guid);
        return nodes;
    }

    /**
     * Get outgoing neighbors.
     */
    std::vector<std::string> get_outgoing(const std::string& node) const
    {
        auto it = m_adjacency.find(node);
        if (it == m_adjacency.end()) return {};
        return std::vector<std::string>(it->second.begin(), it->second.end());
    }

    /**
     * Get incoming neighbors.
     */
    std::vector<std::string> get_incoming(const std::string& node) const
    {
        auto it = m_reverse_adjacency.find(node);
        if (it == m_reverse_adjacency.end()) return {};
        return std::vector<std::string>(it->second.begin(), it->second.end());
    }

    // =========================================
    // Flow Analysis
    // =========================================

    /**
     * Get flow tensor between two accounts.
     */
    DoubleTensor get_flow(const std::string& source, const std::string& target) const
    {
        std::string edge_key = source + "->" + target;
        auto it = m_edges.find(edge_key);
        if (it == m_edges.end())
            return DoubleTensor({m_num_periods}, 0.0);
        return it->second.flow_tensor;
    }

    /**
     * Get total outflow from an account.
     */
    DoubleTensor get_total_outflow(const std::string& node) const
    {
        DoubleTensor result({m_num_periods}, 0.0);
        auto it = m_adjacency.find(node);
        if (it == m_adjacency.end()) return result;

        for (const auto& target : it->second) {
            result += get_flow(node, target);
        }
        return result;
    }

    /**
     * Get total inflow to an account.
     */
    DoubleTensor get_total_inflow(const std::string& node) const
    {
        DoubleTensor result({m_num_periods}, 0.0);
        auto it = m_reverse_adjacency.find(node);
        if (it == m_reverse_adjacency.end()) return result;

        for (const auto& source : it->second) {
            result += get_flow(source, node);
        }
        return result;
    }

    /**
     * Get net flow for an account (inflow - outflow).
     */
    DoubleTensor get_net_flow(const std::string& node) const
    {
        return get_total_inflow(node) - get_total_outflow(node);
    }

    /**
     * Get adjacency matrix as tensor.
     */
    DoubleTensor get_adjacency_matrix() const
    {
        std::vector<std::string> nodes = get_nodes();
        size_t n = nodes.size();

        std::unordered_map<std::string, size_t> node_index;
        for (size_t i = 0; i < n; ++i)
            node_index[nodes[i]] = i;

        DoubleTensor adj({n, n}, 0.0);

        for (const auto& [key, edge] : m_edges) {
            size_t i = node_index[edge.source];
            size_t j = node_index[edge.target];
            adj(i, j) = edge.total_flow;
        }

        return adj;
    }

    /**
     * Get flow matrix for a specific period.
     */
    DoubleTensor get_flow_matrix(size_t period) const
    {
        std::vector<std::string> nodes = get_nodes();
        size_t n = nodes.size();

        std::unordered_map<std::string, size_t> node_index;
        for (size_t i = 0; i < n; ++i)
            node_index[nodes[i]] = i;

        DoubleTensor flow({n, n}, 0.0);

        for (const auto& [key, edge] : m_edges) {
            size_t i = node_index[edge.source];
            size_t j = node_index[edge.target];
            if (period < m_num_periods)
                flow(i, j) = edge.flow_tensor[period];
        }

        return flow;
    }

    // =========================================
    // Centrality Measures
    // =========================================

    /**
     * Compute degree centrality.
     */
    DoubleTensor degree_centrality() const
    {
        std::vector<std::string> nodes = get_nodes();
        size_t n = nodes.size();
        if (n == 0) return DoubleTensor({1}, 0.0);

        std::vector<double> centrality(n);
        for (size_t i = 0; i < n; ++i) {
            const auto& node = nodes[i];
            size_t degree = 0;
            auto it_out = m_adjacency.find(node);
            if (it_out != m_adjacency.end())
                degree += it_out->second.size();
            auto it_in = m_reverse_adjacency.find(node);
            if (it_in != m_reverse_adjacency.end())
                degree += it_in->second.size();
            centrality[i] = static_cast<double>(degree) / (2.0 * (n - 1));
        }

        return DoubleTensor({n}, std::move(centrality));
    }

    /**
     * Compute flow-weighted centrality.
     */
    DoubleTensor flow_centrality() const
    {
        std::vector<std::string> nodes = get_nodes();
        size_t n = nodes.size();
        if (n == 0) return DoubleTensor({1}, 0.0);

        double total_flow = 0.0;
        for (const auto& [key, edge] : m_edges)
            total_flow += edge.total_flow;

        std::vector<double> centrality(n);
        for (size_t i = 0; i < n; ++i) {
            const auto& node = nodes[i];
            double node_flow = 0.0;

            // Outgoing flow
            auto it_out = m_adjacency.find(node);
            if (it_out != m_adjacency.end()) {
                for (const auto& target : it_out->second) {
                    std::string key = node + "->" + target;
                    auto edge_it = m_edges.find(key);
                    if (edge_it != m_edges.end())
                        node_flow += edge_it->second.total_flow;
                }
            }

            // Incoming flow
            auto it_in = m_reverse_adjacency.find(node);
            if (it_in != m_reverse_adjacency.end()) {
                for (const auto& source : it_in->second) {
                    std::string key = source + "->" + node;
                    auto edge_it = m_edges.find(key);
                    if (edge_it != m_edges.end())
                        node_flow += edge_it->second.total_flow;
                }
            }

            centrality[i] = (total_flow > 0) ? node_flow / total_flow : 0.0;
        }

        return DoubleTensor({n}, std::move(centrality));
    }

    /**
     * Compute PageRank-style importance.
     */
    DoubleTensor pagerank(double damping = 0.85, size_t iterations = 100) const
    {
        std::vector<std::string> nodes = get_nodes();
        size_t n = nodes.size();
        if (n == 0) return DoubleTensor({1}, 0.0);

        std::unordered_map<std::string, size_t> node_index;
        for (size_t i = 0; i < n; ++i)
            node_index[nodes[i]] = i;

        DoubleTensor rank({n}, 1.0 / n);
        DoubleTensor new_rank({n}, 0.0);

        for (size_t iter = 0; iter < iterations; ++iter) {
            for (size_t i = 0; i < n; ++i)
                new_rank[i] = (1.0 - damping) / n;

            for (size_t i = 0; i < n; ++i) {
                const auto& node = nodes[i];
                auto it = m_adjacency.find(node);
                if (it == m_adjacency.end() || it->second.empty())
                    continue;

                double share = damping * rank[i] / it->second.size();
                for (const auto& target : it->second) {
                    size_t j = node_index[target];
                    new_rank[j] += share;
                }
            }

            std::swap(rank, new_rank);
        }

        return rank;
    }

    // =========================================
    // Path Analysis
    // =========================================

    /**
     * Find shortest path between accounts (BFS).
     */
    std::vector<std::string> shortest_path(const std::string& source,
                                           const std::string& target) const
    {
        if (m_nodes.find(source) == m_nodes.end() ||
            m_nodes.find(target) == m_nodes.end())
            return {};

        std::unordered_map<std::string, std::string> parent;
        std::queue<std::string> queue;
        std::unordered_set<std::string> visited;

        queue.push(source);
        visited.insert(source);
        parent[source] = "";

        while (!queue.empty()) {
            std::string current = queue.front();
            queue.pop();

            if (current == target) {
                // Reconstruct path
                std::vector<std::string> path;
                std::string node = target;
                while (!node.empty()) {
                    path.push_back(node);
                    node = parent[node];
                }
                std::reverse(path.begin(), path.end());
                return path;
            }

            auto it = m_adjacency.find(current);
            if (it != m_adjacency.end()) {
                for (const auto& neighbor : it->second) {
                    if (visited.find(neighbor) == visited.end()) {
                        visited.insert(neighbor);
                        parent[neighbor] = current;
                        queue.push(neighbor);
                    }
                }
            }
        }

        return {};  // No path found
    }

    /**
     * Find all paths up to max length.
     */
    std::vector<std::vector<std::string>> find_all_paths(
        const std::string& source, const std::string& target, size_t max_length = 5) const
    {
        std::vector<std::vector<std::string>> all_paths;
        std::vector<std::string> current_path;
        std::unordered_set<std::string> visited;

        find_paths_dfs(source, target, current_path, visited, all_paths, max_length);
        return all_paths;
    }

    // =========================================
    // Anomaly Detection
    // =========================================

    /**
     * Detect unusual flows (anomalies).
     */
    std::vector<std::pair<std::string, double>> detect_flow_anomalies(
        double threshold = 2.0) const
    {
        std::vector<std::pair<std::string, double>> anomalies;

        // Calculate flow statistics
        std::vector<double> all_flows;
        for (const auto& [key, edge] : m_edges) {
            for (size_t p = 0; p < m_num_periods; ++p) {
                if (edge.flow_tensor[p] > 0)
                    all_flows.push_back(edge.flow_tensor[p]);
            }
        }

        if (all_flows.empty()) return anomalies;

        DoubleTensor flow_tensor({all_flows.size()}, std::move(all_flows));
        double mean = flow_tensor.mean();
        double stddev = flow_tensor.std();

        // Find anomalies
        for (const auto& [key, edge] : m_edges) {
            for (size_t p = 0; p < m_num_periods; ++p) {
                double flow = edge.flow_tensor[p];
                if (flow > 0) {
                    double z_score = (flow - mean) / stddev;
                    if (std::abs(z_score) > threshold) {
                        anomalies.emplace_back(key + "@" + std::to_string(p), z_score);
                    }
                }
            }
        }

        // Sort by anomaly score
        std::sort(anomalies.begin(), anomalies.end(),
                  [](const auto& a, const auto& b) {
                      return std::abs(a.second) > std::abs(b.second);
                  });

        return anomalies;
    }

    /**
     * Detect circular flows (potential issues).
     */
    std::vector<std::vector<std::string>> detect_circular_flows(size_t max_cycle_length = 5) const
    {
        std::vector<std::vector<std::string>> cycles;

        for (const auto& [start_node, name] : m_nodes) {
            std::vector<std::string> path;
            std::unordered_set<std::string> visited;
            find_cycles_dfs(start_node, start_node, path, visited, cycles, max_cycle_length);
        }

        return cycles;
    }

    // =========================================
    // Embedding
    // =========================================

    /**
     * Generate network embedding for each node.
     */
    DoubleTensor generate_node_embeddings(size_t embedding_dim = 32) const
    {
        std::vector<std::string> nodes = get_nodes();
        size_t n = nodes.size();
        if (n == 0) return DoubleTensor({1, embedding_dim}, 0.0);

        DoubleTensor embeddings({n, embedding_dim});

        // Simple embedding based on flow features
        for (size_t i = 0; i < n; ++i) {
            const auto& node = nodes[i];

            auto inflow = get_total_inflow(node);
            auto outflow = get_total_outflow(node);
            auto net = get_net_flow(node);

            // Features: normalized flows, statistics
            size_t dim = 0;

            // Flow statistics
            embeddings.at({i, dim++}) = inflow.sum();
            embeddings.at({i, dim++}) = outflow.sum();
            embeddings.at({i, dim++}) = net.sum();
            embeddings.at({i, dim++}) = inflow.mean();
            embeddings.at({i, dim++}) = outflow.mean();
            embeddings.at({i, dim++}) = inflow.std();
            embeddings.at({i, dim++}) = outflow.std();

            // Connectivity
            auto it_out = m_adjacency.find(node);
            auto it_in = m_reverse_adjacency.find(node);
            embeddings.at({i, dim++}) = it_out != m_adjacency.end() ? it_out->second.size() : 0;
            embeddings.at({i, dim++}) = it_in != m_reverse_adjacency.end() ? it_in->second.size() : 0;

            // Pad remaining dimensions
            while (dim < embedding_dim) {
                embeddings.at({i, dim++}) = 0.0;
            }
        }

        return embeddings;
    }

private:
    size_t m_num_periods;
    std::unordered_map<std::string, std::string> m_nodes;  // guid -> name
    std::unordered_map<std::string, NetworkEdge> m_edges;  // "src->tgt" -> edge
    std::unordered_map<std::string, std::unordered_set<std::string>> m_adjacency;
    std::unordered_map<std::string, std::unordered_set<std::string>> m_reverse_adjacency;

    void find_paths_dfs(const std::string& current, const std::string& target,
                        std::vector<std::string>& path, std::unordered_set<std::string>& visited,
                        std::vector<std::vector<std::string>>& all_paths, size_t max_length) const
    {
        if (path.size() >= max_length) return;

        visited.insert(current);
        path.push_back(current);

        if (current == target && path.size() > 1) {
            all_paths.push_back(path);
        } else {
            auto it = m_adjacency.find(current);
            if (it != m_adjacency.end()) {
                for (const auto& next : it->second) {
                    if (visited.find(next) == visited.end()) {
                        find_paths_dfs(next, target, path, visited, all_paths, max_length);
                    }
                }
            }
        }

        path.pop_back();
        visited.erase(current);
    }

    void find_cycles_dfs(const std::string& start, const std::string& current,
                         std::vector<std::string>& path, std::unordered_set<std::string>& visited,
                         std::vector<std::vector<std::string>>& cycles, size_t max_length) const
    {
        if (path.size() >= max_length) return;

        path.push_back(current);

        auto it = m_adjacency.find(current);
        if (it != m_adjacency.end()) {
            for (const auto& next : it->second) {
                if (next == start && path.size() > 2) {
                    std::vector<std::string> cycle = path;
                    cycle.push_back(start);
                    cycles.push_back(cycle);
                } else if (visited.find(next) == visited.end()) {
                    visited.insert(next);
                    find_cycles_dfs(start, next, path, visited, cycles, max_length);
                    visited.erase(next);
                }
            }
        }

        path.pop_back();
    }
};

} // namespace tensor_logic
} // namespace gnc

#endif // GNC_TENSOR_NETWORK_HPP
