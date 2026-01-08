/*
 * opencog/atomspace/atomspace.hpp
 *
 * AtomSpace - the hypergraph knowledge database
 * Based on OpenCog AtomSpace design
 *
 * Copyright (C) 2024 GnuCash Developers
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef GNC_OPENCOG_ATOMSPACE_HPP
#define GNC_OPENCOG_ATOMSPACE_HPP

#include <unordered_map>
#include <unordered_set>
#include <shared_mutex>
#include <functional>
#include <optional>

#include "atom.hpp"
#include "atom_types.hpp"

namespace gnc {
namespace opencog {

/**
 * Hash function for atoms (used in AtomSpace storage).
 */
struct AtomHash
{
    size_t operator()(const Handle& h) const
    {
        return h ? h->hash() : 0;
    }
};

/**
 * Equality function for atoms.
 */
struct AtomEqual
{
    bool operator()(const Handle& a, const Handle& b) const
    {
        if (!a || !b) return a == b;
        return *a == *b;
    }
};

/**
 * AtomSpace - the hypergraph knowledge database.
 *
 * The AtomSpace is a typed, labeled hypergraph database that stores
 * atoms (nodes and links) representing knowledge. It provides:
 * - Efficient storage and retrieval of atoms
 * - Type-based indexing
 * - Incoming set tracking (which links point to an atom)
 * - Thread-safe operations
 *
 * In GnuCash context, the AtomSpace stores:
 * - Account structures and hierarchies
 * - Transaction patterns and relationships
 * - Learned categorization rules
 * - Predictions and anomalies
 */
class AtomSpace
{
public:
    using AtomSet = std::unordered_set<Handle, AtomHash, AtomEqual>;
    using AtomPredicate = std::function<bool(const Handle&)>;

    AtomSpace() = default;
    ~AtomSpace() = default;

    // Non-copyable
    AtomSpace(const AtomSpace&) = delete;
    AtomSpace& operator=(const AtomSpace&) = delete;

    /**
     * Add a node to the AtomSpace.
     * If an equivalent node already exists, returns the existing one.
     */
    Handle add_node(AtomType type, const std::string& name)
    {
        auto node = create_node(type, name);
        return add_atom(node);
    }

    /**
     * Add a link to the AtomSpace.
     * If an equivalent link already exists, returns the existing one.
     */
    Handle add_link(AtomType type, const HandleSeq& outgoing)
    {
        auto link = create_link(type, outgoing);
        return add_atom(link);
    }

    /**
     * Add a link with variadic arguments.
     */
    template<typename... Handles>
    Handle add_link(AtomType type, Handles&&... handles)
    {
        HandleSeq outgoing{std::forward<Handles>(handles)...};
        return add_link(type, outgoing);
    }

    /**
     * Add an atom to the AtomSpace.
     * If an equivalent atom already exists, returns the existing one.
     */
    Handle add_atom(const Handle& atom)
    {
        if (!atom) return nullptr;

        std::unique_lock<std::shared_mutex> lock(m_mutex);

        // Check if atom already exists
        auto it = m_atoms.find(atom);
        if (it != m_atoms.end())
            return *it;

        // Add the atom
        m_atoms.insert(atom);
        m_by_type[atom->type()].insert(atom);
        m_by_uuid[atom->uuid()] = atom;

        // Track incoming set for links
        if (atom->is_link()) {
            for (const auto& target : atom->outgoing()) {
                if (target)
                    m_incoming[target->uuid()].insert(atom);
            }
        }

        return atom;
    }

    /**
     * Remove an atom from the AtomSpace.
     * Returns true if the atom was found and removed.
     */
    bool remove_atom(const Handle& atom)
    {
        if (!atom) return false;

        std::unique_lock<std::shared_mutex> lock(m_mutex);

        auto it = m_atoms.find(atom);
        if (it == m_atoms.end())
            return false;

        // Remove from type index
        m_by_type[atom->type()].erase(atom);

        // Remove from UUID index
        m_by_uuid.erase(atom->uuid());

        // Remove from incoming sets
        if (atom->is_link()) {
            for (const auto& target : atom->outgoing()) {
                if (target)
                    m_incoming[target->uuid()].erase(atom);
            }
        }

        // Remove the atom itself
        m_atoms.erase(it);
        return true;
    }

    /**
     * Get an atom by its UUID.
     */
    Handle get_atom(UUID uuid) const
    {
        std::shared_lock<std::shared_mutex> lock(m_mutex);
        auto it = m_by_uuid.find(uuid);
        return (it != m_by_uuid.end()) ? it->second : nullptr;
    }

    /**
     * Get a node by type and name.
     */
    Handle get_node(AtomType type, const std::string& name) const
    {
        auto temp = create_node(type, name);
        std::shared_lock<std::shared_mutex> lock(m_mutex);
        auto it = m_atoms.find(temp);
        return (it != m_atoms.end()) ? *it : nullptr;
    }

    /**
     * Get a link by type and outgoing set.
     */
    Handle get_link(AtomType type, const HandleSeq& outgoing) const
    {
        auto temp = create_link(type, outgoing);
        std::shared_lock<std::shared_mutex> lock(m_mutex);
        auto it = m_atoms.find(temp);
        return (it != m_atoms.end()) ? *it : nullptr;
    }

    /**
     * Get all atoms of a specific type.
     */
    HandleSeq get_atoms_by_type(AtomType type, bool subclasses = false) const
    {
        std::shared_lock<std::shared_mutex> lock(m_mutex);
        HandleSeq result;

        auto it = m_by_type.find(type);
        if (it != m_by_type.end()) {
            result.insert(result.end(), it->second.begin(), it->second.end());
        }

        // TODO: Add subclass handling if needed
        (void)subclasses;

        return result;
    }

    /**
     * Get the incoming set for an atom (all links pointing to it).
     */
    HandleSeq get_incoming(const Handle& atom) const
    {
        if (!atom) return {};

        std::shared_lock<std::shared_mutex> lock(m_mutex);
        auto it = m_incoming.find(atom->uuid());
        if (it == m_incoming.end())
            return {};

        return HandleSeq(it->second.begin(), it->second.end());
    }

    /**
     * Get incoming links of a specific type.
     */
    HandleSeq get_incoming_by_type(const Handle& atom, AtomType type) const
    {
        HandleSeq incoming = get_incoming(atom);
        HandleSeq result;
        for (const auto& link : incoming) {
            if (link->type() == type)
                result.push_back(link);
        }
        return result;
    }

    /**
     * Find atoms matching a predicate.
     */
    HandleSeq find_atoms(const AtomPredicate& predicate) const
    {
        std::shared_lock<std::shared_mutex> lock(m_mutex);
        HandleSeq result;
        for (const auto& atom : m_atoms) {
            if (predicate(atom))
                result.push_back(atom);
        }
        return result;
    }

    /**
     * Get total number of atoms.
     */
    size_t size() const
    {
        std::shared_lock<std::shared_mutex> lock(m_mutex);
        return m_atoms.size();
    }

    /**
     * Get number of atoms of a specific type.
     */
    size_t size(AtomType type) const
    {
        std::shared_lock<std::shared_mutex> lock(m_mutex);
        auto it = m_by_type.find(type);
        return (it != m_by_type.end()) ? it->second.size() : 0;
    }

    /**
     * Check if an atom exists in the AtomSpace.
     */
    bool contains(const Handle& atom) const
    {
        if (!atom) return false;
        std::shared_lock<std::shared_mutex> lock(m_mutex);
        return m_atoms.find(atom) != m_atoms.end();
    }

    /**
     * Clear all atoms from the AtomSpace.
     */
    void clear()
    {
        std::unique_lock<std::shared_mutex> lock(m_mutex);
        m_atoms.clear();
        m_by_type.clear();
        m_by_uuid.clear();
        m_incoming.clear();
    }

    /**
     * Execute a function for each atom.
     */
    void for_each(const std::function<void(const Handle&)>& func) const
    {
        std::shared_lock<std::shared_mutex> lock(m_mutex);
        for (const auto& atom : m_atoms)
            func(atom);
    }

    /**
     * Get statistics about the AtomSpace.
     */
    struct Stats
    {
        size_t total_atoms;
        size_t total_nodes;
        size_t total_links;
        std::unordered_map<AtomType, size_t> by_type;
    };

    Stats get_stats() const
    {
        std::shared_lock<std::shared_mutex> lock(m_mutex);
        Stats stats;
        stats.total_atoms = m_atoms.size();
        stats.total_nodes = 0;
        stats.total_links = 0;

        for (const auto& atom : m_atoms) {
            if (atom->is_node())
                ++stats.total_nodes;
            else
                ++stats.total_links;
        }

        for (const auto& [type, atoms] : m_by_type)
            stats.by_type[type] = atoms.size();

        return stats;
    }

private:
    mutable std::shared_mutex m_mutex;

    // Primary storage
    AtomSet m_atoms;

    // Indexes
    std::unordered_map<AtomType, AtomSet> m_by_type;
    std::unordered_map<UUID, Handle> m_by_uuid;

    // Incoming set index (which links point to each atom)
    std::unordered_map<UUID, AtomSet> m_incoming;
};

} // namespace opencog
} // namespace gnc

#endif // GNC_OPENCOG_ATOMSPACE_HPP
