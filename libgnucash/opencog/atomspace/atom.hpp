/*
 * opencog/atomspace/atom.hpp
 *
 * Atom - the fundamental unit of knowledge in AtomSpace
 * Based on OpenCog Atom design
 *
 * Copyright (C) 2024 GnuCash Developers
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef GNC_OPENCOG_ATOM_HPP
#define GNC_OPENCOG_ATOM_HPP

#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <functional>

#include "atom_types.hpp"
#include "truth_value.hpp"

namespace gnc {
namespace opencog {

// Forward declarations
class Atom;
class AtomSpace;

/**
 * Handle - a reference to an Atom.
 * This is the primary way to refer to atoms in the AtomSpace.
 */
using Handle = std::shared_ptr<Atom>;
using HandleSeq = std::vector<Handle>;
using HandleSet = std::vector<Handle>; // Could use unordered_set with custom hash

/**
 * Unique identifier for atoms.
 */
using UUID = uint64_t;

/**
 * Atom - the fundamental unit of knowledge representation.
 *
 * An Atom is either a Node (with a name) or a Link (with outgoing atoms).
 * Every atom has:
 * - A type (AtomType)
 * - A truth value (probabilistic truth)
 * - An attention value (importance/salience)
 *
 * Atoms are immutable once created - their type and structure cannot change.
 */
class Atom : public std::enable_shared_from_this<Atom>
{
public:
    virtual ~Atom() = default;

    /**
     * Get the type of this atom.
     */
    AtomType type() const { return m_type; }

    /**
     * Get the type name as a string.
     */
    const char* type_name() const { return atom_type_name(m_type); }

    /**
     * Check if this atom is a node.
     */
    virtual bool is_node() const = 0;

    /**
     * Check if this atom is a link.
     */
    virtual bool is_link() const = 0;

    /**
     * Get the name (for nodes).
     */
    virtual const std::string& name() const = 0;

    /**
     * Get the outgoing set (for links).
     */
    virtual const HandleSeq& outgoing() const = 0;

    /**
     * Get the arity (0 for nodes, number of outgoing atoms for links).
     */
    virtual size_t arity() const = 0;

    /**
     * Get the n-th outgoing atom (for links).
     */
    virtual Handle outgoing_atom(size_t index) const = 0;

    /**
     * Get the truth value.
     */
    const TruthValue& truth_value() const { return m_truth_value; }

    /**
     * Set the truth value.
     */
    void set_truth_value(const TruthValue& tv) { m_truth_value = tv; }

    /**
     * Get the attention value (short-term importance).
     */
    double sti() const { return m_sti; }

    /**
     * Get the long-term importance.
     */
    double lti() const { return m_lti; }

    /**
     * Set attention values.
     */
    void set_sti(double sti) { m_sti = sti; }
    void set_lti(double lti) { m_lti = lti; }

    /**
     * Get the unique identifier.
     */
    UUID uuid() const { return m_uuid; }

    /**
     * Get string representation.
     */
    virtual std::string to_string() const = 0;

    /**
     * Get short string representation.
     */
    virtual std::string to_short_string() const = 0;

    /**
     * Compute hash for this atom.
     */
    virtual size_t hash() const = 0;

    /**
     * Check equality with another atom.
     */
    virtual bool operator==(const Atom& other) const = 0;

protected:
    Atom(AtomType type)
        : m_type(type)
        , m_uuid(next_uuid())
        , m_sti(0.0)
        , m_lti(0.0)
    {}

    static UUID next_uuid()
    {
        static std::atomic<UUID> counter{1};
        return counter++;
    }

    AtomType m_type;
    UUID m_uuid;
    TruthValue m_truth_value;
    double m_sti;  // Short-term importance
    double m_lti;  // Long-term importance
};

/**
 * Node - an atom with a name but no outgoing links.
 *
 * Nodes represent basic concepts, entities, or values.
 * Examples: ConceptNode "money", AccountNode "Expenses:Food"
 */
class Node : public Atom
{
public:
    Node(AtomType type, const std::string& name)
        : Atom(type)
        , m_name(name)
    {}

    bool is_node() const override { return true; }
    bool is_link() const override { return false; }

    const std::string& name() const override { return m_name; }

    const HandleSeq& outgoing() const override
    {
        static const HandleSeq empty;
        return empty;
    }

    size_t arity() const override { return 0; }

    Handle outgoing_atom(size_t) const override { return nullptr; }

    std::string to_string() const override
    {
        return std::string("(") + type_name() + " \"" + m_name + "\"" +
               (m_truth_value.is_default() ? "" : " " + m_truth_value.to_string()) + ")";
    }

    std::string to_short_string() const override
    {
        return std::string(type_name()) + ":\"" + m_name + "\"";
    }

    size_t hash() const override
    {
        size_t h = std::hash<AtomType>{}(m_type);
        h ^= std::hash<std::string>{}(m_name) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }

    bool operator==(const Atom& other) const override
    {
        if (m_type != other.type()) return false;
        if (!other.is_node()) return false;
        return m_name == other.name();
    }

private:
    std::string m_name;
};

/**
 * Link - an atom that connects other atoms.
 *
 * Links represent relationships between atoms.
 * Examples: InheritanceLink, EvaluationLink, TransactionLink
 */
class Link : public Atom
{
public:
    Link(AtomType type, const HandleSeq& outgoing)
        : Atom(type)
        , m_outgoing(outgoing)
    {}

    Link(AtomType type, HandleSeq&& outgoing)
        : Atom(type)
        , m_outgoing(std::move(outgoing))
    {}

    template<typename... Handles>
    Link(AtomType type, Handles&&... handles)
        : Atom(type)
        , m_outgoing{std::forward<Handles>(handles)...}
    {}

    bool is_node() const override { return false; }
    bool is_link() const override { return true; }

    const std::string& name() const override
    {
        static const std::string empty;
        return empty;
    }

    const HandleSeq& outgoing() const override { return m_outgoing; }

    size_t arity() const override { return m_outgoing.size(); }

    Handle outgoing_atom(size_t index) const override
    {
        return (index < m_outgoing.size()) ? m_outgoing[index] : nullptr;
    }

    std::string to_string() const override
    {
        std::string result = std::string("(") + type_name();
        if (!m_truth_value.is_default())
            result += " " + m_truth_value.to_string();
        for (const auto& h : m_outgoing) {
            result += "\n  " + (h ? h->to_string() : "null");
        }
        result += ")";
        return result;
    }

    std::string to_short_string() const override
    {
        std::string result = std::string(type_name()) + "(";
        for (size_t i = 0; i < m_outgoing.size(); ++i) {
            if (i > 0) result += ", ";
            result += m_outgoing[i] ? m_outgoing[i]->to_short_string() : "null";
        }
        result += ")";
        return result;
    }

    size_t hash() const override
    {
        size_t h = std::hash<AtomType>{}(m_type);
        for (const auto& atom : m_outgoing) {
            if (atom)
                h ^= atom->hash() + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }

    bool operator==(const Atom& other) const override
    {
        if (m_type != other.type()) return false;
        if (!other.is_link()) return false;
        const auto& other_out = other.outgoing();
        if (m_outgoing.size() != other_out.size()) return false;
        for (size_t i = 0; i < m_outgoing.size(); ++i) {
            if (!m_outgoing[i] || !other_out[i]) {
                if (m_outgoing[i] != other_out[i]) return false;
            } else if (*m_outgoing[i] != *other_out[i]) {
                return false;
            }
        }
        return true;
    }

private:
    HandleSeq m_outgoing;
};

/**
 * Factory functions for creating atoms.
 */
inline Handle create_node(AtomType type, const std::string& name)
{
    return std::make_shared<Node>(type, name);
}

inline Handle create_link(AtomType type, const HandleSeq& outgoing)
{
    return std::make_shared<Link>(type, outgoing);
}

template<typename... Handles>
Handle create_link(AtomType type, Handles&&... handles)
{
    return std::make_shared<Link>(type, std::forward<Handles>(handles)...);
}

} // namespace opencog
} // namespace gnc

#endif // GNC_OPENCOG_ATOM_HPP
