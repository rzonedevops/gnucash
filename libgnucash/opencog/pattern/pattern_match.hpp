/*
 * opencog/pattern/pattern_match.hpp
 *
 * Pattern matching engine for AtomSpace queries
 * Based on OpenCog pattern matcher
 *
 * Copyright (C) 2024 GnuCash Developers
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef GNC_OPENCOG_PATTERN_MATCH_HPP
#define GNC_OPENCOG_PATTERN_MATCH_HPP

#include <unordered_map>
#include <functional>
#include <optional>

#include "../atomspace/atomspace.hpp"

namespace gnc {
namespace opencog {

/**
 * Variable bindings - maps variable names to bound atoms.
 */
using Bindings = std::unordered_map<std::string, Handle>;

/**
 * Result callback for pattern matching.
 */
using MatchCallback = std::function<bool(const Bindings&)>;

/**
 * Pattern - represents a pattern to match against the AtomSpace.
 *
 * A pattern consists of:
 * - Variable declarations (atoms that should be matched)
 * - A body (the structure to match)
 * - Optional clauses (constraints)
 */
class Pattern
{
public:
    Pattern() = default;

    /**
     * Add a variable to the pattern.
     */
    Pattern& add_variable(const std::string& name, AtomType type = AtomTypes::ATOM)
    {
        m_variables[name] = type;
        return *this;
    }

    /**
     * Set the pattern body.
     */
    Pattern& set_body(const Handle& body)
    {
        m_body = body;
        return *this;
    }

    /**
     * Add a constraint clause.
     */
    Pattern& add_clause(const Handle& clause)
    {
        m_clauses.push_back(clause);
        return *this;
    }

    /**
     * Get the variables.
     */
    const std::unordered_map<std::string, AtomType>& variables() const
    {
        return m_variables;
    }

    /**
     * Get the body.
     */
    const Handle& body() const { return m_body; }

    /**
     * Get the clauses.
     */
    const HandleSeq& clauses() const { return m_clauses; }

    /**
     * Check if a name is a variable.
     */
    bool is_variable(const std::string& name) const
    {
        return m_variables.find(name) != m_variables.end();
    }

private:
    std::unordered_map<std::string, AtomType> m_variables;
    Handle m_body;
    HandleSeq m_clauses;
};

/**
 * PatternMatcher - executes pattern matching queries against AtomSpace.
 *
 * The pattern matcher implements graph unification to find subgraphs
 * in the AtomSpace that match a given pattern.
 */
class PatternMatcher
{
public:
    explicit PatternMatcher(AtomSpace& atomspace)
        : m_atomspace(atomspace)
    {}

    /**
     * Execute a pattern match and return all matching bindings.
     */
    std::vector<Bindings> match(const Pattern& pattern)
    {
        std::vector<Bindings> results;
        match(pattern, [&results](const Bindings& bindings) {
            results.push_back(bindings);
            return true;  // Continue searching
        });
        return results;
    }

    /**
     * Execute a pattern match with a callback for each result.
     * Return false from callback to stop searching.
     */
    void match(const Pattern& pattern, const MatchCallback& callback)
    {
        if (!pattern.body()) return;

        Bindings bindings;
        match_recursive(pattern, pattern.body(), bindings, callback);
    }

    /**
     * Find atoms matching a simple type pattern.
     */
    HandleSeq find_by_type(AtomType type)
    {
        return m_atomspace.get_atoms_by_type(type);
    }

    /**
     * Find nodes by type and name pattern (supports wildcards).
     */
    HandleSeq find_nodes(AtomType type, const std::string& name_pattern = "*")
    {
        auto atoms = m_atomspace.get_atoms_by_type(type);
        if (name_pattern == "*")
            return atoms;

        HandleSeq result;
        for (const auto& atom : atoms) {
            if (atom->is_node() && match_name(atom->name(), name_pattern))
                result.push_back(atom);
        }
        return result;
    }

    /**
     * Find links by type with specific target.
     */
    HandleSeq find_links_to(AtomType link_type, const Handle& target)
    {
        return m_atomspace.get_incoming_by_type(target, link_type);
    }

    /**
     * Find links by type containing specific atom.
     */
    HandleSeq find_links_containing(AtomType link_type, const Handle& atom)
    {
        HandleSeq result;
        auto links = m_atomspace.get_atoms_by_type(link_type);
        for (const auto& link : links) {
            for (const auto& out : link->outgoing()) {
                if (out && *out == *atom) {
                    result.push_back(link);
                    break;
                }
            }
        }
        return result;
    }

private:
    /**
     * Recursive pattern matching implementation.
     */
    bool match_recursive(const Pattern& pattern, const Handle& pat,
                        Bindings& bindings, const MatchCallback& callback)
    {
        if (!pat) return false;

        // Check if this is a variable node
        if (pat->is_node() && pat->type() == AtomTypes::VARIABLE_NODE) {
            const std::string& var_name = pat->name();

            // Check if variable is already bound
            auto it = bindings.find(var_name);
            if (it != bindings.end()) {
                // Variable already bound - this is a constraint
                return true;
            }

            // Find candidates for this variable
            AtomType var_type = AtomTypes::ATOM;
            auto type_it = pattern.variables().find(var_name);
            if (type_it != pattern.variables().end())
                var_type = type_it->second;

            HandleSeq candidates;
            if (var_type == AtomTypes::ATOM) {
                m_atomspace.for_each([&candidates](const Handle& h) {
                    candidates.push_back(h);
                });
            } else {
                candidates = m_atomspace.get_atoms_by_type(var_type);
            }

            // Try each candidate
            for (const auto& candidate : candidates) {
                bindings[var_name] = candidate;
                if (!callback(bindings))
                    return false;
            }

            bindings.erase(var_name);
            return true;
        }

        // For non-variable nodes, find exact matches
        if (pat->is_node()) {
            Handle match = m_atomspace.get_node(pat->type(), pat->name());
            if (match)
                return callback(bindings);
            return true;
        }

        // For links, match the structure
        if (pat->is_link()) {
            auto candidates = m_atomspace.get_atoms_by_type(pat->type());
            for (const auto& candidate : candidates) {
                if (candidate->arity() != pat->arity())
                    continue;

                bool matches = true;
                Bindings local_bindings = bindings;

                for (size_t i = 0; i < pat->arity() && matches; ++i) {
                    Handle pat_out = pat->outgoing_atom(i);
                    Handle cand_out = candidate->outgoing_atom(i);

                    if (pat_out && pat_out->is_node() &&
                        pat_out->type() == AtomTypes::VARIABLE_NODE) {
                        // Variable - bind or check
                        const std::string& var_name = pat_out->name();
                        auto it = local_bindings.find(var_name);
                        if (it != local_bindings.end()) {
                            if (!cand_out || *it->second != *cand_out)
                                matches = false;
                        } else {
                            local_bindings[var_name] = cand_out;
                        }
                    } else if (pat_out && cand_out) {
                        // Non-variable - must match exactly
                        if (*pat_out != *cand_out)
                            matches = false;
                    } else if (pat_out || cand_out) {
                        matches = false;
                    }
                }

                if (matches) {
                    if (!callback(local_bindings))
                        return false;
                }
            }
        }

        return true;
    }

    /**
     * Simple wildcard name matching.
     */
    bool match_name(const std::string& name, const std::string& pattern)
    {
        if (pattern == "*") return true;
        if (pattern.empty()) return name.empty();

        // Simple prefix/suffix wildcard support
        if (pattern.front() == '*' && pattern.back() == '*') {
            std::string sub = pattern.substr(1, pattern.size() - 2);
            return name.find(sub) != std::string::npos;
        }
        if (pattern.front() == '*') {
            std::string suffix = pattern.substr(1);
            return name.size() >= suffix.size() &&
                   name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
        }
        if (pattern.back() == '*') {
            std::string prefix = pattern.substr(0, pattern.size() - 1);
            return name.compare(0, prefix.size(), prefix) == 0;
        }

        return name == pattern;
    }

    AtomSpace& m_atomspace;
};

/**
 * Query builder for constructing patterns fluently.
 */
class QueryBuilder
{
public:
    explicit QueryBuilder(AtomSpace& atomspace)
        : m_atomspace(atomspace)
    {}

    /**
     * Declare a variable.
     */
    QueryBuilder& var(const std::string& name, AtomType type = AtomTypes::ATOM)
    {
        m_pattern.add_variable(name, type);
        return *this;
    }

    /**
     * Set the query body.
     */
    QueryBuilder& body(const Handle& b)
    {
        m_pattern.set_body(b);
        return *this;
    }

    /**
     * Add a constraint clause.
     */
    QueryBuilder& where(const Handle& clause)
    {
        m_pattern.add_clause(clause);
        return *this;
    }

    /**
     * Execute the query.
     */
    std::vector<Bindings> execute()
    {
        PatternMatcher matcher(m_atomspace);
        return matcher.match(m_pattern);
    }

    /**
     * Execute with callback.
     */
    void execute(const MatchCallback& callback)
    {
        PatternMatcher matcher(m_atomspace);
        matcher.match(m_pattern, callback);
    }

private:
    AtomSpace& m_atomspace;
    Pattern m_pattern;
};

} // namespace opencog
} // namespace gnc

#endif // GNC_OPENCOG_PATTERN_MATCH_HPP
