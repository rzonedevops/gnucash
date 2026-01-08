/*
 * opencog/atomspace/truth_value.hpp
 *
 * Truth value representation for probabilistic logic
 * Based on OpenCog PLN truth values
 *
 * Copyright (C) 2024 GnuCash Developers
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef GNC_OPENCOG_TRUTH_VALUE_HPP
#define GNC_OPENCOG_TRUTH_VALUE_HPP

#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <memory>

namespace gnc {
namespace opencog {

/**
 * Strength value type (probability between 0 and 1).
 */
using Strength = double;

/**
 * Confidence value type (between 0 and 1).
 */
using Confidence = double;

/**
 * Count type for evidence counting.
 */
using Count = double;

/**
 * Simple Truth Value - the most common truth value type.
 * Represents a probability estimate with confidence.
 *
 * Based on OpenCog's SimpleTruthValue which uses:
 * - Strength: the probability estimate (0 to 1)
 * - Confidence: how confident we are in this estimate (0 to 1)
 *
 * Confidence is related to count by: confidence = count / (count + k)
 * where k is a constant (default 800 in OpenCog).
 */
class TruthValue
{
public:
    static constexpr double DEFAULT_K = 800.0;
    static constexpr Strength DEFAULT_STRENGTH = 0.0;
    static constexpr Confidence DEFAULT_CONFIDENCE = 0.0;

    /**
     * Default constructor - creates a default (unknown) truth value.
     */
    TruthValue()
        : m_strength(DEFAULT_STRENGTH)
        , m_confidence(DEFAULT_CONFIDENCE)
    {}

    /**
     * Construct with strength and confidence.
     */
    TruthValue(Strength strength, Confidence confidence)
        : m_strength(std::clamp(strength, 0.0, 1.0))
        , m_confidence(std::clamp(confidence, 0.0, 1.0))
    {}

    /**
     * Construct from count-based evidence.
     */
    static TruthValue from_count(Strength strength, Count count, double k = DEFAULT_K)
    {
        Confidence conf = count / (count + k);
        return TruthValue(strength, conf);
    }

    // Accessors
    Strength strength() const { return m_strength; }
    Confidence confidence() const { return m_confidence; }

    /**
     * Get the count equivalent of the confidence.
     */
    Count count(double k = DEFAULT_K) const
    {
        if (m_confidence >= 1.0) return std::numeric_limits<double>::infinity();
        return k * m_confidence / (1.0 - m_confidence);
    }

    // Setters
    void set_strength(Strength s) { m_strength = std::clamp(s, 0.0, 1.0); }
    void set_confidence(Confidence c) { m_confidence = std::clamp(c, 0.0, 1.0); }

    /**
     * Check if this is a default/unknown truth value.
     */
    bool is_default() const
    {
        return m_strength == DEFAULT_STRENGTH && m_confidence == DEFAULT_CONFIDENCE;
    }

    /**
     * Check if this truth value indicates "true" (high strength and confidence).
     */
    bool is_true(Strength threshold = 0.5, Confidence conf_threshold = 0.5) const
    {
        return m_strength >= threshold && m_confidence >= conf_threshold;
    }

    /**
     * Check if this truth value indicates "false" (low strength, high confidence).
     */
    bool is_false(Strength threshold = 0.5, Confidence conf_threshold = 0.5) const
    {
        return m_strength < threshold && m_confidence >= conf_threshold;
    }

    /**
     * Merge two truth values (average weighted by confidence).
     */
    TruthValue merge(const TruthValue& other) const
    {
        if (m_confidence == 0.0 && other.m_confidence == 0.0)
            return TruthValue();

        double total_conf = m_confidence + other.m_confidence;
        Strength merged_strength = (m_strength * m_confidence + other.m_strength * other.m_confidence) / total_conf;
        Confidence merged_conf = std::max(m_confidence, other.m_confidence);

        return TruthValue(merged_strength, merged_conf);
    }

    /**
     * Revision - combine evidence from two truth values.
     */
    TruthValue revision(const TruthValue& other, double k = DEFAULT_K) const
    {
        Count c1 = count(k);
        Count c2 = other.count(k);
        Count new_count = c1 + c2;

        if (new_count == 0.0)
            return TruthValue();

        Strength new_strength = (m_strength * c1 + other.m_strength * c2) / new_count;
        return from_count(new_strength, new_count, k);
    }

    /**
     * Negation - returns truth value for NOT this.
     */
    TruthValue negation() const
    {
        return TruthValue(1.0 - m_strength, m_confidence);
    }

    /**
     * Conjunction - AND operation with another truth value.
     */
    TruthValue conjunction(const TruthValue& other) const
    {
        Strength s = m_strength * other.m_strength;
        Confidence c = std::min(m_confidence, other.m_confidence);
        return TruthValue(s, c);
    }

    /**
     * Disjunction - OR operation with another truth value.
     */
    TruthValue disjunction(const TruthValue& other) const
    {
        Strength s = m_strength + other.m_strength - m_strength * other.m_strength;
        Confidence c = std::min(m_confidence, other.m_confidence);
        return TruthValue(s, c);
    }

    /**
     * String representation.
     */
    std::string to_string() const
    {
        std::ostringstream oss;
        oss << "(stv " << m_strength << " " << m_confidence << ")";
        return oss.str();
    }

    // Comparison operators
    bool operator==(const TruthValue& other) const
    {
        return std::abs(m_strength - other.m_strength) < 1e-9 &&
               std::abs(m_confidence - other.m_confidence) < 1e-9;
    }

    bool operator!=(const TruthValue& other) const
    {
        return !(*this == other);
    }

private:
    Strength m_strength;
    Confidence m_confidence;
};

/**
 * Convenience function to create a simple truth value.
 */
inline TruthValue stv(Strength s, Confidence c)
{
    return TruthValue(s, c);
}

/**
 * Create a "true" truth value.
 */
inline TruthValue tv_true(Confidence c = 0.9)
{
    return TruthValue(1.0, c);
}

/**
 * Create a "false" truth value.
 */
inline TruthValue tv_false(Confidence c = 0.9)
{
    return TruthValue(0.0, c);
}

} // namespace opencog
} // namespace gnc

#endif // GNC_OPENCOG_TRUTH_VALUE_HPP
