/*
 * opencog/cogutil/counter.hpp
 *
 * Counter class similar to Python's collections.Counter
 * Based on OpenCog cogutil patterns
 *
 * Copyright (C) 2024 GnuCash Developers
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef GNC_OPENCOG_COUNTER_HPP
#define GNC_OPENCOG_COUNTER_HPP

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <functional>

namespace gnc {
namespace opencog {

/**
 * A counter class that counts hashable objects.
 * Similar to Python's collections.Counter.
 */
template<typename Key, typename Count = size_t,
         typename Hash = std::hash<Key>,
         typename Equal = std::equal_to<Key>>
class Counter
{
public:
    using map_type = std::unordered_map<Key, Count, Hash, Equal>;
    using iterator = typename map_type::iterator;
    using const_iterator = typename map_type::const_iterator;
    using value_type = std::pair<Key, Count>;

    Counter() = default;

    /**
     * Construct from initializer list of keys.
     */
    Counter(std::initializer_list<Key> keys)
    {
        for (const auto& key : keys)
            increment(key);
    }

    /**
     * Construct from range of keys.
     */
    template<typename InputIt>
    Counter(InputIt first, InputIt last)
    {
        for (; first != last; ++first)
            increment(*first);
    }

    /**
     * Increment count for key by delta.
     */
    void increment(const Key& key, Count delta = 1)
    {
        m_counts[key] += delta;
    }

    /**
     * Decrement count for key by delta.
     */
    void decrement(const Key& key, Count delta = 1)
    {
        auto it = m_counts.find(key);
        if (it != m_counts.end()) {
            if (it->second <= delta)
                m_counts.erase(it);
            else
                it->second -= delta;
        }
    }

    /**
     * Get count for key (returns 0 if not present).
     */
    Count count(const Key& key) const
    {
        auto it = m_counts.find(key);
        return (it != m_counts.end()) ? it->second : Count{0};
    }

    /**
     * Access count for key (creates entry if not present).
     */
    Count& operator[](const Key& key)
    {
        return m_counts[key];
    }

    /**
     * Check if key exists with non-zero count.
     */
    bool contains(const Key& key) const
    {
        auto it = m_counts.find(key);
        return it != m_counts.end() && it->second > 0;
    }

    /**
     * Get total count of all elements.
     */
    Count total() const
    {
        Count sum = 0;
        for (const auto& [key, cnt] : m_counts)
            sum += cnt;
        return sum;
    }

    /**
     * Get number of unique elements.
     */
    size_t size() const { return m_counts.size(); }

    /**
     * Check if counter is empty.
     */
    bool empty() const { return m_counts.empty(); }

    /**
     * Clear all counts.
     */
    void clear() { m_counts.clear(); }

    /**
     * Get the n most common elements.
     */
    std::vector<value_type> most_common(size_t n = 0) const
    {
        std::vector<value_type> result(m_counts.begin(), m_counts.end());
        std::sort(result.begin(), result.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        if (n > 0 && n < result.size())
            result.resize(n);

        return result;
    }

    /**
     * Get the n least common elements.
     */
    std::vector<value_type> least_common(size_t n = 0) const
    {
        std::vector<value_type> result(m_counts.begin(), m_counts.end());
        std::sort(result.begin(), result.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        if (n > 0 && n < result.size())
            result.resize(n);

        return result;
    }

    /**
     * Add counts from another counter.
     */
    Counter& operator+=(const Counter& other)
    {
        for (const auto& [key, cnt] : other.m_counts)
            m_counts[key] += cnt;
        return *this;
    }

    /**
     * Subtract counts from another counter.
     */
    Counter& operator-=(const Counter& other)
    {
        for (const auto& [key, cnt] : other.m_counts) {
            auto it = m_counts.find(key);
            if (it != m_counts.end()) {
                if (it->second <= cnt)
                    m_counts.erase(it);
                else
                    it->second -= cnt;
            }
        }
        return *this;
    }

    // Iterators
    iterator begin() { return m_counts.begin(); }
    iterator end() { return m_counts.end(); }
    const_iterator begin() const { return m_counts.begin(); }
    const_iterator end() const { return m_counts.end(); }
    const_iterator cbegin() const { return m_counts.cbegin(); }
    const_iterator cend() const { return m_counts.cend(); }

private:
    map_type m_counts;
};

} // namespace opencog
} // namespace gnc

#endif // GNC_OPENCOG_COUNTER_HPP
