/*
 * opencog/cogutil/concurrent_queue.hpp
 *
 * Thread-safe concurrent queue implementation
 * Based on OpenCog cogutil patterns
 *
 * Copyright (C) 2024 GnuCash Developers
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef GNC_OPENCOG_CONCURRENT_QUEUE_HPP
#define GNC_OPENCOG_CONCURRENT_QUEUE_HPP

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <chrono>

namespace gnc {
namespace opencog {

/**
 * A thread-safe concurrent queue.
 * Multiple threads can push and pop from this queue safely.
 */
template<typename T>
class ConcurrentQueue
{
public:
    ConcurrentQueue() = default;
    ~ConcurrentQueue() = default;

    // Non-copyable
    ConcurrentQueue(const ConcurrentQueue&) = delete;
    ConcurrentQueue& operator=(const ConcurrentQueue&) = delete;

    // Moveable
    ConcurrentQueue(ConcurrentQueue&&) = default;
    ConcurrentQueue& operator=(ConcurrentQueue&&) = default;

    /**
     * Push an item onto the queue.
     * Thread-safe.
     */
    void push(const T& item)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push(item);
        m_condition.notify_one();
    }

    /**
     * Push an item onto the queue (move version).
     * Thread-safe.
     */
    void push(T&& item)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_queue.push(std::move(item));
        m_condition.notify_one();
    }

    /**
     * Pop an item from the queue, blocking if empty.
     * Thread-safe.
     */
    T pop()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_condition.wait(lock, [this] { return !m_queue.empty() || m_cancelled; });

        if (m_cancelled && m_queue.empty())
            throw std::runtime_error("Queue cancelled");

        T item = std::move(m_queue.front());
        m_queue.pop();
        return item;
    }

    /**
     * Try to pop an item from the queue without blocking.
     * Returns std::nullopt if queue is empty.
     */
    std::optional<T> try_pop()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_queue.empty())
            return std::nullopt;

        T item = std::move(m_queue.front());
        m_queue.pop();
        return item;
    }

    /**
     * Try to pop with timeout.
     * Returns std::nullopt if timeout expires.
     */
    template<typename Rep, typename Period>
    std::optional<T> try_pop_for(const std::chrono::duration<Rep, Period>& timeout)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (!m_condition.wait_for(lock, timeout, [this] { return !m_queue.empty() || m_cancelled; }))
            return std::nullopt;

        if (m_cancelled && m_queue.empty())
            return std::nullopt;

        T item = std::move(m_queue.front());
        m_queue.pop();
        return item;
    }

    /**
     * Check if queue is empty.
     */
    bool empty() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_queue.empty();
    }

    /**
     * Get queue size.
     */
    size_t size() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_queue.size();
    }

    /**
     * Cancel all waiting operations.
     */
    void cancel()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_cancelled = true;
        m_condition.notify_all();
    }

    /**
     * Clear the queue.
     */
    void clear()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::queue<T> empty;
        std::swap(m_queue, empty);
    }

private:
    mutable std::mutex m_mutex;
    std::condition_variable m_condition;
    std::queue<T> m_queue;
    bool m_cancelled = false;
};

} // namespace opencog
} // namespace gnc

#endif // GNC_OPENCOG_CONCURRENT_QUEUE_HPP
