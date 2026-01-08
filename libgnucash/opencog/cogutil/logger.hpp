/*
 * opencog/cogutil/logger.hpp
 *
 * Logging utilities for OpenCog subsystem
 * Based on OpenCog cogutil patterns
 *
 * Copyright (C) 2024 GnuCash Developers
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef GNC_OPENCOG_LOGGER_HPP
#define GNC_OPENCOG_LOGGER_HPP

#include <string>
#include <sstream>
#include <functional>
#include <mutex>

namespace gnc {
namespace opencog {

/**
 * Log levels for the cognitive subsystem.
 */
enum class LogLevel
{
    NONE = 0,
    ERROR = 1,
    WARN = 2,
    INFO = 3,
    DEBUG = 4,
    FINE = 5
};

/**
 * Logger for the OpenCog cognitive subsystem.
 * Integrates with GnuCash's qof logging system.
 */
class Logger
{
public:
    using LogCallback = std::function<void(LogLevel, const std::string&, const std::string&)>;

    static Logger& instance()
    {
        static Logger logger;
        return logger;
    }

    void set_level(LogLevel level) { m_level = level; }
    LogLevel get_level() const { return m_level; }

    void set_callback(LogCallback callback)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_callback = std::move(callback);
    }

    void log(LogLevel level, const std::string& component, const std::string& message)
    {
        if (level > m_level)
            return;

        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_callback)
            m_callback(level, component, message);
    }

    template<typename... Args>
    void error(const std::string& component, Args&&... args)
    {
        log(LogLevel::ERROR, component, format(std::forward<Args>(args)...));
    }

    template<typename... Args>
    void warn(const std::string& component, Args&&... args)
    {
        log(LogLevel::WARN, component, format(std::forward<Args>(args)...));
    }

    template<typename... Args>
    void info(const std::string& component, Args&&... args)
    {
        log(LogLevel::INFO, component, format(std::forward<Args>(args)...));
    }

    template<typename... Args>
    void debug(const std::string& component, Args&&... args)
    {
        log(LogLevel::DEBUG, component, format(std::forward<Args>(args)...));
    }

    template<typename... Args>
    void fine(const std::string& component, Args&&... args)
    {
        log(LogLevel::FINE, component, format(std::forward<Args>(args)...));
    }

private:
    Logger() : m_level(LogLevel::INFO) {}

    template<typename... Args>
    std::string format(Args&&... args)
    {
        std::ostringstream oss;
        (oss << ... << std::forward<Args>(args));
        return oss.str();
    }

    LogLevel m_level;
    LogCallback m_callback;
    std::mutex m_mutex;
};

// Convenience macros
#define GNC_COG_LOG(level, component, ...) \
    gnc::opencog::Logger::instance().log(level, component, __VA_ARGS__)

#define GNC_COG_ERROR(component, ...) \
    gnc::opencog::Logger::instance().error(component, __VA_ARGS__)

#define GNC_COG_WARN(component, ...) \
    gnc::opencog::Logger::instance().warn(component, __VA_ARGS__)

#define GNC_COG_INFO(component, ...) \
    gnc::opencog::Logger::instance().info(component, __VA_ARGS__)

#define GNC_COG_DEBUG(component, ...) \
    gnc::opencog::Logger::instance().debug(component, __VA_ARGS__)

} // namespace opencog
} // namespace gnc

#endif // GNC_OPENCOG_LOGGER_HPP
