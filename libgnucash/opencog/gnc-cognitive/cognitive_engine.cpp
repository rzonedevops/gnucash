/*
 * opencog/gnc-cognitive/cognitive_engine.cpp
 *
 * GnuCash Cognitive Engine implementation
 *
 * Copyright (C) 2024 GnuCash Developers
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "cognitive_engine.hpp"
#include "../cogutil/logger.hpp"

#include <algorithm>
#include <sstream>
#include <regex>
#include <cmath>

namespace gnc {
namespace opencog {

namespace {
    // Singleton instance
    std::unique_ptr<CognitiveEngine> g_engine;
}

CognitiveEngine& cognitive_engine()
{
    if (!g_engine)
        g_engine = std::make_unique<CognitiveEngine>();
    return *g_engine;
}

CognitiveEngine::CognitiveEngine()
    : m_initialized(false)
{
}

CognitiveEngine::~CognitiveEngine()
{
    if (m_initialized)
        shutdown();
}

bool CognitiveEngine::initialize()
{
    if (m_initialized)
        return true;

    GNC_COG_INFO("CognitiveEngine", "Initializing GnuCash Cognitive Engine...");

    // Build initial knowledge base
    build_initial_knowledge();

    m_initialized = true;
    GNC_COG_INFO("CognitiveEngine", "Cognitive Engine initialized successfully");
    return true;
}

void CognitiveEngine::shutdown()
{
    if (!m_initialized)
        return;

    GNC_COG_INFO("CognitiveEngine", "Shutting down Cognitive Engine...");

    m_atomspace.clear();
    m_category_counts.clear();
    m_vendor_categories.clear();
    m_keyword_categories.clear();
    m_detected_patterns.clear();

    m_initialized = false;
}

void CognitiveEngine::build_initial_knowledge()
{
    // Create fundamental concept nodes for accounting
    m_atomspace.add_node(AtomTypes::CONCEPT_NODE, "Income");
    m_atomspace.add_node(AtomTypes::CONCEPT_NODE, "Expense");
    m_atomspace.add_node(AtomTypes::CONCEPT_NODE, "Asset");
    m_atomspace.add_node(AtomTypes::CONCEPT_NODE, "Liability");
    m_atomspace.add_node(AtomTypes::CONCEPT_NODE, "Equity");

    // Common expense categories
    const std::vector<std::string> expense_categories = {
        "Food", "Groceries", "Restaurants", "Transportation", "Gas",
        "Utilities", "Rent", "Mortgage", "Insurance", "Healthcare",
        "Entertainment", "Shopping", "Clothing", "Education", "Subscriptions",
        "Travel", "Personal Care", "Gifts", "Charity", "Taxes"
    };

    auto expense_concept = m_atomspace.get_node(AtomTypes::CONCEPT_NODE, "Expense");
    for (const auto& cat : expense_categories) {
        auto cat_node = m_atomspace.add_node(AtomTypes::CATEGORY_NODE, cat);
        m_atomspace.add_link(AtomTypes::INHERITANCE_LINK, cat_node, expense_concept);
    }

    // Common income categories
    const std::vector<std::string> income_categories = {
        "Salary", "Wages", "Freelance", "Investment", "Interest",
        "Dividends", "Rental Income", "Business Income", "Bonus", "Commission"
    };

    auto income_concept = m_atomspace.get_node(AtomTypes::CONCEPT_NODE, "Income");
    for (const auto& cat : income_categories) {
        auto cat_node = m_atomspace.add_node(AtomTypes::CATEGORY_NODE, cat);
        m_atomspace.add_link(AtomTypes::INHERITANCE_LINK, cat_node, income_concept);
    }

    // Build keyword-to-category mappings
    m_keyword_categories["grocery"] = Counter<std::string>{{"Groceries"}};
    m_keyword_categories["supermarket"] = Counter<std::string>{{"Groceries"}};
    m_keyword_categories["walmart"] = Counter<std::string>{{"Groceries", "Shopping"}};
    m_keyword_categories["amazon"] = Counter<std::string>{{"Shopping"}};
    m_keyword_categories["restaurant"] = Counter<std::string>{{"Restaurants"}};
    m_keyword_categories["cafe"] = Counter<std::string>{{"Restaurants"}};
    m_keyword_categories["coffee"] = Counter<std::string>{{"Restaurants"}};
    m_keyword_categories["uber"] = Counter<std::string>{{"Transportation"}};
    m_keyword_categories["lyft"] = Counter<std::string>{{"Transportation"}};
    m_keyword_categories["gas"] = Counter<std::string>{{"Gas"}};
    m_keyword_categories["shell"] = Counter<std::string>{{"Gas"}};
    m_keyword_categories["chevron"] = Counter<std::string>{{"Gas"}};
    m_keyword_categories["electric"] = Counter<std::string>{{"Utilities"}};
    m_keyword_categories["water"] = Counter<std::string>{{"Utilities"}};
    m_keyword_categories["netflix"] = Counter<std::string>{{"Subscriptions"}};
    m_keyword_categories["spotify"] = Counter<std::string>{{"Subscriptions"}};
    m_keyword_categories["pharmacy"] = Counter<std::string>{{"Healthcare"}};
    m_keyword_categories["doctor"] = Counter<std::string>{{"Healthcare"}};
    m_keyword_categories["hospital"] = Counter<std::string>{{"Healthcare"}};
}

void CognitiveEngine::import_account(const std::string& guid, const std::string& name,
                                     const std::string& account_type, const std::string& parent_guid)
{
    auto account_node = m_atomspace.add_node(AtomTypes::ACCOUNT_NODE, guid);

    // Add name as a property
    auto name_node = m_atomspace.add_node(AtomTypes::CONCEPT_NODE, name);
    m_atomspace.add_link(AtomTypes::EVALUATION_LINK,
        m_atomspace.add_node(AtomTypes::PREDICATE_NODE, "has-name"),
        m_atomspace.add_link(AtomTypes::LIST_LINK, account_node, name_node));

    // Add account type
    auto type_node = m_atomspace.add_node(AtomTypes::TYPE_NODE, account_type);
    m_atomspace.add_link(AtomTypes::INHERITANCE_LINK, account_node, type_node);

    // Add parent relationship if exists
    if (!parent_guid.empty()) {
        auto parent_node = m_atomspace.add_node(AtomTypes::ACCOUNT_NODE, parent_guid);
        m_atomspace.add_link(AtomTypes::ACCOUNT_HIERARCHY_LINK, account_node, parent_node);
    }
}

void CognitiveEngine::import_transaction(const std::string& guid, const std::string& description,
                                         const std::string& date, double amount,
                                         const std::string& debit_account, const std::string& credit_account)
{
    auto txn_node = m_atomspace.add_node(AtomTypes::TRANSACTION_NODE, guid);

    // Description
    auto desc_node = m_atomspace.add_node(AtomTypes::CONCEPT_NODE, description);
    m_atomspace.add_link(AtomTypes::EVALUATION_LINK,
        m_atomspace.add_node(AtomTypes::PREDICATE_NODE, "has-description"),
        m_atomspace.add_link(AtomTypes::LIST_LINK, txn_node, desc_node));

    // Date
    auto date_node = m_atomspace.add_node(AtomTypes::DATE_NODE, date);
    m_atomspace.add_link(AtomTypes::TEMPORAL_LINK, txn_node, date_node);

    // Amount
    auto amount_node = m_atomspace.add_node(AtomTypes::AMOUNT_NODE, std::to_string(amount));
    m_atomspace.add_link(AtomTypes::EVALUATION_LINK,
        m_atomspace.add_node(AtomTypes::PREDICATE_NODE, "has-amount"),
        m_atomspace.add_link(AtomTypes::LIST_LINK, txn_node, amount_node));

    // Account links
    if (!debit_account.empty()) {
        auto debit_node = m_atomspace.add_node(AtomTypes::ACCOUNT_NODE, debit_account);
        m_atomspace.add_link(AtomTypes::FLOW_LINK,
            m_atomspace.add_node(AtomTypes::PREDICATE_NODE, "debit"),
            m_atomspace.add_link(AtomTypes::LIST_LINK, txn_node, debit_node));
    }

    if (!credit_account.empty()) {
        auto credit_node = m_atomspace.add_node(AtomTypes::ACCOUNT_NODE, credit_account);
        m_atomspace.add_link(AtomTypes::FLOW_LINK,
            m_atomspace.add_node(AtomTypes::PREDICATE_NODE, "credit"),
            m_atomspace.add_link(AtomTypes::LIST_LINK, txn_node, credit_node));
    }
}

void CognitiveEngine::import_vendor(const std::string& name, const std::string& category)
{
    auto vendor_node = m_atomspace.add_node(AtomTypes::VENDOR_NODE, name);

    if (!category.empty()) {
        auto cat_node = m_atomspace.add_node(AtomTypes::CATEGORY_NODE, category);
        m_atomspace.add_link(AtomTypes::CATEGORIZATION_LINK, vendor_node, cat_node);
        m_vendor_categories[name].increment(category);
    }
}

CategorizationResult CognitiveEngine::categorize_transaction(const std::string& description,
                                                             double amount,
                                                             const std::string& vendor)
{
    CategorizationResult result;
    result.confidence = 0.0;

    Counter<std::string> category_scores;

    // Check vendor-based categorization
    if (!vendor.empty()) {
        auto it = m_vendor_categories.find(vendor);
        if (it != m_vendor_categories.end()) {
            for (const auto& [cat, count] : it->second) {
                category_scores.increment(cat, count * 10);  // High weight for vendor match
            }
        }
    }

    // Extract keywords and check keyword-based categorization
    std::vector<std::string> keywords;
    extract_keywords(description, keywords);

    for (const auto& keyword : keywords) {
        auto it = m_keyword_categories.find(keyword);
        if (it != m_keyword_categories.end()) {
            for (const auto& [cat, count] : it->second) {
                category_scores.increment(cat, count * 5);
            }
        }
    }

    // Get most likely categories
    auto ranked = category_scores.most_common(5);

    if (!ranked.empty()) {
        result.category = ranked[0].first;

        // Calculate confidence based on score distribution
        double total_score = category_scores.total();
        if (total_score > 0) {
            result.confidence = static_cast<double>(ranked[0].second) / total_score;
        }

        // Add alternatives
        for (size_t i = 1; i < ranked.size(); ++i) {
            result.alternative_categories.push_back(ranked[i].first);
        }

        result.reasoning = "Based on ";
        if (!vendor.empty() && m_vendor_categories.contains(vendor)) {
            result.reasoning += "vendor history";
        } else if (!keywords.empty()) {
            result.reasoning += "keyword matching";
        }
    } else {
        result.category = "Uncategorized";
        result.confidence = 0.0;
        result.reasoning = "No matching patterns found";
    }

    // Consider amount for certain categories
    if (amount > 1000 && result.confidence < 0.5) {
        result.alternative_categories.insert(result.alternative_categories.begin(), "Large Purchase");
    }

    return result;
}

void CognitiveEngine::learn_categorization(const std::string& description, const std::string& vendor,
                                           const std::string& chosen_category)
{
    // Update vendor mapping
    if (!vendor.empty()) {
        m_vendor_categories[vendor].increment(chosen_category);
    }

    // Update keyword mappings
    std::vector<std::string> keywords;
    extract_keywords(description, keywords);
    for (const auto& keyword : keywords) {
        m_keyword_categories[keyword].increment(chosen_category);
    }

    // Update global category counts
    m_category_counts.increment(chosen_category);

    // Create learning atom in AtomSpace
    auto cat_node = m_atomspace.add_node(AtomTypes::CATEGORY_NODE, chosen_category);
    auto desc_node = m_atomspace.add_node(AtomTypes::CONCEPT_NODE, description);

    // Strengthen the categorization link
    auto existing = m_atomspace.get_link(AtomTypes::CATEGORIZATION_LINK, {desc_node, cat_node});
    if (existing) {
        TruthValue tv = existing->truth_value();
        existing->set_truth_value(tv.revision(stv(1.0, 0.9)));
    } else {
        auto link = m_atomspace.add_link(AtomTypes::CATEGORIZATION_LINK, desc_node, cat_node);
        link->set_truth_value(stv(1.0, 0.5));
    }
}

std::vector<SpendingPattern> CognitiveEngine::detect_spending_patterns()
{
    std::vector<SpendingPattern> patterns;

    // Find all transactions
    auto transactions = m_atomspace.get_atoms_by_type(AtomTypes::TRANSACTION_NODE);

    // Group by vendor/description
    Counter<std::string> vendor_counts;
    for (const auto& txn : transactions) {
        // Get description
        auto incoming = m_atomspace.get_incoming_by_type(txn, AtomTypes::EVALUATION_LINK);
        for (const auto& link : incoming) {
            auto pred = link->outgoing_atom(0);
            if (pred && pred->name() == "has-description") {
                auto list = link->outgoing_atom(1);
                if (list && list->arity() >= 2) {
                    auto desc = list->outgoing_atom(1);
                    if (desc)
                        vendor_counts.increment(desc->name());
                }
            }
        }
    }

    // Identify recurring patterns
    for (const auto& [desc, count] : vendor_counts) {
        if (count >= 3) {  // At least 3 occurrences
            SpendingPattern pattern;
            pattern.name = desc;
            pattern.description = "Recurring transaction: " + desc;
            pattern.frequency = (count >= 12) ? "monthly" : (count >= 52) ? "weekly" : "periodic";
            pattern.confidence = std::min(1.0, count / 10.0);
            patterns.push_back(pattern);
        }
    }

    m_detected_patterns = patterns;
    return patterns;
}

std::vector<Handle> CognitiveEngine::find_recurring_transactions()
{
    PatternMatcher matcher(m_atomspace);
    return matcher.find_by_type(AtomTypes::TRANSACTION_NODE);
}

std::vector<Handle> CognitiveEngine::detect_anomalies(double threshold)
{
    std::vector<Handle> anomalies;

    auto transactions = m_atomspace.get_atoms_by_type(AtomTypes::TRANSACTION_NODE);

    // Calculate statistics for anomaly detection
    std::vector<double> amounts;
    for (const auto& txn : transactions) {
        auto incoming = m_atomspace.get_incoming_by_type(txn, AtomTypes::EVALUATION_LINK);
        for (const auto& link : incoming) {
            auto pred = link->outgoing_atom(0);
            if (pred && pred->name() == "has-amount") {
                auto list = link->outgoing_atom(1);
                if (list && list->arity() >= 2) {
                    auto amt = list->outgoing_atom(1);
                    if (amt) {
                        try {
                            amounts.push_back(std::stod(amt->name()));
                        } catch (...) {}
                    }
                }
            }
        }
    }

    if (amounts.size() < 2)
        return anomalies;

    // Calculate mean and standard deviation
    double sum = 0, sq_sum = 0;
    for (double amt : amounts) {
        sum += amt;
        sq_sum += amt * amt;
    }
    double mean = sum / amounts.size();
    double variance = sq_sum / amounts.size() - mean * mean;
    double stddev = std::sqrt(variance);

    // Find anomalies (transactions outside threshold * stddev)
    size_t idx = 0;
    for (const auto& txn : transactions) {
        if (idx < amounts.size()) {
            double z_score = std::abs(amounts[idx] - mean) / stddev;
            if (z_score > threshold) {
                anomalies.push_back(txn);

                // Mark as anomaly in AtomSpace
                m_atomspace.add_link(AtomTypes::ANOMALY_LINK, txn,
                    m_atomspace.add_node(AtomTypes::NUMBER_NODE, std::to_string(z_score)));
            }
        }
        ++idx;
    }

    return anomalies;
}

double CognitiveEngine::predict_cash_flow(int days_ahead)
{
    // Simple prediction based on recent patterns
    auto patterns = detect_spending_patterns();

    double predicted_expenses = 0.0;
    double predicted_income = 0.0;

    for (const auto& pattern : patterns) {
        // Estimate based on frequency
        double daily_rate = pattern.average_amount;
        if (pattern.frequency == "monthly")
            daily_rate /= 30.0;
        else if (pattern.frequency == "weekly")
            daily_rate /= 7.0;

        predicted_expenses += daily_rate * days_ahead;
    }

    return predicted_income - predicted_expenses;
}

std::vector<std::pair<std::string, double>> CognitiveEngine::predict_expenses(int days_ahead)
{
    std::vector<std::pair<std::string, double>> predictions;

    auto patterns = detect_spending_patterns();
    for (const auto& pattern : patterns) {
        double daily_rate = pattern.average_amount;
        if (pattern.frequency == "monthly")
            daily_rate /= 30.0;
        else if (pattern.frequency == "weekly")
            daily_rate /= 7.0;

        predictions.emplace_back(pattern.name, daily_rate * days_ahead);
    }

    return predictions;
}

std::vector<FinancialInsight> CognitiveEngine::generate_insights()
{
    std::vector<FinancialInsight> insights;

    // Detect patterns
    auto patterns = detect_spending_patterns();
    for (const auto& pattern : patterns) {
        if (pattern.confidence > 0.7) {
            FinancialInsight insight;
            insight.type = InsightType::SPENDING_PATTERN;
            insight.title = "Recurring: " + pattern.name;
            insight.description = pattern.description;
            insight.confidence = pattern.confidence;
            insight.timestamp = std::chrono::system_clock::now();
            insights.push_back(insight);
        }
    }

    // Detect anomalies
    auto anomalies = detect_anomalies();
    for (const auto& anomaly : anomalies) {
        FinancialInsight insight;
        insight.type = InsightType::ANOMALY;
        insight.title = "Unusual Transaction Detected";
        insight.description = "Transaction " + anomaly->name() + " is significantly different from typical spending";
        insight.confidence = 0.8;
        insight.timestamp = std::chrono::system_clock::now();
        insights.push_back(insight);
    }

    // Notify subscribers
    for (const auto& insight : insights)
        notify_insight(insight);

    return insights;
}

void CognitiveEngine::subscribe_insights(InsightCallback callback)
{
    m_insight_callbacks.push_back(std::move(callback));
}

std::vector<std::string> CognitiveEngine::get_budget_recommendations()
{
    std::vector<std::string> recommendations;

    auto patterns = detect_spending_patterns();

    // Analyze spending and provide recommendations
    double total_subscription = 0;
    for (const auto& pattern : patterns) {
        if (pattern.frequency == "monthly" && pattern.average_amount > 0) {
            total_subscription += pattern.average_amount;
        }
    }

    if (total_subscription > 200) {
        recommendations.push_back("Review your monthly subscriptions - you're spending $" +
                                 std::to_string(static_cast<int>(total_subscription)) + "/month");
    }

    return recommendations;
}

std::string CognitiveEngine::query(const std::string& question)
{
    // Simple natural language query processing
    std::string lower_question = question;
    std::transform(lower_question.begin(), lower_question.end(), lower_question.begin(), ::tolower);

    if (lower_question.find("how much") != std::string::npos &&
        lower_question.find("spend") != std::string::npos) {
        auto patterns = detect_spending_patterns();
        double total = 0;
        for (const auto& p : patterns)
            total += p.average_amount;
        return "Based on detected patterns, average spending is approximately $" +
               std::to_string(static_cast<int>(total));
    }

    if (lower_question.find("pattern") != std::string::npos ||
        lower_question.find("recurring") != std::string::npos) {
        auto patterns = detect_spending_patterns();
        if (patterns.empty())
            return "No recurring patterns detected yet.";

        std::string response = "Detected " + std::to_string(patterns.size()) + " patterns:\n";
        for (const auto& p : patterns) {
            response += "- " + p.name + " (" + p.frequency + ")\n";
        }
        return response;
    }

    if (lower_question.find("anomal") != std::string::npos ||
        lower_question.find("unusual") != std::string::npos) {
        auto anomalies = detect_anomalies();
        return "Found " + std::to_string(anomalies.size()) + " unusual transactions.";
    }

    return "I can help you analyze spending patterns, detect anomalies, and categorize transactions. "
           "Try asking about 'spending patterns' or 'unusual transactions'.";
}

CognitiveEngine::Stats CognitiveEngine::get_stats() const
{
    Stats stats;
    auto as_stats = m_atomspace.get_stats();

    stats.total_atoms = as_stats.total_atoms;
    stats.accounts_known = m_atomspace.size(AtomTypes::ACCOUNT_NODE);
    stats.transactions_known = m_atomspace.size(AtomTypes::TRANSACTION_NODE);
    stats.patterns_detected = m_detected_patterns.size();

    size_t rules = 0;
    for (const auto& [_, counter] : m_vendor_categories)
        rules += counter.size();
    for (const auto& [_, counter] : m_keyword_categories)
        rules += counter.size();
    stats.categorization_rules = rules;

    return stats;
}

void CognitiveEngine::extract_keywords(const std::string& text, std::vector<std::string>& keywords)
{
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    // Simple word extraction
    std::istringstream iss(lower);
    std::string word;
    while (iss >> word) {
        // Remove punctuation
        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
        if (word.length() >= 3)  // Skip short words
            keywords.push_back(word);
    }
}

double CognitiveEngine::calculate_pattern_strength(const HandleSeq& transactions)
{
    if (transactions.empty())
        return 0.0;
    return std::min(1.0, transactions.size() / 5.0);
}

void CognitiveEngine::notify_insight(const FinancialInsight& insight)
{
    for (const auto& callback : m_insight_callbacks)
        callback(insight);
}

} // namespace opencog
} // namespace gnc
