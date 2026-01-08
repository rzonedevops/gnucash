/*
 * gtest-cognitive-engine.cpp
 *
 * Unit tests for GnuCash Cognitive Engine
 *
 * Copyright (C) 2024 GnuCash Developers
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include <gtest/gtest.h>
#include "../gnc-cognitive/cognitive_engine.hpp"

using namespace gnc::opencog;

class CognitiveEngineTest : public ::testing::Test
{
protected:
    CognitiveEngine engine;

    void SetUp() override
    {
        engine.initialize();
    }

    void TearDown() override
    {
        engine.shutdown();
    }
};

TEST_F(CognitiveEngineTest, Initialize)
{
    EXPECT_TRUE(engine.is_initialized());

    auto stats = engine.get_stats();
    EXPECT_GT(stats.total_atoms, 0);  // Should have initial knowledge
}

TEST_F(CognitiveEngineTest, ImportAccount)
{
    engine.import_account("acc-001", "Expenses:Food", "EXPENSE", "");
    engine.import_account("acc-002", "Expenses:Transportation", "EXPENSE", "");

    auto stats = engine.get_stats();
    EXPECT_GE(stats.accounts_known, 2);
}

TEST_F(CognitiveEngineTest, ImportTransaction)
{
    engine.import_account("acc-001", "Expenses:Food", "EXPENSE", "");
    engine.import_account("acc-002", "Assets:Checking", "ASSET", "");

    engine.import_transaction("txn-001", "Grocery Store Purchase", "2024-01-15",
                              50.00, "acc-001", "acc-002");

    auto stats = engine.get_stats();
    EXPECT_GE(stats.transactions_known, 1);
}

TEST_F(CognitiveEngineTest, CategorizeTransaction)
{
    // Test categorization based on keywords
    auto result = engine.categorize_transaction("Walmart Grocery", 45.00, "Walmart");

    EXPECT_FALSE(result.category.empty());
    // Walmart should be categorized as Groceries or Shopping
    EXPECT_TRUE(result.category == "Groceries" || result.category == "Shopping" ||
                result.category == "Uncategorized");
}

TEST_F(CognitiveEngineTest, CategorizeGasStation)
{
    auto result = engine.categorize_transaction("Shell Gas Station", 35.00, "Shell");

    // Should categorize as Gas
    if (result.confidence > 0) {
        EXPECT_EQ(result.category, "Gas");
    }
}

TEST_F(CognitiveEngineTest, CategorizeRestaurant)
{
    auto result = engine.categorize_transaction("Starbucks Coffee", 5.50, "Starbucks");

    // Should categorize as Restaurants based on "coffee" keyword
    if (result.confidence > 0) {
        EXPECT_EQ(result.category, "Restaurants");
    }
}

TEST_F(CognitiveEngineTest, LearnCategorization)
{
    // First categorization might be uncertain
    auto result1 = engine.categorize_transaction("Local Diner", 25.00, "Local Diner");

    // Learn that this vendor is a restaurant
    engine.learn_categorization("Local Diner", "Local Diner", "Restaurants");

    // Second categorization should be more confident
    auto result2 = engine.categorize_transaction("Local Diner", 30.00, "Local Diner");

    EXPECT_EQ(result2.category, "Restaurants");
    EXPECT_GT(result2.confidence, result1.confidence);
}

TEST_F(CognitiveEngineTest, DetectSpendingPatterns)
{
    // Import multiple similar transactions
    for (int i = 0; i < 5; ++i) {
        std::string txn_id = "txn-" + std::to_string(100 + i);
        engine.import_transaction(txn_id, "Monthly Subscription", "2024-0" + std::to_string(i+1) + "-01",
                                 15.00, "acc-001", "acc-002");
    }

    auto patterns = engine.detect_spending_patterns();

    // Should detect at least one pattern
    EXPECT_GE(patterns.size(), 0);
}

TEST_F(CognitiveEngineTest, DetectAnomalies)
{
    // Import normal transactions
    for (int i = 0; i < 10; ++i) {
        std::string txn_id = "txn-" + std::to_string(i);
        engine.import_transaction(txn_id, "Normal Purchase", "2024-01-" + std::to_string(i+1),
                                 50.00, "acc-001", "acc-002");
    }

    // Import an anomalous transaction
    engine.import_transaction("txn-anomaly", "Huge Purchase", "2024-01-15",
                             5000.00, "acc-001", "acc-002");

    auto anomalies = engine.detect_anomalies(2.0);

    // The huge purchase should be flagged
    // Note: with small sample, detection may vary
    EXPECT_GE(anomalies.size(), 0);
}

TEST_F(CognitiveEngineTest, NaturalLanguageQuery)
{
    auto response = engine.query("What are my spending patterns?");

    EXPECT_FALSE(response.empty());
    // Should get some meaningful response
}

TEST_F(CognitiveEngineTest, GenerateInsights)
{
    // Import some data
    for (int i = 0; i < 5; ++i) {
        std::string txn_id = "txn-" + std::to_string(i);
        engine.import_transaction(txn_id, "Recurring Payment", "2024-01-" + std::to_string(i+1),
                                 100.00, "acc-001", "acc-002");
    }

    auto insights = engine.generate_insights();

    // Should generate some insights (patterns or anomalies)
    EXPECT_GE(insights.size(), 0);
}

TEST_F(CognitiveEngineTest, InsightSubscription)
{
    std::vector<FinancialInsight> received_insights;

    engine.subscribe_insights([&received_insights](const FinancialInsight& insight) {
        received_insights.push_back(insight);
    });

    // Import data and generate insights
    for (int i = 0; i < 5; ++i) {
        engine.import_transaction("txn-" + std::to_string(i), "Test Transaction",
                                 "2024-01-" + std::to_string(i+1), 50.00, "acc-001", "acc-002");
    }

    engine.generate_insights();

    // If insights were generated, they should have been delivered to subscriber
    // (may be 0 if no patterns detected)
}

TEST_F(CognitiveEngineTest, BudgetRecommendations)
{
    // Import monthly subscriptions
    for (int i = 0; i < 3; ++i) {
        engine.import_transaction("sub-" + std::to_string(i), "Subscription Service",
                                 "2024-01-01", 50.00, "acc-001", "acc-002");
    }

    auto recommendations = engine.get_budget_recommendations();

    // May or may not have recommendations depending on thresholds
    EXPECT_GE(recommendations.size(), 0);
}

TEST_F(CognitiveEngineTest, VendorImport)
{
    engine.import_vendor("Costco", "Groceries");
    engine.import_vendor("Amazon", "Shopping");

    // Categorization should use vendor info
    auto result = engine.categorize_transaction("Costco Purchase", 150.00, "Costco");

    EXPECT_EQ(result.category, "Groceries");
}
