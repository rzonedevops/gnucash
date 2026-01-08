/*
 * gtest-pattern-match.cpp
 *
 * Unit tests for pattern matching engine
 *
 * Copyright (C) 2024 GnuCash Developers
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include <gtest/gtest.h>
#include "../atomspace/atomspace.hpp"
#include "../pattern/pattern_match.hpp"

using namespace gnc::opencog;

class PatternMatchTest : public ::testing::Test
{
protected:
    AtomSpace atomspace;

    void SetUp() override
    {
        // Create some test data
        auto animal = atomspace.add_node(AtomTypes::CONCEPT_NODE, "Animal");
        auto dog = atomspace.add_node(AtomTypes::CONCEPT_NODE, "Dog");
        auto cat = atomspace.add_node(AtomTypes::CONCEPT_NODE, "Cat");
        auto bird = atomspace.add_node(AtomTypes::CONCEPT_NODE, "Bird");

        atomspace.add_link(AtomTypes::INHERITANCE_LINK, dog, animal);
        atomspace.add_link(AtomTypes::INHERITANCE_LINK, cat, animal);
        atomspace.add_link(AtomTypes::INHERITANCE_LINK, bird, animal);

        // GnuCash-specific data
        auto expenses = atomspace.add_node(AtomTypes::ACCOUNT_NODE, "Expenses");
        auto food = atomspace.add_node(AtomTypes::ACCOUNT_NODE, "Expenses:Food");
        auto transport = atomspace.add_node(AtomTypes::ACCOUNT_NODE, "Expenses:Transportation");

        atomspace.add_link(AtomTypes::ACCOUNT_HIERARCHY_LINK, food, expenses);
        atomspace.add_link(AtomTypes::ACCOUNT_HIERARCHY_LINK, transport, expenses);
    }
};

TEST_F(PatternMatchTest, FindByType)
{
    PatternMatcher matcher(atomspace);

    auto concepts = matcher.find_by_type(AtomTypes::CONCEPT_NODE);
    EXPECT_EQ(concepts.size(), 4);  // Animal, Dog, Cat, Bird

    auto accounts = matcher.find_by_type(AtomTypes::ACCOUNT_NODE);
    EXPECT_EQ(accounts.size(), 3);  // Expenses, Expenses:Food, Expenses:Transportation
}

TEST_F(PatternMatchTest, FindNodes)
{
    PatternMatcher matcher(atomspace);

    // Exact match
    auto dogs = matcher.find_nodes(AtomTypes::CONCEPT_NODE, "Dog");
    EXPECT_EQ(dogs.size(), 1);

    // Wildcard match
    auto all_concepts = matcher.find_nodes(AtomTypes::CONCEPT_NODE, "*");
    EXPECT_EQ(all_concepts.size(), 4);

    // Prefix match
    auto expenses = matcher.find_nodes(AtomTypes::ACCOUNT_NODE, "Expenses*");
    EXPECT_EQ(expenses.size(), 3);
}

TEST_F(PatternMatchTest, FindLinksTo)
{
    PatternMatcher matcher(atomspace);

    auto animal = atomspace.get_node(AtomTypes::CONCEPT_NODE, "Animal");
    auto links = matcher.find_links_to(AtomTypes::INHERITANCE_LINK, animal);

    EXPECT_EQ(links.size(), 3);  // Dog, Cat, Bird all inherit from Animal
}

TEST_F(PatternMatchTest, FindLinksContaining)
{
    PatternMatcher matcher(atomspace);

    auto dog = atomspace.get_node(AtomTypes::CONCEPT_NODE, "Dog");
    auto links = matcher.find_links_containing(AtomTypes::INHERITANCE_LINK, dog);

    EXPECT_EQ(links.size(), 1);  // Dog -> Animal
}

TEST_F(PatternMatchTest, SimplePatternMatch)
{
    PatternMatcher matcher(atomspace);

    // Find all X where (InheritanceLink X Animal)
    Pattern pattern;
    pattern.add_variable("X", AtomTypes::CONCEPT_NODE);

    auto animal = atomspace.get_node(AtomTypes::CONCEPT_NODE, "Animal");
    auto var_x = create_node(AtomTypes::VARIABLE_NODE, "X");
    pattern.set_body(create_link(AtomTypes::INHERITANCE_LINK, var_x, animal));

    auto results = matcher.match(pattern);

    // Should find Dog, Cat, Bird
    EXPECT_EQ(results.size(), 3);
}

TEST_F(PatternMatchTest, QueryBuilder)
{
    auto animal = atomspace.get_node(AtomTypes::CONCEPT_NODE, "Animal");
    auto var_x = create_node(AtomTypes::VARIABLE_NODE, "X");

    auto results = QueryBuilder(atomspace)
        .var("X", AtomTypes::CONCEPT_NODE)
        .body(create_link(AtomTypes::INHERITANCE_LINK, var_x, animal))
        .execute();

    EXPECT_EQ(results.size(), 3);

    // Check that each result has X bound
    for (const auto& binding : results) {
        EXPECT_TRUE(binding.find("X") != binding.end());
    }
}

TEST_F(PatternMatchTest, AccountHierarchyPattern)
{
    PatternMatcher matcher(atomspace);

    // Find all accounts under Expenses
    auto expenses = atomspace.get_node(AtomTypes::ACCOUNT_NODE, "Expenses");
    auto links = matcher.find_links_to(AtomTypes::ACCOUNT_HIERARCHY_LINK, expenses);

    EXPECT_EQ(links.size(), 2);  // Food and Transportation
}
