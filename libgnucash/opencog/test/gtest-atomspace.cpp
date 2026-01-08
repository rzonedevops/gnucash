/*
 * gtest-atomspace.cpp
 *
 * Unit tests for AtomSpace hypergraph database
 *
 * Copyright (C) 2024 GnuCash Developers
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include <gtest/gtest.h>
#include "../atomspace/atomspace.hpp"

using namespace gnc::opencog;

class AtomSpaceTest : public ::testing::Test
{
protected:
    AtomSpace atomspace;
};

TEST_F(AtomSpaceTest, AddNode)
{
    auto node = atomspace.add_node(AtomTypes::CONCEPT_NODE, "TestConcept");

    ASSERT_NE(node, nullptr);
    EXPECT_TRUE(node->is_node());
    EXPECT_FALSE(node->is_link());
    EXPECT_EQ(node->type(), AtomTypes::CONCEPT_NODE);
    EXPECT_EQ(node->name(), "TestConcept");
}

TEST_F(AtomSpaceTest, AddDuplicateNode)
{
    auto node1 = atomspace.add_node(AtomTypes::CONCEPT_NODE, "TestConcept");
    auto node2 = atomspace.add_node(AtomTypes::CONCEPT_NODE, "TestConcept");

    // Should return the same node
    EXPECT_EQ(node1, node2);
    EXPECT_EQ(atomspace.size(), 1);
}

TEST_F(AtomSpaceTest, AddLink)
{
    auto node1 = atomspace.add_node(AtomTypes::CONCEPT_NODE, "Animal");
    auto node2 = atomspace.add_node(AtomTypes::CONCEPT_NODE, "Dog");

    auto link = atomspace.add_link(AtomTypes::INHERITANCE_LINK, node2, node1);

    ASSERT_NE(link, nullptr);
    EXPECT_FALSE(link->is_node());
    EXPECT_TRUE(link->is_link());
    EXPECT_EQ(link->type(), AtomTypes::INHERITANCE_LINK);
    EXPECT_EQ(link->arity(), 2);
    EXPECT_EQ(link->outgoing_atom(0), node2);
    EXPECT_EQ(link->outgoing_atom(1), node1);
}

TEST_F(AtomSpaceTest, GetNode)
{
    atomspace.add_node(AtomTypes::CONCEPT_NODE, "TestConcept");

    auto found = atomspace.get_node(AtomTypes::CONCEPT_NODE, "TestConcept");
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->name(), "TestConcept");

    auto not_found = atomspace.get_node(AtomTypes::CONCEPT_NODE, "NonExistent");
    EXPECT_EQ(not_found, nullptr);
}

TEST_F(AtomSpaceTest, GetAtomsByType)
{
    atomspace.add_node(AtomTypes::CONCEPT_NODE, "Concept1");
    atomspace.add_node(AtomTypes::CONCEPT_NODE, "Concept2");
    atomspace.add_node(AtomTypes::PREDICATE_NODE, "Predicate1");

    auto concepts = atomspace.get_atoms_by_type(AtomTypes::CONCEPT_NODE);
    EXPECT_EQ(concepts.size(), 2);

    auto predicates = atomspace.get_atoms_by_type(AtomTypes::PREDICATE_NODE);
    EXPECT_EQ(predicates.size(), 1);
}

TEST_F(AtomSpaceTest, IncomingSet)
{
    auto animal = atomspace.add_node(AtomTypes::CONCEPT_NODE, "Animal");
    auto dog = atomspace.add_node(AtomTypes::CONCEPT_NODE, "Dog");
    auto cat = atomspace.add_node(AtomTypes::CONCEPT_NODE, "Cat");

    atomspace.add_link(AtomTypes::INHERITANCE_LINK, dog, animal);
    atomspace.add_link(AtomTypes::INHERITANCE_LINK, cat, animal);

    auto incoming = atomspace.get_incoming(animal);
    EXPECT_EQ(incoming.size(), 2);
}

TEST_F(AtomSpaceTest, RemoveAtom)
{
    auto node = atomspace.add_node(AtomTypes::CONCEPT_NODE, "ToRemove");
    EXPECT_EQ(atomspace.size(), 1);

    bool removed = atomspace.remove_atom(node);
    EXPECT_TRUE(removed);
    EXPECT_EQ(atomspace.size(), 0);

    // Should not find after removal
    auto not_found = atomspace.get_node(AtomTypes::CONCEPT_NODE, "ToRemove");
    EXPECT_EQ(not_found, nullptr);
}

TEST_F(AtomSpaceTest, TruthValue)
{
    auto node = atomspace.add_node(AtomTypes::CONCEPT_NODE, "TestTruthValue");

    // Default truth value
    EXPECT_TRUE(node->truth_value().is_default());

    // Set truth value
    node->set_truth_value(stv(0.8, 0.9));
    EXPECT_DOUBLE_EQ(node->truth_value().strength(), 0.8);
    EXPECT_DOUBLE_EQ(node->truth_value().confidence(), 0.9);
}

TEST_F(AtomSpaceTest, GnuCashSpecificTypes)
{
    auto account = atomspace.add_node(AtomTypes::ACCOUNT_NODE, "Expenses:Food");
    auto transaction = atomspace.add_node(AtomTypes::TRANSACTION_NODE, "txn-001");
    auto vendor = atomspace.add_node(AtomTypes::VENDOR_NODE, "Grocery Store");

    EXPECT_EQ(account->type(), AtomTypes::ACCOUNT_NODE);
    EXPECT_EQ(transaction->type(), AtomTypes::TRANSACTION_NODE);
    EXPECT_EQ(vendor->type(), AtomTypes::VENDOR_NODE);

    // Create a categorization link
    auto category = atomspace.add_node(AtomTypes::CATEGORY_NODE, "Groceries");
    auto link = atomspace.add_link(AtomTypes::CATEGORIZATION_LINK, vendor, category);

    EXPECT_EQ(link->type(), AtomTypes::CATEGORIZATION_LINK);
}

TEST_F(AtomSpaceTest, AtomToString)
{
    auto node = atomspace.add_node(AtomTypes::CONCEPT_NODE, "TestConcept");
    std::string str = node->to_string();

    EXPECT_NE(str.find("ConceptNode"), std::string::npos);
    EXPECT_NE(str.find("TestConcept"), std::string::npos);
}

TEST_F(AtomSpaceTest, ConcurrentAccess)
{
    // Test thread safety with multiple threads
    const int num_threads = 4;
    const int nodes_per_thread = 100;

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, t, nodes_per_thread]() {
            for (int i = 0; i < nodes_per_thread; ++i) {
                std::string name = "Thread" + std::to_string(t) + "_Node" + std::to_string(i);
                atomspace.add_node(AtomTypes::CONCEPT_NODE, name);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(atomspace.size(), num_threads * nodes_per_thread);
}
