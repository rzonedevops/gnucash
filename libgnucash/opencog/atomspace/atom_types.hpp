/*
 * opencog/atomspace/atom_types.hpp
 *
 * Atom type definitions for the AtomSpace
 * Based on OpenCog Atomese
 *
 * Copyright (C) 2024 GnuCash Developers
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef GNC_OPENCOG_ATOM_TYPES_HPP
#define GNC_OPENCOG_ATOM_TYPES_HPP

#include <cstdint>
#include <string>
#include <unordered_map>

namespace gnc {
namespace opencog {

/**
 * Type identifier for atoms.
 * Each atom type has a unique ID.
 */
using AtomType = uint16_t;

/**
 * Core atom types for the GnuCash cognitive framework.
 * These form the foundation of knowledge representation.
 */
namespace AtomTypes
{
    // Base types
    constexpr AtomType ATOM = 0;
    constexpr AtomType NODE = 1;
    constexpr AtomType LINK = 2;

    // Node types
    constexpr AtomType CONCEPT_NODE = 10;
    constexpr AtomType PREDICATE_NODE = 11;
    constexpr AtomType SCHEMA_NODE = 12;
    constexpr AtomType VARIABLE_NODE = 13;
    constexpr AtomType NUMBER_NODE = 14;
    constexpr AtomType TYPE_NODE = 15;

    // GnuCash-specific node types
    constexpr AtomType ACCOUNT_NODE = 100;
    constexpr AtomType TRANSACTION_NODE = 101;
    constexpr AtomType SPLIT_NODE = 102;
    constexpr AtomType COMMODITY_NODE = 103;
    constexpr AtomType PRICE_NODE = 104;
    constexpr AtomType VENDOR_NODE = 105;
    constexpr AtomType CUSTOMER_NODE = 106;
    constexpr AtomType CATEGORY_NODE = 107;
    constexpr AtomType DATE_NODE = 108;
    constexpr AtomType AMOUNT_NODE = 109;

    // Link types - Structural
    constexpr AtomType LIST_LINK = 200;
    constexpr AtomType SET_LINK = 201;
    constexpr AtomType ORDERED_LINK = 202;
    constexpr AtomType UNORDERED_LINK = 203;

    // Link types - Logical
    constexpr AtomType AND_LINK = 210;
    constexpr AtomType OR_LINK = 211;
    constexpr AtomType NOT_LINK = 212;
    constexpr AtomType IMPLICATION_LINK = 213;
    constexpr AtomType EQUIVALENCE_LINK = 214;

    // Link types - Relational
    constexpr AtomType INHERITANCE_LINK = 220;
    constexpr AtomType SIMILARITY_LINK = 221;
    constexpr AtomType MEMBER_LINK = 222;
    constexpr AtomType EVALUATION_LINK = 223;
    constexpr AtomType EXECUTION_LINK = 224;

    // Link types - Contextual
    constexpr AtomType CONTEXT_LINK = 230;
    constexpr AtomType DEFINE_LINK = 231;
    constexpr AtomType STATE_LINK = 232;

    // GnuCash-specific link types
    constexpr AtomType TRANSACTION_LINK = 300;      // Links transaction to splits
    constexpr AtomType ACCOUNT_HIERARCHY_LINK = 301; // Parent-child account relationship
    constexpr AtomType CATEGORIZATION_LINK = 302;   // Transaction categorization
    constexpr AtomType TEMPORAL_LINK = 303;         // Time-based relationships
    constexpr AtomType FLOW_LINK = 304;             // Money flow between accounts
    constexpr AtomType PATTERN_LINK = 305;          // Recognized spending patterns
    constexpr AtomType PREDICTION_LINK = 306;       // Predicted future transactions
    constexpr AtomType ANOMALY_LINK = 307;          // Unusual transaction markers

    // Pattern matching types
    constexpr AtomType BIND_LINK = 400;
    constexpr AtomType GET_LINK = 401;
    constexpr AtomType QUERY_LINK = 402;
    constexpr AtomType SATISFACTION_LINK = 403;
    constexpr AtomType PRESENT_LINK = 404;
    constexpr AtomType ABSENT_LINK = 405;

    // Maximum type value
    constexpr AtomType MAX_TYPE = 500;
}

/**
 * Get the name of an atom type.
 */
inline const char* atom_type_name(AtomType type)
{
    static const std::unordered_map<AtomType, const char*> names = {
        {AtomTypes::ATOM, "Atom"},
        {AtomTypes::NODE, "Node"},
        {AtomTypes::LINK, "Link"},
        {AtomTypes::CONCEPT_NODE, "ConceptNode"},
        {AtomTypes::PREDICATE_NODE, "PredicateNode"},
        {AtomTypes::SCHEMA_NODE, "SchemaNode"},
        {AtomTypes::VARIABLE_NODE, "VariableNode"},
        {AtomTypes::NUMBER_NODE, "NumberNode"},
        {AtomTypes::TYPE_NODE, "TypeNode"},
        {AtomTypes::ACCOUNT_NODE, "AccountNode"},
        {AtomTypes::TRANSACTION_NODE, "TransactionNode"},
        {AtomTypes::SPLIT_NODE, "SplitNode"},
        {AtomTypes::COMMODITY_NODE, "CommodityNode"},
        {AtomTypes::PRICE_NODE, "PriceNode"},
        {AtomTypes::VENDOR_NODE, "VendorNode"},
        {AtomTypes::CUSTOMER_NODE, "CustomerNode"},
        {AtomTypes::CATEGORY_NODE, "CategoryNode"},
        {AtomTypes::DATE_NODE, "DateNode"},
        {AtomTypes::AMOUNT_NODE, "AmountNode"},
        {AtomTypes::LIST_LINK, "ListLink"},
        {AtomTypes::SET_LINK, "SetLink"},
        {AtomTypes::AND_LINK, "AndLink"},
        {AtomTypes::OR_LINK, "OrLink"},
        {AtomTypes::NOT_LINK, "NotLink"},
        {AtomTypes::IMPLICATION_LINK, "ImplicationLink"},
        {AtomTypes::INHERITANCE_LINK, "InheritanceLink"},
        {AtomTypes::SIMILARITY_LINK, "SimilarityLink"},
        {AtomTypes::MEMBER_LINK, "MemberLink"},
        {AtomTypes::EVALUATION_LINK, "EvaluationLink"},
        {AtomTypes::CONTEXT_LINK, "ContextLink"},
        {AtomTypes::TRANSACTION_LINK, "TransactionLink"},
        {AtomTypes::ACCOUNT_HIERARCHY_LINK, "AccountHierarchyLink"},
        {AtomTypes::CATEGORIZATION_LINK, "CategorizationLink"},
        {AtomTypes::TEMPORAL_LINK, "TemporalLink"},
        {AtomTypes::FLOW_LINK, "FlowLink"},
        {AtomTypes::PATTERN_LINK, "PatternLink"},
        {AtomTypes::PREDICTION_LINK, "PredictionLink"},
        {AtomTypes::ANOMALY_LINK, "AnomalyLink"},
        {AtomTypes::BIND_LINK, "BindLink"},
        {AtomTypes::GET_LINK, "GetLink"},
        {AtomTypes::QUERY_LINK, "QueryLink"},
    };

    auto it = names.find(type);
    return (it != names.end()) ? it->second : "UnknownType";
}

/**
 * Check if a type is a node type.
 */
inline bool is_node_type(AtomType type)
{
    return (type >= AtomTypes::NODE && type < AtomTypes::LIST_LINK) ||
           (type >= AtomTypes::ACCOUNT_NODE && type < AtomTypes::LIST_LINK);
}

/**
 * Check if a type is a link type.
 */
inline bool is_link_type(AtomType type)
{
    return type >= AtomTypes::LIST_LINK;
}

} // namespace opencog
} // namespace gnc

#endif // GNC_OPENCOG_ATOM_TYPES_HPP
