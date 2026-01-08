# GnuCash OpenCog Cognitive Subsystem

## Overview

This subsystem integrates OpenCog's cognitive architecture into GnuCash, providing AI-powered financial intelligence capabilities. Based on the OpenCog framework developed by Dr. Ben Goertzel and contributors, this implementation brings knowledge representation, pattern matching, and probabilistic reasoning to personal finance management.

## Architecture

The cognitive subsystem consists of several key components:

### 1. Cogutil (`cogutil/`)
Low-level C++ utilities providing:
- **ConcurrentQueue**: Thread-safe queue for async operations
- **Counter**: Python-style counter for statistical operations
- **Logger**: Logging infrastructure for the cognitive subsystem

### 2. AtomSpace (`atomspace/`)
A hypergraph database for knowledge representation:
- **Atoms**: The fundamental units (Nodes and Links)
- **TruthValue**: Probabilistic truth with strength and confidence
- **AtomTypes**: GnuCash-specific atom types for financial data
- **AtomSpace**: The main database with indexing and querying

### 3. Pattern Matching (`pattern/`)
Graph pattern matching engine:
- **Pattern**: Defines patterns with variables and constraints
- **PatternMatcher**: Executes pattern queries against AtomSpace
- **QueryBuilder**: Fluent interface for constructing queries

### 4. GnuCash Cognitive (`gnc-cognitive/`)
Integration layer for GnuCash-specific intelligence:
- **CognitiveEngine**: Main interface for AI features
- Transaction categorization
- Spending pattern detection
- Anomaly detection
- Financial predictions
- Natural language queries

## GnuCash-Specific Atom Types

The system defines custom atom types for financial data:

| Type | Purpose |
|------|---------|
| `AccountNode` | Represents GnuCash accounts |
| `TransactionNode` | Represents transactions |
| `SplitNode` | Represents transaction splits |
| `VendorNode` | Represents payees/vendors |
| `CategoryNode` | Represents spending categories |
| `AmountNode` | Represents monetary amounts |
| `DateNode` | Represents transaction dates |

### Link Types

| Type | Purpose |
|------|---------|
| `TransactionLink` | Connects transactions to splits |
| `AccountHierarchyLink` | Parent-child account relationships |
| `CategorizationLink` | Associates transactions with categories |
| `FlowLink` | Money flow between accounts |
| `PatternLink` | Detected spending patterns |
| `AnomalyLink` | Marks unusual transactions |

## Features

### Automatic Transaction Categorization
```cpp
auto result = engine.categorize_transaction("Grocery Store", 45.00, "Walmart");
// result.category = "Groceries"
// result.confidence = 0.85
```

### Pattern Detection
```cpp
auto patterns = engine.detect_spending_patterns();
for (const auto& p : patterns) {
    // p.name, p.frequency, p.average_amount
}
```

### Anomaly Detection
```cpp
auto anomalies = engine.detect_anomalies(2.0);  // 2 std deviations
// Returns unusual transactions
```

### Natural Language Queries
```cpp
std::string answer = engine.query("What are my spending patterns?");
```

## Building

The subsystem is built as part of the normal GnuCash build:

```bash
mkdir build && cd build
cmake -G Ninja ..
ninja gnc-opencog
```

### Running Tests
```bash
ninja check
# Or specifically:
./libgnucash/opencog/test/gtest-atomspace
./libgnucash/opencog/test/gtest-pattern-match
./libgnucash/opencog/test/gtest-cognitive-engine
```

## Future Development

Planned features include:
- **Probabilistic Logic Networks (PLN)**: Advanced reasoning
- **MOSES**: Learning financial rules through genetic programming
- **Attention Allocation**: Focus on important financial data
- **Natural Language Understanding**: Parse financial queries
- **Budget Optimization**: AI-powered budget recommendations

## References

- [OpenCog Wiki](https://wiki.opencog.org/)
- [AtomSpace Documentation](https://wiki.opencog.org/w/AtomSpace)
- [Pattern Matching](https://wiki.opencog.org/w/Pattern_matching)
- [Probabilistic Logic Networks](https://wiki.opencog.org/w/Probabilistic_Logic_Networks)

## License

This subsystem is part of GnuCash and is licensed under GPL-2.0-or-later.
