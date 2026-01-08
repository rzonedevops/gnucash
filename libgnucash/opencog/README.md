# GnuCash OpenCog Cognitive Subsystem

## Overview

This subsystem integrates OpenCog's cognitive architecture into GnuCash, providing AI-powered financial intelligence capabilities. Based on the OpenCog framework developed by Dr. Ben Goertzel and contributors, this implementation brings knowledge representation, pattern matching, probabilistic reasoning, and tensor-enhanced accounting to personal finance management.

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

### 4. ATen Tensor Library (`aten/`)
High-performance tensor operations based on PyTorch ATen:
- **Tensor**: Multi-dimensional array with broadcasting
- **tensor_ops**: Advanced operations (softmax, normalization, etc.)
- Matrix operations, aggregations, element-wise math

### 5. ATenSpace (`atenspace/`)
Hybrid symbolic-neural knowledge representation:
- **TensorAtom**: Atoms with embedded tensor representations
- **TensorEmbedding**: Semantic, temporal, structural embeddings
- **ATenSpace**: AtomSpace extended with tensor search
- Attention mechanisms and similarity search

### 6. Tensor Logic (`tensor-logic/`)
Multi-Entity, Multi-Scale, Network-Aware Tensor-Enhanced Accounting:
- **TensorAccount**: Accounts as multi-dimensional tensors
- **TensorNetwork**: Graph representation of money flows
- **TensorLogicEngine**: Unified interface for tensor operations

### 7. GnuCash Cognitive (`gnc-cognitive/`)
Integration layer for GnuCash-specific intelligence:
- **CognitiveEngine**: Main interface for AI features
- Transaction categorization, pattern detection, anomaly detection

## ATen Tensor Library

The ATen library provides efficient tensor operations for financial computations.

### Basic Usage

```cpp
#include <opencog/aten/tensor.hpp>
using namespace gnc::aten;

// Create tensors
auto zeros = DoubleTensor::zeros({3, 4});
auto ones = DoubleTensor::ones({3, 4});
auto data = DoubleTensor({1.0, 2.0, 3.0, 4.0});

// Operations
auto result = zeros + ones;        // Element-wise addition
auto product = data * 2.0;         // Scalar multiplication
auto matrix = DoubleTensor::eye(3); // Identity matrix
auto matmul = a.matmul(b);         // Matrix multiplication

// Aggregations
double sum = data.sum();
double mean = data.mean();
double std = data.std();
```

## ATenSpace - Hybrid Knowledge Representation

ATenSpace bridges symbolic AI (AtomSpace) with neural embeddings.

```cpp
#include <opencog/atenspace/atenspace.hpp>
using namespace gnc::atenspace;

ATenSpace space;

// Create tensor-enabled nodes
auto node = space.add_tensor_node(AtomTypes::ACCOUNT_NODE, "Expenses:Food");

// Set semantic embedding
auto embedding = space.generate_semantic_embedding("Expenses:Food");
space.set_embedding(node, EmbeddingType::SEMANTIC, embedding);

// Semantic similarity search
auto similar = space.semantic_search("food expenses", 5);

// Attention-based aggregation
auto attention = space.compute_attention(atoms, query);
auto aggregated = space.attention_aggregate(atoms, query);
```

## Tensor Logic Engine

The Tensor Logic Engine provides multi-entity, multi-scale, network-aware accounting.

### Multi-Entity Accounting

```cpp
#include <opencog/tensor-logic/tensor_logic_engine.hpp>
using namespace gnc::tensor_logic;

auto& engine = tensor_logic_engine();
engine.initialize();

// Create tensor account (3 entities, 12 periods, 1 currency)
auto account = engine.create_account("guid", "Expenses", 3, 12, 1);

// Import data for each entity
engine.import_account_data("guid", 0, 0, 0, 1000.0, 500.0, 200.0);
engine.import_account_data("guid", 1, 0, 0, 2000.0, 800.0, 400.0);
engine.import_account_data("guid", 2, 0, 0, 1500.0, 600.0, 300.0);

// Consolidate across entities
auto consolidated = engine.consolidate_entities("guid");
```

### Multi-Scale Analysis

```cpp
// Analyze at multiple time scales
auto analysis = engine.multi_scale_analysis("guid");

// Aggregate to quarterly
auto quarterly = engine.aggregate_to_scale("guid", TimeScale::QUARTERLY);
```

### Network Flow Analysis

```cpp
// Record transactions
engine.record_transaction("checking", "expenses_food", 50.0, 0);
engine.record_transaction("checking", "expenses_gas", 30.0, 0);
engine.record_transaction("income", "checking", 2000.0, 0);

// Analyze network
auto flow_analysis = engine.analyze_flow_network();

// Find flow paths
auto path = engine.find_flow_path("income", "expenses_food");

// Detect circular flows
auto cycles = engine.detect_circular_flows();

// Detect anomalies
auto anomalies = engine.detect_flow_anomalies(2.0);
```

### Forecasting

```cpp
// Forecast balance
auto balance_forecast = engine.forecast_balance("guid", 3);
// balance_forecast.predicted_values, balance_forecast.confidence_intervals

// Forecast cash flow
auto cash_forecast = engine.forecast_cash_flow("guid", 3);
```

### Similarity & Clustering

```cpp
// Find similar accounts
auto similar = engine.find_similar_accounts("guid", 5);

// Cluster accounts by behavior
auto clusters = engine.cluster_accounts(3);
```

## GnuCash-Specific Atom Types

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

## References

- [OpenCog Wiki](https://wiki.opencog.org/)
- [AtomSpace Documentation](https://wiki.opencog.org/w/AtomSpace)
- [Pattern Matching](https://wiki.opencog.org/w/Pattern_matching)
- [PyTorch ATen](https://pytorch.org/cppdocs/)
- [ATenSpace (o9nn)](https://github.com/o9nn/ATenSpace)

## License

This subsystem is part of GnuCash and is licensed under GPL-2.0-or-later.
