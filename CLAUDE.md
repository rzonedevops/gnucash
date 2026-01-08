# CLAUDE.md - GnuCash Development Guide

## Project Overview

GnuCash is a personal and small business double-entry accounting application written primarily in C/C++ with Guile (Scheme) scripting support. The project uses CMake as its build system and supports Linux, macOS, and Windows platforms.

- **Homepage**: https://www.gnucash.org/
- **Wiki**: https://wiki.gnucash.org/wiki/GnuCash
- **Bug Reports**: https://bugs.gnucash.org

## Project Structure

```
gnucash/              # Main application code
├── gnome/            # GNOME/GTK UI components
├── gnome-utils/      # GTK utility functions
├── gnome-search/     # Search dialog components
├── import-export/    # Import/export modules (CSV, OFX, QIF, etc.)
├── register/         # Account register UI
├── report/           # Report generation (Scheme-based)
├── gtkbuilder/       # GTK Builder UI definitions
├── gschemas/         # GSettings schema files
└── html/             # HTML templates

libgnucash/           # Core library code
├── engine/           # Core accounting engine (accounts, transactions, splits)
├── app-utils/        # Application utilities
├── backend/          # Data storage backends
│   ├── xml/          # XML file backend
│   ├── sql/          # SQL backend base
│   └── dbi/          # libdbi SQL implementation
├── core-utils/       # Low-level utilities
├── gnc-module/       # Module loading system
├── quotes/           # Online price quote retrieval
├── tax/              # Tax-related calculations
└── opencog/          # OpenCog Cognitive Subsystem (AI-powered intelligence)
    ├── cogutil/      # Core utilities (threading, logging, counters)
    ├── atomspace/    # Hypergraph knowledge database
    ├── pattern/      # Pattern matching engine
    ├── reasoning/    # Probabilistic reasoning (PLN)
    └── gnc-cognitive/# GnuCash-specific cognitive integration

bindings/             # Language bindings
├── guile/            # Guile (Scheme) bindings
└── python/           # Python bindings (optional)

common/               # Shared code and cmake modules
├── cmake_modules/    # Custom CMake modules

data/                 # Data files (icons, account templates)
po/                   # Translation files
doc/                  # Documentation and examples
```

## Build System

GnuCash uses **CMake** (minimum 3.14.5) with optional **Ninja** for faster builds.

### Quick Build (Out-of-Tree)

```bash
# Create and enter build directory
mkdir ~/gnucash-build && cd ~/gnucash-build

# Configure with CMake (Ninja generator recommended)
cmake -G Ninja -DCMAKE_INSTALL_PREFIX=/opt/gnucash /path/to/source

# Build
ninja

# Run tests
ninja check

# Install (optional)
ninja install
```

### Common CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `-DWITH_SQL=ON/OFF` | ON | SQL backend support (libdbi) |
| `-DWITH_AQBANKING=ON/OFF` | ON | Online banking support |
| `-DWITH_GNUCASH=ON/OFF` | ON | Build full application (vs library only) |
| `-DWITH_OFX=ON/OFF` | ON | OFX import support |
| `-DWITH_PYTHON=ON/OFF` | OFF | Python bindings |
| `-DCMAKE_BUILD_TYPE=Debug/Release/Asan` | - | Build type |

### Running from Build Directory

```bash
bin/gnucash
```

## Testing

Tests use **Google Test** framework. Run all tests with:

```bash
# Using Ninja
ninja check

# Using Make
make check
```

Test logs are written to: `Testing/Temporary/LastTest.log`

### Test File Patterns

- `test-*.cpp` - Traditional test files
- `gtest-*.cpp` - Google Test based tests
- `utest-*.cpp` - Unit tests

## Coding Standards

Refer to: https://wiki.gnucash.org/wiki/CodingStandard

### Key Conventions

- **C++ Standard**: C++17
- **Compiler**: GCC 8.0+ or Clang 6.0+
- **Comments**: C-style preferred, C++ style (`//`) discouraged but tolerated
- **Build Directory**: Always use out-of-tree builds (preferably outside source tree)

## Key Dependencies

| Dependency | Minimum Version | Purpose |
|------------|-----------------|---------|
| GLib2 | 2.68.1 | Core library |
| GTK+3 | 3.22.30 | UI toolkit |
| Guile | 2.0.9/2.2/3.0 | Scheme scripting |
| Boost | 1.67.0 | C++ utilities |
| libxml2 | 2.9.4 | XML processing |
| SWIG | 3.0.12 | Bindings generator |
| Google Test | 1.8.0 | Testing framework |
| libdbi | 0.8.3 | SQL backend |

## Debugging

### Enable Trace Messages

See `libgnucash/engine/qoflog.h` for logging documentation.

### GDB Usage

```bash
gdb /path/to/gnucash
(gdb) catch fork
(gdb) set follow-fork-mode child
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `GNC_DEBUG` | Enable early debugging output |
| `GUILE_LOAD_PATH` | Override Guile load path |
| `GNC_MODULE_PATH` | Override module load path |

## Common Development Tasks

### Adding a New Report

Reports are Scheme-based and located in `gnucash/report/reports/`. Follow existing patterns in that directory.

### Adding Import/Export Format

See `gnucash/import-export/` for existing implementations (CSV, OFX, QIF).

### Modifying the Engine

Core accounting logic is in `libgnucash/engine/`. Key files:
- `Account.cpp` - Account handling
- `Transaction.cpp` - Transaction handling
- `Split.cpp` - Split (entry) handling
- `gnc-commodity.cpp` - Currency/commodity handling

## CI/CD

GitHub Actions workflows in `.github/workflows/`:
- `ci-tests.yml` - Main CI tests (Ubuntu, ASAN)
- `coverage.yml` - Code coverage
- `mac-tests.yaml` - macOS tests

## OpenCog Cognitive Subsystem

GnuCash includes an integrated cognitive framework based on OpenCog architecture, providing AI-powered financial intelligence.

### Components

| Component | Purpose |
|-----------|---------|
| **AtomSpace** | Hypergraph database for knowledge representation |
| **Pattern Matcher** | Graph query engine for finding patterns |
| **Cognitive Engine** | GnuCash-specific AI integration |
| **Truth Values** | Probabilistic reasoning with confidence |

### Key Features

- **Automatic Transaction Categorization**: AI-powered category suggestions
- **Spending Pattern Detection**: Identifies recurring transactions
- **Anomaly Detection**: Flags unusual transactions
- **Financial Predictions**: Cash flow forecasting
- **Natural Language Queries**: Ask questions about your finances

### Usage Example

```cpp
#include <opencog/gnc-cognitive/cognitive_engine.hpp>

using namespace gnc::opencog;

// Initialize the cognitive engine
auto& engine = cognitive_engine();
engine.initialize();

// Categorize a transaction
auto result = engine.categorize_transaction("Walmart Grocery", 45.00, "Walmart");
// result.category = "Groceries", result.confidence = 0.85

// Detect spending patterns
auto patterns = engine.detect_spending_patterns();

// Natural language query
std::string answer = engine.query("What are my recurring expenses?");
```

### GnuCash-Specific Atom Types

The cognitive subsystem defines financial atom types:

- `AccountNode`, `TransactionNode`, `SplitNode` - Core financial entities
- `VendorNode`, `CategoryNode` - Classification nodes
- `CategorizationLink`, `FlowLink`, `PatternLink` - Relationship links
- `AnomalyLink`, `PredictionLink` - AI-detected insights

### Running Cognitive Tests

```bash
# Build and run OpenCog tests
ninja gtest-atomspace gtest-pattern-match gtest-cognitive-engine
./libgnucash/opencog/test/gtest-atomspace
./libgnucash/opencog/test/gtest-cognitive-engine
```

## Useful Commands

```bash
# Generate tags
make TAGS

# Build distribution tarball
ninja dist
ninja distcheck

# Check exported symbols (from build directory)
nm -A `find . -name '*.so'` | grep ' T '
```
