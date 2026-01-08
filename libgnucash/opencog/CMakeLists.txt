# CMakeLists.txt for GnuCash OpenCog Cognitive Subsystem
#
# This subsystem provides AI-powered financial intelligence based on
# OpenCog's cognitive architecture, including:
# - AtomSpace hypergraph database
# - ATen tensor library for numerical computations
# - ATenSpace hybrid symbolic-neural knowledge representation
# - Tensor Logic for multi-entity, multi-scale, network-aware accounting
# - Pattern matching engine
# - Probabilistic reasoning
# - Transaction categorization
# - Anomaly detection

set(OPENCOG_SOURCES
    gnc-cognitive/cognitive_engine.cpp
)

set(OPENCOG_HEADERS
    # Cogutil - Core utilities
    cogutil/concurrent_queue.hpp
    cogutil/counter.hpp
    cogutil/logger.hpp

    # AtomSpace - Hypergraph database
    atomspace/atom_types.hpp
    atomspace/truth_value.hpp
    atomspace/atom.hpp
    atomspace/atomspace.hpp

    # Pattern matching
    pattern/pattern_match.hpp

    # ATen - Tensor library
    aten/tensor.hpp
    aten/tensor_ops.hpp

    # ATenSpace - Hybrid symbolic-neural knowledge
    atenspace/tensor_atom.hpp
    atenspace/atenspace.hpp

    # Tensor Logic - Multi-entity, multi-scale, network-aware accounting
    tensor-logic/tensor_account.hpp
    tensor-logic/tensor_network.hpp
    tensor-logic/tensor_logic_engine.hpp

    # GnuCash cognitive integration
    gnc-cognitive/cognitive_engine.hpp
)

# Create the OpenCog library
add_library(gnc-opencog STATIC
    ${OPENCOG_SOURCES}
)

target_include_directories(gnc-opencog
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# C++17 is required for OpenCog features
target_compile_features(gnc-opencog PUBLIC cxx_std_17)

# Thread support is required
find_package(Threads REQUIRED)
target_link_libraries(gnc-opencog
    PUBLIC
        Threads::Threads
)

# Optional: Link with GLib for integration with GnuCash logging
if(GLIB2_FOUND)
    target_link_libraries(gnc-opencog
        PRIVATE
            PkgConfig::GLIB2
    )
endif()

# Installation
install(TARGETS gnc-opencog
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install headers
install(FILES ${OPENCOG_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gnucash/opencog
)

# Add tests subdirectory if it exists
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test)
    add_subdirectory(test)
endif()
