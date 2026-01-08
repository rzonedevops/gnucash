# CMakeLists.txt for OpenCog subsystem tests

set(OPENCOG_TEST_SOURCES
    gtest-atomspace.cpp
    gtest-pattern-match.cpp
    gtest-cognitive-engine.cpp
)

# Find Google Test
find_package(GTest REQUIRED)

foreach(test_source ${OPENCOG_TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)

    add_executable(${test_name} ${test_source})

    target_link_libraries(${test_name}
        PRIVATE
            gnc-opencog
            GTest::gtest
            GTest::gtest_main
    )

    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
