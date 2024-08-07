set(IRL_INCLUDE_DIR "${IntroRL_SOURCE_DIR}/include/introRL")

set(
    HEADER_LIST
    "${IRL_INCLUDE_DIR}/act.hpp"
    "${IRL_INCLUDE_DIR}/afUtils.hpp"
    "${IRL_INCLUDE_DIR}/agents.hpp"
    "${IRL_INCLUDE_DIR}/algorithm.hpp"
    "${IRL_INCLUDE_DIR}/banditTypes.hpp"
    "${IRL_INCLUDE_DIR}/basicTypes.hpp"
    "${IRL_INCLUDE_DIR}/cartesian.hpp"
    "${IRL_INCLUDE_DIR}/environments.hpp"
    "${IRL_INCLUDE_DIR}/iteration.hpp"
    "${IRL_INCLUDE_DIR}/iterationTypes.hpp"
    "${IRL_INCLUDE_DIR}/linear.hpp"
    "${IRL_INCLUDE_DIR}/math.hpp"
    "${IRL_INCLUDE_DIR}/reduce.hpp"
    "${IRL_INCLUDE_DIR}/results.hpp"
    "${IRL_INCLUDE_DIR}/stats.hpp"
    "${IRL_INCLUDE_DIR}/subplotters.hpp")

source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${HEADER_LIST})

set(
    SOURCE_LIST
    act.cpp
    afUtils.cpp
    basicTypes.cpp
    environments.cpp
    iteration.cpp
    linear.cpp
    math.cpp
    results.cpp
    stats.cpp
    subplotters.cpp)

add_library(introRL ${HEADER_LIST} ${SOURCE_LIST})
target_compile_definitions(introRL PRIVATE NOMINMAX=1)
target_compile_features(introRL PUBLIC cxx_std_23)
target_include_directories(introRL PUBLIC ../include)
target_link_libraries(introRL PRIVATE ArrayFire::af Matplot++::matplot twig::stronk)