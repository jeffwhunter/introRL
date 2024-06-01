set(IRL_INCLUDE_DIR "${IntroRL_SOURCE_DIR}/include/introRL")

set(
    HEADER_LIST
    "${IRL_INCLUDE_DIR}/actions.hpp"
    "${IRL_INCLUDE_DIR}/charts.hpp"
    "${IRL_INCLUDE_DIR}/environments.hpp"
    "${IRL_INCLUDE_DIR}/linear.hpp"
    "${IRL_INCLUDE_DIR}/reinforcement.hpp")

source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${HEADER_LIST})

set(
    SOURCE_LIST
    actions.cpp
    charts.cpp
    environments.cpp
    linear.cpp
    reinforcement.cpp)

add_library(introRL ${HEADER_LIST} ${SOURCE_LIST})
target_compile_features(introRL PUBLIC cxx_std_23)
target_include_directories(introRL PUBLIC ../include)
target_link_libraries(introRL PRIVATE ArrayFire::af Matplot++::matplot)