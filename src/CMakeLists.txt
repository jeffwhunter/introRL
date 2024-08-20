file(GLOB_RECURSE HEADER_LIST ${IntroRL_SOURCE_DIR}/include/*.hpp)

source_group(
    TREE "${PROJECT_SOURCE_DIR}/include"
    PREFIX "Header Files"
    FILES ${HEADER_LIST})

file(GLOB_RECURSE SOURCE_LIST *.cpp)

source_group(
    TREE "${CMAKE_CURRENT_SOURCE_DIR}"
    PREFIX "Source Files"
    FILES ${SOURCE_LIST})

add_library(introRL ${HEADER_LIST} ${SOURCE_LIST})

target_compile_definitions(introRL PRIVATE NOMINMAX=1)
target_compile_features(introRL PUBLIC cxx_std_23)
target_include_directories(introRL PUBLIC ../include)
target_link_libraries(
    introRL
    PRIVATE
    ArrayFire::af
    blend2d::blend2d
    Matplot++::matplot
    mp++::mp++
    twig::stronk)