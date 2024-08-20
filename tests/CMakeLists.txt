find_package(Catch2 REQUIRED)
find_package(trompeloeil REQUIRED)

file(GLOB_RECURSE SOURCE_LIST *.cpp)

source_group(
    TREE "${CMAKE_CURRENT_SOURCE_DIR}"
    PREFIX "Source Files"
    FILES ${SOURCE_LIST})

add_executable(introRLtests ${SOURCE_LIST})
target_compile_definitions(introRLtests PRIVATE NOMINMAX=1)
target_compile_features(introRLtests PRIVATE cxx_std_23)
target_link_libraries(
    introRLtests
    PRIVATE
    introRL
    ArrayFire::af
    Catch2::Catch2WithMain
    mp++::mp++
    trompeloeil::trompeloeil)