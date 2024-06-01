find_package(Catch2 REQUIRED)
find_package(trompeloeil REQUIRED)

set(HEADER_LIST testUtils.hpp)

set(
    SOURCE_LIST
    actionsTests.cpp
    environmentsTests.cpp
    linearTests.cpp
    reinforcementTests.cpp)

add_executable(introRLtests ${HEADER_LIST} ${SOURCE_LIST})
target_compile_features(introRLtests PRIVATE cxx_std_23)
target_link_libraries(
    introRLtests
    PRIVATE
    introRL
    ArrayFire::af
    Catch2::Catch2WithMain
    trompeloeil::trompeloeil)