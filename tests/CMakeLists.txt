find_package(Catch2 REQUIRED)
find_package(trompeloeil REQUIRED)

set(
    SOURCE_LIST
    actTest.cpp
    agentsTest.cpp
    algorithmTest.cpp
    environmentsTest.cpp
    linearTest.cpp
    resultsTest.cpp)

add_executable(introRLtests ${SOURCE_LIST})
target_compile_features(introRLtests PRIVATE cxx_std_23)
target_link_libraries(
    introRLtests
    PRIVATE
    introRL
    ArrayFire::af
    Catch2::Catch2WithMain
    trompeloeil::trompeloeil)