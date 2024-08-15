#include <catch2/catch_test_macros.hpp>

#include <introRL/math/map.hpp>

namespace irl::math
{
    TEST_CASE("math.map.argmax.returns the key of the maximum")
    {
        std::map<int, int> values{
            {3, 7},
            {1, 0},
            {12, 9},
            {-5, 2},
            {-17, -19},
            {55, -55},
            {8, 100}};

        REQUIRE(argmax(values) == 8);
    }
}