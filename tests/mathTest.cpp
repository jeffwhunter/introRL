#include <vector>

#include <arrayfire.h>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <catch2/catch_test_macros.hpp>

#include <introRL/afUtils.hpp>
#include <introRL/math.hpp>

namespace irl::math
{
    TEST_CASE("math.power.handles normal values")
    {
        REQUIRE(power(2, 3) == 8);
        REQUIRE(power(3, 2) == 9);
    }

    TEST_CASE("math.power.handles zero base")
    {
        REQUIRE(power(0, 2) == 0);
        REQUIRE(power(0, 3) == 0);
    }

    TEST_CASE("math.power.handles zero power")
    {
        REQUIRE(power(2, 0) == 1);
        REQUIRE(power(3, 0) == 1);
    }

    TEST_CASE("math.round.rounds properly")
    {
        REQUIRE_THAT(
            toVector<float>(round(af::array{.1f, .23f, .456f, .7890f}, 2)),
            Catch::Matchers::RangeEquals(std::vector{.1f, .23f, .46f, .79f}));

        REQUIRE_THAT(
            toVector<double>(round(af::array{-.1, -.02, -.003, -.0004}, 3)),
            Catch::Matchers::RangeEquals(std::vector{-.1, -.02, -.003, .0}));
    }
}