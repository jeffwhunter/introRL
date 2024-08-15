#include <vector>

#include <arrayfire.h>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <catch2/catch_test_macros.hpp>

#include <introRL/afUtils.hpp>
#include <introRL/math/af.hpp>

namespace irl::math
{
    TEST_CASE("math.af.argMax.returns indices of the maximum")
    {
        constexpr unsigned rows{4};
        constexpr unsigned columns{5};

        auto m{af::moddims(af::range(af::dim4{20}), af::dim4{rows, columns}) % columns};

        REQUIRE_THAT(
            toVector<unsigned>(argMax<0>(m)),
            Catch::Matchers::RangeEquals(std::to_array({3, 0, 1, 2, 3})));

        REQUIRE_THAT(
            toVector<unsigned>(argMax<1>(m)),
            Catch::Matchers::RangeEquals(std::to_array({1, 2, 3, 4})));
    }

    TEST_CASE("math.af.power.handles normal values")
    {
        REQUIRE(power(2, 3) == 8);
        REQUIRE(power(3, 2) == 9);
    }

    TEST_CASE("math.af.power.handles zero base")
    {
        REQUIRE(power(0, 2) == 0);
        REQUIRE(power(0, 3) == 0);
    }

    TEST_CASE("math.af.power.handles zero power")
    {
        REQUIRE(power(2, 0) == 1);
        REQUIRE(power(3, 0) == 1);
    }

    TEST_CASE("math.af.round.rounds properly")
    {
        REQUIRE_THAT(
            toVector<float>(round(af::array{.1f, .23f, .456f, .7890f}, 2)),
            Catch::Matchers::RangeEquals(std::vector{.1f, .23f, .46f, .79f}));

        REQUIRE_THAT(
            toVector<double>(round(af::array{-.1, -.02, -.003, -.0004}, 3)),
            Catch::Matchers::RangeEquals(std::vector{-.1, -.02, -.003, .0}));
    }
}