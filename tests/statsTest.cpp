#include <array>
#include <cmath>

#include <arrayfire.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <introRL/afUtils.hpp>
#include <introRL/stats.hpp>

namespace irl
{
    TEST_CASE("stats.probable.is equatable")
    {
        REQUIRE(Probable<int>{5, .4f} == Probable<int>{5, .4000f});
        REQUIRE(Probable<int>{5, .4f} != Probable<int>{5, .5f});
        REQUIRE(Probable<int>{5, .4f} != Probable<int>{4, .4f});
    }

    TEST_CASE("stats.poisson.returns properly shaped answers")
    {
        af::dim4 dims{2, 3, 5, 7};

        REQUIRE(poisson(0, af::constant(0, dims, s32)).dims() == dims);
    }

    TEST_CASE("stats.poisson.properly calculates probability")
    {
        auto testee{toVector<float>(poisson(2, af::range(af::dim4{5}, 0, s32)))};

        REQUIRE_THAT(
            testee,
            Catch::Matchers::RangeEquals(
                std::to_array({.1353f, .2707f, .2707f, .1804f, .0902f}),
                [](float l, float r)
                {
                    float d{std::abs(l - r)};
                    return d < 0.0001;
                }));
    }
}