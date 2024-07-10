#include <arrayfire.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <introRL/afUtils.hpp>
#include <introRL/reduce.hpp>

namespace irl::reduce
{
    TEST_CASE("reduce.argMax.returns indices of the maximum")
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
}