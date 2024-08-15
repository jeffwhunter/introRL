#include <vector>

#include <arrayfire.h>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <catch2/catch_test_macros.hpp>

#include <introRL/afUtils.hpp>
#include <introRL/linear.hpp>

namespace irl
{
    TEST_CASE("linear.linearIndex.linearizes indices")
    {
        REQUIRE_THAT(
            toVector<unsigned>(linearIndex(af::array{0u, 2u, 4u})),
            Catch::Matchers::RangeEquals(std::vector<unsigned>{0, 7, 14}));
    }
}