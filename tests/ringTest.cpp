#include <array>
#include <ranges>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <introRL/ring.hpp>

namespace irl
{
    TEST_CASE("Ring.operator[].wraps around")
    {
        Ring<int, 3> testee{};

        for (int index : std::views::iota(0, 5))
        {
            testee[index] = index;
        }

        REQUIRE_THAT(testee, Catch::Matchers::RangeEquals(std::to_array({3, 4, 2})));
    }
}