#include <ranges>

#include <catch2/catch_test_macros.hpp>

#include <introRL/ticker.hpp>
#include <introRL/types.hpp>

namespace irl
{
    TEST_CASE("ticker.Ticker.call.ticks at some rate")
    {
        constexpr unsigned rate{10};

        size_t calls{};
        Ticker<rate> testee{[&] { ++calls; }};

        for (size_t _ : std::views::iota(0u, rate))
        {
            testee();
        }

        REQUIRE(calls == 1);
    }
}