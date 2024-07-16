#include <array>

#include <arrayfire.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <catch2/trompeloeil.hpp>

#include <introRL/results.hpp>

namespace irl::bandit::results
{
    TEST_CASE("bandit.results.RewardsAndOptimality.emits proper rewards")
    {
        af::array keys{0u, 0u, 1u, 1u, 2u, 2u, 3u, 3u};

        RewardsAndOptimality testee{ParameterCount{4}, ReductionKeys{keys}};

        testee.update(
            LinearActions{af::array{0u, 0u, 0u, 1u, 1u, 0u, 1u, 1u}},
            LinearActions{af::constant(1u, keys.dims(0))},
            Rewards{af::array{0.f, 0.f, 0.f, 2.f, 3.f, 0.f, 4.f, 5.f}});

        testee.update(
            LinearActions{af::constant(1u, keys.dims(0))},
            LinearActions{af::constant(1u, keys.dims(0))},
            Rewards{af::array{0.f, 0.f, -2.f, 0.f, 0.f, -30.f, -9.f, 9.f}});

        auto&& [rewards, optimality]{testee.value()};

        REQUIRE_THAT(
            rewards[0],
            Catch::Matchers::RangeEquals(std::to_array({0.f, 0.f})));

        REQUIRE_THAT(
            rewards[1],
            Catch::Matchers::RangeEquals(std::to_array({1.f, -1.f})));

        REQUIRE_THAT(
            rewards[2],
            Catch::Matchers::RangeEquals(std::to_array({1.5f, -15.f})));

        REQUIRE_THAT(
            rewards[3],
            Catch::Matchers::RangeEquals(std::to_array({4.5f, 0.f})));

        REQUIRE_THAT(
            optimality[0],
            Catch::Matchers::RangeEquals(std::to_array({0.f, 1.f})));

        REQUIRE_THAT(
            optimality[1],
            Catch::Matchers::RangeEquals(std::to_array({.5f, 1.f})));

        REQUIRE_THAT(
            optimality[2],
            Catch::Matchers::RangeEquals(std::to_array({.5f, 1.f})));

        REQUIRE_THAT(
            optimality[3],
            Catch::Matchers::RangeEquals(std::to_array({1.f, 1.f})));
    }

    TEST_CASE("bandit.results.RollingRewards.emits proper rewards")
    {
        af::array keys{0u, 0u, 1u, 1u, 2u, 2u, 3u, 3u};

        RollingRewards<0u> testee{ParameterCount{4}, ReductionKeys{keys}};

        testee.update(
            LinearActions{af::array{0u}},
            LinearActions{af::array{0u}},
            Rewards{af::array{0.f, 0.f, 0.f, 2.f, 3.f, 0.f, 4.f, 5.f}});

        testee.update(
            LinearActions{af::array{0u}},
            LinearActions{af::array{0u}},
            Rewards{af::array{0.f, 0.f, -2.f, 0.f, 0.f, -30.f, -9.f, 9.f}});

        REQUIRE_THAT(
            testee.value(),
            Catch::Matchers::RangeEquals(std::to_array({0.f, 0.f, -6.75f, 2.25f})));
    }

    TEST_CASE("bandit.results.RollingRewards.delays recording until start step")
    {
        af::array keys{0u, 0u, 1u, 1u, 2u, 2u, 3u, 3u};

        RollingRewards<1u> testee{ParameterCount{4}, ReductionKeys{keys}};

        testee.update(
            LinearActions{af::array{0u}},
            LinearActions{af::array{0u}},
            Rewards{af::array{0.f, 0.f, 0.f, 2.f, 3.f, 0.f, 4.f, 5.f}});

        testee.update(
            LinearActions{af::array{0u}},
            LinearActions{af::array{0u}},
            Rewards{af::array{0.f, 0.f, -2.f, 0.f, 0.f, -30.f, -9.f, 9.f}});

        REQUIRE_THAT(
            testee.value(),
            Catch::Matchers::RangeEquals(std::to_array({0.f, -1.f, -15.f, 0.f})));
    }
}