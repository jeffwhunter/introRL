#include <arrayfire.h>
#include <catch2/catch_test_macros.hpp>

#include <introRL/environments.hpp>
#include <introRL/types.hpp>

namespace irl::bandit::environments
{
    TEST_CASE("bandit.environments.Stationary.reward has the proper shape")
    {
        constexpr unsigned nRuns{3};

        Stationary testee{ActionCount{10}, RunCount{nRuns}};

        REQUIRE(
            testee.reward(Actions{af::constant(0, nRuns)}).unwrap<Rewards>().dims() ==
            af::dim4{nRuns});
    }

    TEST_CASE("bandit.environments.Stationary.reward has the proper type")
    {
        constexpr unsigned nRuns{3};

        Stationary testee{ActionCount{10}, RunCount{nRuns}};

        REQUIRE(
            testee.reward(Actions{af::constant(0, nRuns)}).unwrap<Rewards>().type() ==
            f32);
    }

    TEST_CASE("bandit.environments.Stationary.optimal has the proper shape")
    {
        constexpr unsigned nRuns{3};

        Stationary testee{ActionCount{10}, RunCount{nRuns}};

        REQUIRE(testee.optimal().unwrap<Actions>().dims() == af::dim4{nRuns});
    }

    TEST_CASE("bandit.environments.Stationary.optimal has the proper type")
    {
        constexpr unsigned nRuns{3};

        Stationary testee{ActionCount{10}, RunCount{nRuns}};

        REQUIRE(testee.optimal().unwrap<Actions>().type() == u32);
    }

    TEST_CASE("bandit.environments.Walking.update changes optimal")
    {
        constexpr unsigned nRuns{5};
        constexpr float stepSize{10};

        Walking<stepSize> testee{ActionCount{10}, RunCount{nRuns}};

        const auto former{testee.optimal().unwrap<Actions>()};
        testee.update();
        const auto latter{testee.optimal().unwrap<Actions>()};

        REQUIRE(!af::allTrue<bool>(former == latter));
    }
}