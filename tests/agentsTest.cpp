#include <ranges>

#include <arrayfire.h>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <catch2/catch_test_macros.hpp>

#include <introRL/agents.hpp>
#include <introRL/linear.hpp>
#include <introRL/types.hpp>

namespace irl::bandit::agents
{
    TEST_CASE("bandit.agents.EpsilonGreedyAverage.act has the correct shape")
    {
        constexpr unsigned nRuns{3};
        constexpr unsigned maxActions{10};

        for (auto nActions : std::views::iota(1u) | std::views::take(maxActions))
        {
            EpsilonGreedyAverage testee{
                DeviceParameters{af::constant(0, nRuns)},
                ActionCount{nActions}};

            REQUIRE(testee.act().unwrap<Actions>().dims() == af::dim4{nRuns});
        }
    }

    TEST_CASE("bandit.agents.EpsilonGreedyAverage.picks max when greedy")
    {
        constexpr unsigned nActions{5};

        const af::array actionIndices{0u, 2u, 4u};
        const dim_t nRuns{actionIndices.elements()};

        EpsilonGreedyAverage testee{
            DeviceParameters{af::constant(0, nRuns)},
            ActionCount{nActions}};

        testee.update(Actions{actionIndices}, Rewards{af::constant(1, nRuns)});

        REQUIRE(
            af::allTrue<bool>(
                testee.act().unwrap<Actions>() ==
                linear::index(actionIndices)));
    }

    TEST_CASE("bandit.agents.EpsilonGreedy.act has the correct shape")
    {
        constexpr unsigned nRuns{3};
        constexpr unsigned maxActions{10};
        constexpr float stepSize{.1};

        for (auto nActions : std::views::iota(1u) | std::views::take(maxActions))
        {
            EpsilonGreedy<stepSize> testee{
                DeviceParameters{af::constant(0, nRuns)},
                ActionCount{nActions}};

            REQUIRE(testee.act().unwrap<Actions>().dims() == af::dim4{nRuns});
        }
    }

    TEST_CASE("bandit.agents.EpsilonGreedy.picks max when greedy")
    {
        constexpr unsigned nActions{5};
        constexpr float stepSize{.1};

        const af::array actionIndices{0u, 1u, 2u, 3u, 4u};
        const dim_t nRuns{actionIndices.elements()};

        EpsilonGreedy<stepSize> testee{
            DeviceParameters{af::constant(0, nRuns)},
            ActionCount{nActions}};

        testee.update(Actions{actionIndices}, Rewards{af::constant(1, nRuns)});

        REQUIRE(
            af::allTrue<bool>(
                testee.act().unwrap<Actions>() ==
                linear::index(actionIndices)));
    }

    TEST_CASE("bandit.agents.EpsilonGreedy.picks randomly with zero step size")
    {
        constexpr unsigned nActions{5};
        constexpr float stepSize{0.};

        const af::array actionIndices{0u, 1u, 2u, 3u, 4u};
        const dim_t nRuns{actionIndices.elements()};

        EpsilonGreedy<stepSize> testee{
            DeviceParameters{af::constant(0, nRuns)},
            ActionCount{nActions}};

        testee.update(Actions{actionIndices}, Rewards{af::constant(1, nRuns)});

        REQUIRE(
            !af::allTrue<bool>(
                testee.act().unwrap<Actions>() ==
                linear::index(actionIndices)));
    }

    TEST_CASE("bandit.agents.Optimistic.act has the correct shape")
    {
        constexpr unsigned nRuns{3};
        constexpr unsigned maxActions{10};
        constexpr float stepSize{.1};

        for (auto nActions : std::views::iota(1u) | std::views::take(maxActions))
        {
            Optimistic<stepSize> testee{
                DeviceParameters{af::constant(0, nRuns)},
                ActionCount{nActions}};

            REQUIRE(testee.act().unwrap<Actions>().dims() == af::dim4{nRuns});
        }
    }

    TEST_CASE("bandit.agents.Optimistic.picks max")
    {
        constexpr unsigned nActions{5};
        constexpr float stepSize{.1};

        const af::array actionIndices{0u, 1u, 2u, 3u, 4u};
        const dim_t nRuns{actionIndices.elements()};

        Optimistic<stepSize> testee{
            DeviceParameters{af::constant(0, nRuns)},
            ActionCount{nActions}};

        testee.update(Actions{actionIndices}, Rewards{af::constant(1, nRuns)});

        REQUIRE(
            af::allTrue<bool>(
                testee.act().unwrap<Actions>() ==
                linear::index(actionIndices)));
    }

    TEST_CASE("bandit.agents.Optimistic.picks randomly with zero step size")
    {
        constexpr unsigned nActions{5};
        constexpr float stepSize{0.};

        const af::array actionIndices{0u, 1u, 2u, 3u, 4u};
        const dim_t nRuns{actionIndices.elements()};

        Optimistic<stepSize> testee{
            DeviceParameters{af::constant(0, nRuns)},
            ActionCount{nActions}};

        testee.update(Actions{actionIndices}, Rewards{af::constant(1, nRuns)});

        REQUIRE(
            !af::allTrue<bool>(
                testee.act().unwrap<Actions>() ==
                linear::index(actionIndices)));
    }

    TEST_CASE("bandit.agents.Optimistic.picks randomly with optimistic initialization")
    {
        constexpr unsigned nActions{5};
        constexpr float stepSize{.1};
        constexpr float optimism{10.};

        const af::array actionIndices{0u, 1u, 2u, 3u, 4u};
        const dim_t nRuns{actionIndices.elements()};

        Optimistic<stepSize> testee{
            DeviceParameters{af::constant(optimism, nRuns)},
            ActionCount{nActions}};

        testee.update(Actions{actionIndices}, Rewards{af::constant(1, nRuns)});

        REQUIRE(
            !af::allTrue<bool>(
                testee.act().unwrap<Actions>() ==
                linear::index(actionIndices)));
    }

    TEST_CASE("bandit.agents.UpperConfidence.act has the correct shape")
    {
        constexpr unsigned nRuns{3};
        constexpr unsigned maxActions{10};

        for (auto nActions : std::views::iota(1u) | std::views::take(maxActions))
        {
            UpperConfidence testee{
                DeviceParameters{af::constant(0, nRuns)},
                ActionCount{nActions}};

            REQUIRE(testee.act().unwrap<Actions>().dims() == af::dim4{nRuns});
        }
    }

    TEST_CASE("bandit.agents.UpperConfidence.picks max")
    {
        constexpr unsigned nActions{5};

        const af::array actionIndices{0u, 1u, 2u, 3u, 4u};
        const dim_t nRuns{actionIndices.elements()};

        UpperConfidence testee{
            DeviceParameters{af::constant(0, nRuns)},
            ActionCount{nActions}};

        testee.update(Actions{actionIndices}, Rewards{af::constant(1, nRuns)});

        REQUIRE(
            af::allTrue<bool>(
                testee.act().unwrap<Actions>() ==
                linear::index(actionIndices)));
    }

    TEST_CASE("bandit.agents.UpperConfidence.picks randomly with non-zero cee")
    {
        constexpr unsigned nActions{5};
        constexpr float cee{1.};

        const af::array actionIndices{0u, 1u, 2u, 3u, 4u};
        const dim_t nRuns{actionIndices.elements()};

        UpperConfidence testee{
            DeviceParameters{af::constant(cee, nRuns)},
            ActionCount{nActions}};

        testee.update(Actions{actionIndices}, Rewards{af::constant(1, nRuns)});

        REQUIRE(
            !af::allTrue<bool>(
                testee.act().unwrap<Actions>() ==
                linear::index(actionIndices)));
    }

    TEST_CASE("bandit.agents.GradientBaseline.act has the correct shape")
    {
        constexpr unsigned nRuns{3};
        constexpr unsigned maxActions{10};

        for (auto nActions : std::views::iota(1u) | std::views::take(maxActions))
        {
            GradientBaseline testee{
                DeviceParameters{af::constant(0, nRuns)},
                ActionCount{nActions}};

            REQUIRE(testee.act().unwrap<Actions>().dims() == af::dim4{nRuns});
        }
    }

    TEST_CASE("bandit.agents.GradientBaseline.picks randomly on the first step")
    {
        constexpr unsigned nActions{5};
        constexpr float alpha{10.};
        constexpr float reward{10.};

        const af::array actionIndices{0u, 1u, 2u, 3u, 4u};
        const dim_t nRuns{actionIndices.elements()};

        GradientBaseline testee{
            DeviceParameters{af::constant(alpha, nRuns)},
            ActionCount{nActions}};

        testee.update(Actions{actionIndices}, Rewards{af::constant(reward, nRuns)});

        REQUIRE(
            !af::allTrue<bool>(
                testee.act().unwrap<Actions>() ==
                linear::index(actionIndices)));
    }

    TEST_CASE("bandit.agents.GradientBaseline.picks randomly with unchanging rewards")
    {
        constexpr unsigned nActions{5};
        constexpr float alpha{10.};
        constexpr float reward{10.};

        const af::array actionIndices{0u, 1u, 2u, 3u, 4u};
        const dim_t nRuns{actionIndices.elements()};

        GradientBaseline testee{
            DeviceParameters{af::constant(alpha, nRuns)},
            ActionCount{nActions}};

        const Actions actions{actionIndices};
        const Rewards rewards{af::constant(reward, nRuns)};

        testee.update(actions, rewards);
        testee.update(actions, rewards);

        REQUIRE(
            !af::allTrue<bool>(
                testee.act().unwrap<Actions>() ==
                linear::index(actionIndices)));
    }

    TEST_CASE("bandit.agents.GradientBaseline.picks max")
    {
        constexpr unsigned nActions{5};
        constexpr float alpha{10.};
        constexpr float reward{10.};

        const af::array actionIndices{0u, 1u, 2u, 3u, 4u};
        const dim_t nRuns{actionIndices.elements()};

        GradientBaseline testee{
            DeviceParameters{af::constant(alpha, nRuns)},
            ActionCount{nActions}};

        const Actions actions{actionIndices};

        testee.update(actions, Rewards{af::constant(0, nRuns)});
        testee.update(actions, Rewards{af::constant(reward, nRuns)});

        REQUIRE(
            af::allTrue<bool>(
                testee.act().unwrap<Actions>() ==
                linear::index(actionIndices)));
    }
}