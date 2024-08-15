#include <algorithm>
#include <ranges>

#include <catch2/catch_test_macros.hpp>

#include <introRL/monte/agents.hpp>

namespace irl::monte
{
    TEST_CASE("monte.agents.Explorer.controls exploration")
    {
        auto testee{Explorer::make(0, 0, 0)};

        REQUIRE(!testee.should_explore(0.f));
        REQUIRE(testee.should_explore(1.f));
    }

    TEST_CASE("monte.agents.Explorer.samples action in range")
    {
        int minAction{-2};
        int maxAction{8};
        size_t samples{10};

        auto testee{Explorer::make(minAction, maxAction, 0)};

        for (auto _ : std::views::iota(0U, samples))
        {
            auto action{testee.explore()};

            bool success{
                std::ranges::all_of(
                    action,
                    [=](auto e)
                    {
                        return minAction <= e && e <= maxAction;
                    })};

            REQUIRE(success);
        }
    }

    TEST_CASE("monte.agents.ExpertAgent.goes forward before turn")
    {
        auto explorer{Explorer::make(0, 0, 0)};

        ExpertAgent testee{1, 10, 10, explorer};

        REQUIRE(
            testee.act(State{.position{Position::make(15, 0)}}, 0.f).value ==
            Action::make(-1, 0));
    }

    TEST_CASE("monte.agents.ExpertAgent.damps horizontal velocity before turn")
    {
        auto explorer{Explorer::make(0, 0, 0)};

        ExpertAgent testee{1, 10, 10, explorer};

        REQUIRE(
            testee.act(
                State{
                    .position{Position::make(15, 0)},
                    .velocity{Velocity::make(0, 1)}},
                0.f
            ).value ==
            Action::make(-1, -1));

        REQUIRE(
            testee.act(
                State{
                    .position{Position::make(15, 0)},
                    .velocity{Velocity::make(0, -1)}},
                    0.f
                    ).value ==
            Action::make(-1, 1));
    }

    TEST_CASE("monte.agents.ExpertAgent.turns")
    {
        auto explorer{Explorer::make(0, 0, 0)};

        ExpertAgent testee{1, 10, 10, explorer};

        REQUIRE(
            testee.act(State{.position{Position::make(5, 0)}}, 0.f).value ==
            Action::make(0, 1));
    }

    TEST_CASE("monte.agents.ExpertAgent.damps vertical velocity after turn")
    {
        auto explorer{Explorer::make(0, 0, 0)};

        ExpertAgent testee{1, 10, 10, explorer};

        REQUIRE(
            testee.act(
                State{.position{Position::make(5, 0)}, .velocity{Velocity::make(1, 0)}},
                0.f
            ).value ==
            Action::make(-1, 1));

        REQUIRE(
            testee.act(
                State{.position{Position::make(5, 0)}, .velocity{Velocity::make(-1, 0)}},
                0.f
            ).value ==
            Action::make(1, 1));
    }

    TEST_CASE("monte.agents.TableAgent.acts according to a table")
    {
        auto explorer{Explorer::make(0, 0, 0)};

        TableAgent testee{
            {
                {State{.position{Position::make(1, 0)}}, Action::make(1, 0)},
                {State{.position{Position::make(0, 1)}}, Action::make(-1, 0)},
                {State{.velocity{Velocity::make(1, 0)}}, Action::make(0, 1)},
                {State{.velocity{Velocity::make(0, 1)}}, Action::make(0, -1)}
            },
            explorer};

        REQUIRE(
            testee.act(State{.position{Position::make(1, 0)}}, 0.f).value ==
            Action::make(1, 0));

        REQUIRE(
            testee.act(State{.position{Position::make(0, 1)}}, 0.f).value ==
            Action::make(-1, 0));

        REQUIRE(
            testee.act(State{.velocity{Velocity::make(1, 0)}}, 0.f).value ==
            Action::make(0, 1));

        REQUIRE(
            testee.act(State{.velocity{Velocity::make(0, 1)}}, 0.f).value ==
            Action::make(0, -1));
    }
}