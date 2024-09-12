#include <random>
#include <ranges>

#include <catch2/catch_test_macros.hpp>

#include <introRL/td/agents.hpp>
#include <introRL/td/types.hpp>

namespace irl::td
{
    TEST_CASE("td.agents.TableAgent.act.can be greedy")
    {
        const Epsilon E{0};
        std::mt19937 generator{0};

        Q q{};

        q(GridState::make(0, 1), GridAction::make(-1, 1)) = 1.f;
        q(GridState::make(0, 1), GridAction::make(1, -1)) = 2.f;
        q(GridState::make(1, 0), GridAction::make(0, -1)) = 2.f;
        q(GridState::make(1, 0), GridAction::make(-1, 0)) = 1.f;

        EGreedy testee{E, generator};

        REQUIRE(
            testee.act(
                q,
                GridState::make(0, 1),
                {GridAction::make(-1, 1), GridAction::make(1, -1)}) ==
            GridAction::make(1, -1));

        REQUIRE(
            testee.act(
                q,
                GridState::make(1, 0),
                {GridAction::make(0, -1), GridAction::make(-1, 0)}) ==
            GridAction::make(0, -1));
    }

    TEST_CASE("td.agents.TableAgent.act.only picks from given actions")
    {
        const Epsilon E{0};
        std::mt19937 generator{0};

        Q q{};

        q(GridState::make(0, 1), GridAction::make(-1, 1)) = 1.f;
        q(GridState::make(0, 1), GridAction::make(1, -1)) = 2.f;
        q(GridState::make(1, 0), GridAction::make(0, -1)) = 2.f;
        q(GridState::make(1, 0), GridAction::make(-1, 0)) = 1.f;

        EGreedy testee{E, generator};

        REQUIRE(
            testee.act(q, GridState::make(0, 1), {GridAction::make(-1, 1)}) ==
            GridAction::make(-1, 1));

        REQUIRE(
            testee.act(q, GridState::make(1, 0), {GridAction::make(-1, 0)}) ==
            GridAction::make(-1, 0));
    }
}