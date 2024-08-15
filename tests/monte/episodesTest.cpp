#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <introRL/monte/episodes.hpp>
#include <introRL/stats.hpp>

namespace irl::monte
{
    TEST_CASE("monte.episodes.Episode.stores steps")
    {
        constexpr State stateTarget{
            .position{Position::make(0, 1)},
            .velocity{Velocity::make(2, 3)}};

        constexpr Probable<Action> actionTarget{
            .value{Action::make(4, 5)},
            .probability{.6f}};

        Episode testee{};

        testee.append(stateTarget, actionTarget);

        const auto& step{testee.getStep(0)};

        REQUIRE(step.state == stateTarget);
        REQUIRE(step.action == actionTarget);
    }

    TEST_CASE("monte.episodes.Episode.returns bigT")
    {
        Episode testee{};
        testee.append(State{}, Probable<Action>{});
        testee.append(State{}, Probable<Action>{});
        testee.setFinalPosition(Position::make());

        REQUIRE(testee.bigT() == 2U);
    }

    TEST_CASE("monte.episodes.Episode.all positions includes final position")
    {
        std::array targetPositions{
            Position::make(0, 1),
            Position::make(2, 3),
            Position::make(4, 5)};

        Episode testee{};
        testee.append(State{.position{targetPositions[0]}}, Probable<Action>{});
        testee.append(State{.position{targetPositions[1]}}, Probable<Action>{});
        testee.setFinalPosition(targetPositions[2]);

        REQUIRE_THAT(
            testee.getAllPositions(),
            Catch::Matchers::RangeEquals(targetPositions));
    }
}