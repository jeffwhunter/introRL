#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <introRL/td/environments.hpp>
#include <introRL/td/types.hpp>

namespace irl::td
{
    TEST_CASE("td.environments.Environment.valid.does not return invalid actions")
    {
        Actions actions{
            Action::make(0, -1),
            Action::make(1, -1),
            Action::make(1, 0),
            Action::make(1, 1),
            Action::make(0, 1),
            Action::make(-1, 1),
            Action::make(-1, 0),
            Action::make(-1, -1)};

        Environment<Width{3}, Height{3}> testee{State::make(), State::make(), {}};

        // top left
        REQUIRE_THAT(
            testee.valid(actions, State::make(0, 0)),
            Catch::Matchers::RangeEquals(
                std::set{
                    Action::make(1, 0),
                    Action::make(1, 1),
                    Action::make(0, 1)}));

        // top
        REQUIRE_THAT(
            testee.valid(actions, State::make(1, 0)),
            Catch::Matchers::RangeEquals(
                std::set{
                    Action::make(1, 0),
                    Action::make(1, 1),
                    Action::make(0, 1),
                    Action::make(-1, 1),
                    Action::make(-1, 0)}));

        // top right
        REQUIRE_THAT(
            testee.valid(actions, State::make(2, 0)),
            Catch::Matchers::RangeEquals(
                std::set{
                    Action::make(0, 1),
                    Action::make(-1, 1),
                    Action::make(-1, 0)}));

        // right
        REQUIRE_THAT(
            testee.valid(actions, State::make(2, 1)),
            Catch::Matchers::RangeEquals(
                std::set{
                    Action::make(0, 1),
                    Action::make(-1, 1),
                    Action::make(-1, 0),
                    Action::make(-1, -1),
                    Action::make(0, -1)}));

        // bottom right
        REQUIRE_THAT(
            testee.valid(actions, State::make(2, 2)),
            Catch::Matchers::RangeEquals(
                std::set{
                    Action::make(-1, 0),
                    Action::make(-1, -1),
                    Action::make(0, -1)}));

        // bottom
        REQUIRE_THAT(
            testee.valid(actions, State::make(1, 2)),
            Catch::Matchers::RangeEquals(
                std::set{
                    Action::make(-1, 0),
                    Action::make(-1, -1),
                    Action::make(0, -1),
                    Action::make(1, -1),
                    Action::make(1, 0)}));

        // bottom left
        REQUIRE_THAT(
            testee.valid(actions, State::make(0, 2)),
            Catch::Matchers::RangeEquals(
                std::set{
                    Action::make(0, -1),
                    Action::make(1, -1),
                    Action::make(1, 0)}));

        // left
        REQUIRE_THAT(
            testee.valid(actions, State::make(0, 1)),
            Catch::Matchers::RangeEquals(
                std::set{
                    Action::make(0, -1),
                    Action::make(1, -1),
                    Action::make(1, 0),
                    Action::make(1, 1),
                    Action::make(0, 1)}));
    }

    TEST_CASE("td.environments.Environment.step.obeys the wind")
    {
        constexpr Width W{3};
        constexpr Height H{2};

        Environment<W, H> down{State::make(), State::make(), {0, 1, 0}};
        Environment<W, H> up{State::make(), State::make(), {0, -1, 0}};

        REQUIRE(down.step(State::make(), Action::make(1, 0)) == State::make(1, 0));
        REQUIRE(down.step(State::make(1, 0), Action::make(1, 0)) == State::make(2, 1));
        REQUIRE(down.step(State::make(1, 1), Action::make(1, 0)) == State::make(2, 1));

        REQUIRE(up.step(State::make(0, 1), Action::make(1, 0)) == State::make(1, 1));
        REQUIRE(up.step(State::make(1, 1), Action::make(1, 0)) == State::make(2, 0));
        REQUIRE(up.step(State::make(1, 0), Action::make(1, 0)) == State::make(2, 0));
    }

    TEST_CASE("td.environments.Environment.start.returns the start")
    {
        auto start{State::make(2, 3)};

        REQUIRE(
            Environment<Width{0}, Height{0}>{start, State::make(), {}}.start() ==
            start);
    }

    TEST_CASE("td.environments.Environment.done.properly identifies the goal")
    {
        auto goal{State::make(2, 3)};
        Environment<Width{0}, Height{0}> testee{State::make(), goal, {}};

        REQUIRE(!testee.done(State::make(2, 0)));
        REQUIRE(!testee.done(State::make(0, 3)));
        REQUIRE(!testee.done(State::make(2, 2)));
        REQUIRE(!testee.done(State::make(2, 4)));
        REQUIRE(!testee.done(State::make(1, 3)));
        REQUIRE(!testee.done(State::make(3, 3)));
        REQUIRE(testee.done(goal));
    }
}