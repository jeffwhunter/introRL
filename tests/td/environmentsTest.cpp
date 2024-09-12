#include <random>
#include <ranges>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <introRL/td/environments.hpp>
#include <introRL/td/types.hpp>

namespace irl::td
{
    TEST_CASE("td.environments.Walk.start.returns the middle")
    {
        Walk<StateCount{3}> three{};
        REQUIRE(three.start() == WalkState{2});

        Walk<StateCount{5}> five{};
        REQUIRE(five.start() == WalkState{3});

        Walk<StateCount{7}> seven{};
        REQUIRE(seven.start() == WalkState{4});
    }

    TEST_CASE("td.environments.Walk.step.goes up when true")
    {
        Walk<StateCount{3}> testee{};
        REQUIRE(testee.step(WalkState{2}, true) == WalkState{3});
        REQUIRE(testee.step(WalkState{2}, false) == WalkState{1});
    }

    TEST_CASE("td.environments.Walk.done.returns true on the borders")
    {
        Walk<StateCount{3}> testee{};
        REQUIRE(testee.done(WalkState{0}));
        REQUIRE(!testee.done(WalkState{1}));
        REQUIRE(!testee.done(WalkState{2}));
        REQUIRE(!testee.done(WalkState{3}));
        REQUIRE(testee.done(WalkState{4}));
    }

    TEST_CASE("td.environments.Walk.reward.only reward terminal states")
    {
        Walk<StateCount{3}> testee{};
        REQUIRE(testee.reward(WalkState{0}) == -1.f);
        REQUIRE(testee.reward(WalkState{1}) == .0f);
        REQUIRE(testee.reward(WalkState{2}) == .0f);
        REQUIRE(testee.reward(WalkState{3}) == .0f);
        REQUIRE(testee.reward(WalkState{4}) == 1.f);
    }

    TEST_CASE("td.environments.Windy.valid.does not return invalid actions")
    {
        GridActions actions{
            GridAction::make(0, -1),
            GridAction::make(1, -1),
            GridAction::make(1, 0),
            GridAction::make(1, 1),
            GridAction::make(0, 1),
            GridAction::make(-1, 1),
            GridAction::make(-1, 0),
            GridAction::make(-1, -1)};

        Windy<Width{3}, Height{3}> testee{GridState::make(), GridState::make(), {}};

        // top left
        REQUIRE_THAT(
            testee.valid(actions, GridState::make(0, 0)),
            Catch::Matchers::RangeEquals(
                std::set{
                    GridAction::make(1, 0),
                    GridAction::make(1, 1),
                    GridAction::make(0, 1)}));

        // top
        REQUIRE_THAT(
            testee.valid(actions, GridState::make(1, 0)),
            Catch::Matchers::RangeEquals(
                std::set{
                    GridAction::make(1, 0),
                    GridAction::make(1, 1),
                    GridAction::make(0, 1),
                    GridAction::make(-1, 1),
                    GridAction::make(-1, 0)}));

        // top right
        REQUIRE_THAT(
            testee.valid(actions, GridState::make(2, 0)),
            Catch::Matchers::RangeEquals(
                std::set{
                    GridAction::make(0, 1),
                    GridAction::make(-1, 1),
                    GridAction::make(-1, 0)}));

        // right
        REQUIRE_THAT(
            testee.valid(actions, GridState::make(2, 1)),
            Catch::Matchers::RangeEquals(
                std::set{
                    GridAction::make(0, 1),
                    GridAction::make(-1, 1),
                    GridAction::make(-1, 0),
                    GridAction::make(-1, -1),
                    GridAction::make(0, -1)}));

        // bottom right
        REQUIRE_THAT(
            testee.valid(actions, GridState::make(2, 2)),
            Catch::Matchers::RangeEquals(
                std::set{
                    GridAction::make(-1, 0),
                    GridAction::make(-1, -1),
                    GridAction::make(0, -1)}));

        // bottom
        REQUIRE_THAT(
            testee.valid(actions, GridState::make(1, 2)),
            Catch::Matchers::RangeEquals(
                std::set{
                    GridAction::make(-1, 0),
                    GridAction::make(-1, -1),
                    GridAction::make(0, -1),
                    GridAction::make(1, -1),
                    GridAction::make(1, 0)}));

        // bottom left
        REQUIRE_THAT(
            testee.valid(actions, GridState::make(0, 2)),
            Catch::Matchers::RangeEquals(
                std::set{
                    GridAction::make(0, -1),
                    GridAction::make(1, -1),
                    GridAction::make(1, 0)}));

        // left
        REQUIRE_THAT(
            testee.valid(actions, GridState::make(0, 1)),
            Catch::Matchers::RangeEquals(
                std::set{
                    GridAction::make(0, -1),
                    GridAction::make(1, -1),
                    GridAction::make(1, 0),
                    GridAction::make(1, 1),
                    GridAction::make(0, 1)}));
    }

    TEST_CASE("td.environments.Windy.step.obeys the wind")
    {
        constexpr Width W{3};
        constexpr Height H{2};

        Windy<W, H> down{GridState::make(), GridState::make(), {0, 1, 0}};
        Windy<W, H> up{GridState::make(), GridState::make(), {0, -1, 0}};

        REQUIRE(down.step(GridState::make(), GridAction::make(1, 0)) == GridState::make(1, 0));
        REQUIRE(down.step(GridState::make(1, 0), GridAction::make(1, 0)) == GridState::make(2, 1));
        REQUIRE(down.step(GridState::make(1, 1), GridAction::make(1, 0)) == GridState::make(2, 1));

        REQUIRE(up.step(GridState::make(0, 1), GridAction::make(1, 0)) == GridState::make(1, 1));
        REQUIRE(up.step(GridState::make(1, 1), GridAction::make(1, 0)) == GridState::make(2, 0));
        REQUIRE(up.step(GridState::make(1, 0), GridAction::make(1, 0)) == GridState::make(2, 0));
    }

    TEST_CASE("td.environments.Windy.start.returns the start")
    {
        auto start{GridState::make(2, 3)};

        REQUIRE(
            Windy<Width{0}, Height{0}>{start, GridState::make(), {}}.start() ==
            start);
    }

    TEST_CASE("td.environments.Windy.done.properly identifies the goal")
    {
        auto goal{GridState::make(2, 3)};
        Windy<Width{0}, Height{0}> testee{GridState::make(), goal, {}};

        REQUIRE(!testee.done(GridState::make(2, 0)));
        REQUIRE(!testee.done(GridState::make(0, 3)));
        REQUIRE(!testee.done(GridState::make(2, 2)));
        REQUIRE(!testee.done(GridState::make(2, 4)));
        REQUIRE(!testee.done(GridState::make(1, 3)));
        REQUIRE(!testee.done(GridState::make(3, 3)));
        REQUIRE(testee.done(goal));
    }

    TEST_CASE("td.environments.RandomWindy.step.windy columns are random")
    {
        constexpr Width W{2};
        constexpr Height H{3};

        std::mt19937 generator{0};

        RandomWindy<W, H> down{GridState::make(), GridState::make(), {1}, generator};
        RandomWindy<W, H> up{GridState::make(), GridState::make(), {-1}, generator};

        std::set<GridState> downResults{};
        std::set<GridState> upResults{};

        for (int i : std::views::iota(0, 100))
        {
            downResults.insert(down.step(GridState::make(0, 0), GridAction::make(1, 0)));
            upResults.insert(up.step(GridState::make(0, 2), GridAction::make(1, 0)));
        }

        REQUIRE_THAT(
            downResults,
            Catch::Matchers::UnorderedRangeEquals(
                std::set{
                    GridState::make(1, 0),
                    GridState::make(1, 1),
                    GridState::make(1, 2)}));

        REQUIRE_THAT(
            upResults,
            Catch::Matchers::UnorderedRangeEquals(
                std::set{
                    GridState::make(1, 0),
                    GridState::make(1, 1),
                    GridState::make(1, 2)}));
    }

    TEST_CASE("td.environments.RandomWindy.step.calm columns are not random")
    {
        constexpr Width W{2};
        constexpr Height H{3};

        std::mt19937 generator{0};

        RandomWindy<W, H> env{GridState::make(), GridState::make(), {0}, generator};

        std::set<GridState> results{};

        for (int i : std::views::iota(0, 100))
        {
            results.insert(env.step(GridState::make(0, 1), GridAction::make(1, 0)));
        }

        REQUIRE_THAT(
            results,
            Catch::Matchers::UnorderedRangeEquals(std::set{GridState::make(1, 1)}));
    }
}