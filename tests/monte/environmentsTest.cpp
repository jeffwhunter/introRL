#include <random>
#include <set>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <introRL/monte/environments.hpp>

namespace irl::monte
{
    TEST_CASE("monte.environments.Environment.step moves through clear space")
    {
        using TEnv = Environment<1, 3>;
        std::mt19937 generator{0};

        constexpr auto track{std::to_array({S, _, _})};

        auto testee{TEnv::make(TEnv::Track{track.data()}, generator)};

        REQUIRE(
            testee.step(testee.reset(), Action::make(0, 2)).position ==
            Position::make(0, 2));
    }

    TEST_CASE("monte.environments.Environment.step stops at a finish line")
    {
        using TEnv = Environment<1, 3>;
        std::mt19937 generator{0};

        constexpr auto track{std::to_array({S, F, _})};

        auto testee{TEnv::make(TEnv::Track{track.data()}, generator)};

        REQUIRE(
            testee.step(testee.reset(), Action::make(0, 2)).position ==
            Position::make(0, 1));
    }

    TEST_CASE("monte.environments.Environment.step does not move through walls")
    {
        using TEnv = Environment<1, 3>;
        std::mt19937 generator{0};

        constexpr auto track{std::to_array({S, X, _})};

        auto testee{TEnv::make(TEnv::Track{track.data()}, generator)};

        REQUIRE(
            testee.step(testee.reset(), Action::make(0, 2)).position ==
            Position::make());
    }

    TEST_CASE("monte.environments.Environment.reset returns a starting space")
    {
        using TEnv = Environment<1, 3>;
        std::mt19937 generator{0};

        constexpr auto track{std::to_array({_, S, _})};

        auto testee{TEnv::make(TEnv::Track{track.data()}, generator)};

        REQUIRE(testee.reset().position == Position::make(0, 1));
    }

    TEST_CASE("monte.environments.Environment.reset returns motionless state")
    {
        using TEnv = Environment<1, 3>;
        std::mt19937 generator{0};

        constexpr auto track{std::to_array({_, S, _})};

        auto testee{TEnv::make(TEnv::Track{track.data()}, generator)};

        REQUIRE(testee.reset().velocity == Velocity::make());
    }

    TEST_CASE("monte.environments.Environment.done is only true on the finish line")
    {
        using TEnv = Environment<1, 4>;
        std::mt19937 generator{0};

        constexpr auto track{std::to_array({S, _, X, F})};

        auto testee{TEnv::make(TEnv::Track{track.data()}, generator)};

        REQUIRE(!testee.done(State{.position{Position::make(0, 0)}}));
        REQUIRE(!testee.done(State{.position{Position::make(0, 1)}}));
        REQUIRE(!testee.done(State{.position{Position::make(0, 2)}}));
        REQUIRE(testee.done(State{.position{Position::make(0, 3)}}));
    }

    TEST_CASE("monte.environments.Environment.starts returns all the starts")
    {
        using TEnv = Environment<1, 3>;
        std::mt19937 generator{0};

        constexpr auto track{std::to_array({S, _, S})};

        auto testee{TEnv::make(TEnv::Track{track.data()}, generator)};

        REQUIRE_THAT(
            testee.starts(),
            Catch::Matchers::RangeEquals(
                std::set<Position>{
                    Position::make(0, 0),
                    Position::make(0, 2)}));
    }
}