#include <iterator>
#include <map>

#include <catch2/catch_test_macros.hpp>
#include <catch2/trompeloeil.hpp>

#include <introRL/monte/algorithm.hpp>
#include <introRL/monte/types.hpp>
#include <introRL/sparse.hpp>
#include <introRL/stats.hpp>

namespace irl::monte
{
    class MockAgent
    {
    public:
        MAKE_MOCK2(act, Probable<Action>(State, float));
    };

    class MockEnvironment
    {
    public:
        MAKE_MOCK0(reset, State());
        MAKE_MOCK2(step, State(State, Action));
        MAKE_MOCK1(done, bool(State));
        MAKE_MOCK0(starts, const std::set<Position>& ());
    };

    using trompeloeil::eq;

    TEST_CASE("monte.algorithm.generateEpisode.includes final position when short")
    {
        MockAgent agent{};
        ALLOW_CALL(agent, act(ANY(State), ANY(float)))
            .RETURN(Probable<Action>{Action::make(), 0.f});

        MockEnvironment environment{};
        ALLOW_CALL(environment, done(eq(State{})))
            .RETURN(false);
        ALLOW_CALL(environment, done(eq(State{.position{Position::make(0, 1)}})))
            .RETURN(true);
        ALLOW_CALL(environment, step(ANY(State), ANY(Action)))
            .RETURN(State{.position{Position::make(0, 1)}});

        REQUIRE(
            std::ranges::distance(
                generateEpisode<StepCount{5}>(agent, environment, State{}, 0.f)
                    .getAllPositions()) ==
            2);
    }

    TEST_CASE(
        "monte.algorithm.generateEpisode.does not include final position when long")
    {
        MockAgent agent{};
        ALLOW_CALL(agent, act(ANY(State), ANY(float)))
            .RETURN(Probable<Action>{Action::make(), 0.f});

        MockEnvironment environment{};
        ALLOW_CALL(environment, done(eq(State{})))
            .RETURN(false);
        ALLOW_CALL(environment, done(eq(State{.position{Position::make(0, 1)}})))
            .RETURN(true);
        ALLOW_CALL(environment, step(ANY(State), ANY(Action)))
            .RETURN(State{.position{Position::make(0, 1)}});

        REQUIRE(
            std::ranges::distance(
                generateEpisode<StepCount{1}>(agent, environment, State{}, 0.f)
                    .getAllPositions()) ==
            1);
    }

    TEST_CASE("monte.algorithm.demoAllStarts.makes one demo per start")
    {
        std::set<Position> starts{
            Position::make(),
            Position::make(0, 1),
            Position::make(1, 0),
            Position::make(1, 1)};

        MockAgent agent{};
        ALLOW_CALL(agent, act(ANY(State), ANY(float)))
            .RETURN(Probable<Action>{Action::make(), 0.f});

        MockEnvironment environment{};
        ALLOW_CALL(environment, done(ANY(State))).RETURN(true);
        ALLOW_CALL(environment, step(ANY(State), ANY(Action))).RETURN(State{});
        ALLOW_CALL(environment, starts()).RETURN(starts);

        REQUIRE(demoAllStarts<StepCount{1}>(agent, environment).size() == starts.size());
    }

    TEST_CASE("monte.algorithm.rewind.collects information on encountered experiences")
    {
        State state{.position{Position::make(0, 1)}, .velocity{Velocity::make(2, 3)}};
        auto action{Action::make(4, 5)};

        SparseMatrix<State, Action, float> c{};
        SparseMatrix<State, Action, float> q{};
        std::map<State, Action> pi{};

        Episode episode{};
        episode.append(state, Probable<Action>{action, 1.f});

        rewind<.5f>(c, q, pi, episode);

        REQUIRE(c(state, action) == 1.f);
        REQUIRE(q(state, action) == 1.f);
        REQUIRE(pi[state] == action);
    }

    TEST_CASE("monte.algorithm.control.returns an agent that mimics it's teacher")
    {
        State start{.position{Position::make(0, 1)}, .velocity{Velocity::make(2, 3)}};

        State state{.position{Position::make()}};
        auto action{Action::make(4, 5)};

        MockAgent agent{};
        ALLOW_CALL(agent, act(ANY(State), ANY(float)))
            .RETURN(Probable<Action>{action, 0.f});

        MockEnvironment environment{};
        ALLOW_CALL(environment, done(eq(start))).RETURN(false);
        ALLOW_CALL(environment, done(eq(state))).RETURN(true);
        ALLOW_CALL(environment, step(ANY(State), ANY(Action))).RETURN(state);
        ALLOW_CALL(environment, reset()).RETURN(start);

        auto explorer{Explorer::make(0, 0, 0)};

        REQUIRE(
            control<EpisodeCount{1}, StepCount{5}, 0.f>(agent, environment, explorer)
                .act(start, 0.f).value ==
            action);
    }
}