#include <random>
#include <set>

#include <catch2/catch_test_macros.hpp>
#include <catch2/trompeloeil.hpp>

#include <introRL/math/sparse.hpp>
#include <introRL/td/algorithm.hpp>
#include <introRL/td/types.hpp>
#include <introRL/types.hpp>

namespace irl::td
{
    class MockEnvironment
    {
    public:
        MAKE_MOCK0(start, State());
        MAKE_MOCK0(goal, State());
        MAKE_MOCK2(valid, Actions(Actions, State));
        MAKE_MOCK2(step, State(State, Action));
        MAKE_MOCK1(done, bool(State));
        MAKE_MOCK1(wind, int(size_t));
    };

    class MockAgent
    {
    public:
        MAKE_MOCK3(act, Action(const Q&, const State&, const Actions&));
    };

    TEST_CASE("td.algorithm.sarsa.compiles action value estimates")
    {
        constexpr StepCount N_STEPS{1};
        const Alpha A{.5};

        auto start{State::make()};
        auto goal{State::make()};
        Actions actions{Action::make(0, 1), Action::make(1, 0)};
        std::set<Action> valid{Action::make(0, 1), Action::make(1, 0)};
        auto action{Action::make()};
        auto stepResult{State::make(1, 1)};

        MockEnvironment environment{};
        ALLOW_CALL(environment, start()).RETURN(start);
        ALLOW_CALL(environment, start()).RETURN(goal);
        ALLOW_CALL(environment, valid(ANY(Actions), ANY(State))).RETURN(valid);
        ALLOW_CALL(environment, step(ANY(State), ANY(Action))).RETURN(stepResult);
        ALLOW_CALL(environment, done(ANY(State))).RETURN(false);
        ALLOW_CALL(environment, wind(ANY(size_t))).RETURN(0);

        MockAgent agent{};
        ALLOW_CALL(agent, act(ANY(Q), ANY(State), ANY(Actions))).RETURN(action);

        SarsaController<N_STEPS> testee{A, actions};

        REQUIRE(testee.sarsa(environment, agent, [] {}).q(start, action) == -.5f);
    }

    TEST_CASE("td.algorithm.sarsa.runs for the correct number of steps")
    {
        constexpr StepCount N_STEPS{2};
        const Alpha A{.5};

        auto start{State::make()};
        auto goal{State::make()};
        Actions actions{Action::make(0, 1), Action::make(1, 0)};
        std::set<Action> valid{Action::make(0, 1), Action::make(1, 0)};
        auto action{Action::make()};
        auto stepResult{State::make(1, 1)};

        MockEnvironment environment{};
        ALLOW_CALL(environment, start()).RETURN(start);
        ALLOW_CALL(environment, start()).RETURN(goal);
        ALLOW_CALL(environment, valid(ANY(Actions), ANY(State))).RETURN(valid);
        ALLOW_CALL(environment, step(ANY(State), ANY(Action))).RETURN(stepResult);
        ALLOW_CALL(environment, done(ANY(State))).RETURN(false);
        ALLOW_CALL(environment, wind(ANY(size_t))).RETURN(0);

        MockAgent agent{};
        ALLOW_CALL(agent, act(ANY(Q), ANY(State), ANY(Actions))).RETURN(action);

        SarsaController<N_STEPS> testee{A, actions};

        REQUIRE(testee.sarsa(environment, agent, [] {}).episodes.back() == N_STEPS);
    }
}