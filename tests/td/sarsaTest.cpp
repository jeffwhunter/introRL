#include <random>
#include <set>

#include <catch2/catch_test_macros.hpp>
#include <catch2/trompeloeil.hpp>

#include <introRL/math/sparse.hpp>
#include <introRL/td/sarsa.hpp>
#include <introRL/td/types.hpp>
#include <introRL/types.hpp>

namespace irl::td
{
    class MockEnvironment
    {
    public:
        MAKE_MOCK0(start, GridState());
        MAKE_MOCK0(goal, GridState());
        MAKE_MOCK2(valid, GridActions(GridActions, GridState));
        MAKE_MOCK2(step, GridState(GridState, GridAction));
        MAKE_MOCK1(done, bool(GridState));
        MAKE_MOCK1(wind, int(size_t));
    };

    class MockAgent
    {
    public:
        MAKE_MOCK3(act, GridAction(const Q&, const GridState&, const GridActions&));
    };

    TEST_CASE("td.algorithm.sarsa.compiles action value estimates")
    {
        constexpr StepCount N_STEPS{1};
        const Alpha A{.5};

        auto start{GridState::make()};
        auto goal{GridState::make()};
        GridActions actions{GridAction::make(0, 1), GridAction::make(1, 0)};
        std::set<GridAction> valid{GridAction::make(0, 1), GridAction::make(1, 0)};
        auto action{GridAction::make()};
        auto stepResult{GridState::make(1, 1)};

        MockEnvironment environment{};
        ALLOW_CALL(environment, start()).RETURN(start);
        ALLOW_CALL(environment, start()).RETURN(goal);
        ALLOW_CALL(environment, valid(ANY(GridActions), ANY(GridState))).RETURN(valid);
        ALLOW_CALL(environment, step(ANY(GridState), ANY(GridAction))).RETURN(stepResult);
        ALLOW_CALL(environment, done(ANY(GridState))).RETURN(false);
        ALLOW_CALL(environment, wind(ANY(size_t))).RETURN(0);

        MockAgent agent{};
        ALLOW_CALL(agent, act(ANY(Q), ANY(GridState), ANY(GridActions))).RETURN(action);

        SarsaController<N_STEPS> testee{A, actions};

        REQUIRE(testee.sarsa(environment, agent, [] {}).q(start, action) == -.5f);
    }

    TEST_CASE("td.algorithm.sarsa.runs for the correct number of steps")
    {
        constexpr StepCount N_STEPS{2};
        const Alpha A{.5};

        auto start{GridState::make()};
        auto goal{GridState::make()};
        GridActions actions{GridAction::make(0, 1), GridAction::make(1, 0)};
        std::set<GridAction> valid{GridAction::make(0, 1), GridAction::make(1, 0)};
        auto action{GridAction::make()};
        auto stepResult{GridState::make(1, 1)};

        MockEnvironment environment{};
        ALLOW_CALL(environment, start()).RETURN(start);
        ALLOW_CALL(environment, start()).RETURN(goal);
        ALLOW_CALL(environment, valid(ANY(GridActions), ANY(GridState))).RETURN(valid);
        ALLOW_CALL(environment, step(ANY(GridState), ANY(GridAction))).RETURN(stepResult);
        ALLOW_CALL(environment, done(ANY(GridState))).RETURN(false);
        ALLOW_CALL(environment, wind(ANY(size_t))).RETURN(0);

        MockAgent agent{};
        ALLOW_CALL(agent, act(ANY(Q), ANY(GridState), ANY(GridActions))).RETURN(action);

        SarsaController<N_STEPS> testee{A, actions};

        REQUIRE(testee.sarsa(environment, agent, [] {}).episodes.back() == N_STEPS);
    }
}