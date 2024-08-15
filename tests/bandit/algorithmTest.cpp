#include <memory>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include <arrayfire.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/trompeloeil.hpp>

#include <introRL/bandit/algorithm.hpp>
#include <introRL/bandit/types.hpp>
#include <introRL/types.hpp>

namespace irl::bandit
{
    TEST_CASE("bandit.algorithm.make.appropriately duplicates parameters")
    {
        constexpr unsigned runsPerParameter{3};
        constexpr unsigned nActions{5};

        const std::vector parameters{0.f, 1.f, 2.f, 3.f};

        struct MockAgentFactory
        {
            MockAgentFactory(const DeviceParameters& parameters, ActionCount nActions) :
                parameters{parameters.unwrap<DeviceParameters>()},
                nActions{nActions.unwrap<ActionCount>()}
            {}

            af::array parameters;
            unsigned nActions;
        };

        struct MockEnvironmentFactory
        {
            MockEnvironmentFactory(ActionCount nActions, RunCount nRuns) :
                nActions{nActions.unwrap<ActionCount>()},
                nRuns{nRuns.unwrap<RunCount>()}
            {}

            unsigned nActions;
            unsigned nRuns;
        };

        struct MockResultFactory
        {
            MockResultFactory(
                ParameterCount nParameters,
                const ReductionKeys& reductionKeys
            ) :
                nParameters{nParameters.unwrap<ParameterCount>()},
                reductionKeys{reductionKeys.unwrap<ReductionKeys>()}
            {}

            unsigned nParameters;
            af::array reductionKeys;
        };

        const auto [agent, environment, result]{
            make<MockAgentFactory, MockEnvironmentFactory, MockResultFactory>(
                parameters,
                ActionCount{nActions},
                RunsPerParameter{runsPerParameter})};

        REQUIRE(
            af::sum(agent.parameters).scalar<float>() ==
            std::accumulate(
                parameters.begin(),
                parameters.end(),
                0.f) * runsPerParameter);
        REQUIRE(agent.nActions == nActions);
        REQUIRE(environment.nActions == nActions);
        REQUIRE(environment.nRuns == parameters.size() * runsPerParameter);
        REQUIRE(result.nParameters == parameters.size());
        REQUIRE(
            af::sum(result.reductionKeys).scalar<unsigned>() ==
            (runsPerParameter * parameters.size() * (parameters.size() - 1) / 2.));
    }

    class MockAgent
    {
    public:
        MAKE_MOCK0(act, LinearActions());
        MAKE_MOCK2(update, void(LinearActions, Rewards));
    };

    class MockEnvironment
    {
    public:
        MAKE_MOCK1(reward, Rewards(LinearActions));
        MAKE_MOCK0(optimal, LinearActions());
        MAKE_MOCK0(update, void());
    };

    class MockResult
    {
    public:
        MAKE_MOCK3(update, void(LinearActions, LinearActions, Rewards));
        MAKE_MOCK0(value, int());
    };

    TEST_CASE("bandit.algorithm.run.steps an exact amount of times")
    {
        constexpr unsigned stepCount{10};

        unsigned steps{0};

        MockAgent agent{};
        REQUIRE_CALL(agent, act()).RETURN(LinearActions{af::array{0u}}).TIMES(stepCount);
        REQUIRE_CALL(agent, update(ANY(LinearActions), ANY(Rewards))).TIMES(stepCount);

        MockEnvironment environment{};
        REQUIRE_CALL(environment, reward(ANY(LinearActions)))
            .RETURN(Rewards{af::array{0.f}})
            .TIMES(stepCount);
        REQUIRE_CALL(environment, optimal())
            .RETURN(LinearActions{af::array{0u}})
            .TIMES(stepCount);
        REQUIRE_CALL(environment, update())
            .TIMES(stepCount);

        MockResult result{};
        REQUIRE_CALL(
            result,
            update(ANY(LinearActions), ANY(LinearActions), ANY(Rewards))
        )
            .TIMES(stepCount);
        REQUIRE_CALL(result, value())
            .RETURN(0)
            .TIMES(1);

        static_cast<void>(
            run(agent, environment, result, StepCount{stepCount}, [&] { ++steps; }));

        REQUIRE(steps == stepCount);
    }

    TEST_CASE("bandit.algorithm.run.returns the value of the result")
    {
        constexpr unsigned stepCount{10};
        constexpr float resultValue{20.f};

        unsigned steps{0};

        MockAgent agent{};
        ALLOW_CALL(agent, act()).RETURN(LinearActions{af::array{0u}});
        ALLOW_CALL(agent, update(ANY(LinearActions), ANY(Rewards)));

        MockEnvironment environment{};
        ALLOW_CALL(environment, reward(ANY(LinearActions)))
            .RETURN(Rewards{af::array{0.f}});
        ALLOW_CALL(environment, optimal()).RETURN(LinearActions{af::array{0u}});
        ALLOW_CALL(environment, update());

        MockResult result{};
        ALLOW_CALL(result, update(ANY(LinearActions), ANY(LinearActions), ANY(Rewards)));
        ALLOW_CALL(result, value()).RETURN(resultValue);

        REQUIRE(
            run(agent, environment, result, StepCount{stepCount}, [] {}) == resultValue);
    }
}