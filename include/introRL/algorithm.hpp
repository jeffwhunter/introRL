#pragma once

#include <functional>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <arrayfire.h>

#include "introRL/afUtils.hpp"
#include "introRL/banditTypes.hpp"
#include "introRL/basicTypes.hpp"

namespace irl::bandit::algorithm
{
    /// <summary>
    /// Types that are not void.
    /// </summary>
    template<class T>
    concept not_void = !std::is_void_v<T>;

    /// <summary>
    /// Types that can act as agents in bandit processes.
    /// </summary>
    template <class TBanditAgent>
    concept BanditAgent = requires (
        TBanditAgent agent,
        const Actions& actions,
        const Rewards& rewards)
    {
        { agent.act() } -> std::same_as<Actions>;
        agent.update(actions, rewards);
    };

    /// <summary>
    /// Types that can be called with DeviceParameters and an ActionCount to create an
    /// agent.
    /// </summary>
    template <class TBanditAgentFactory>
    concept BanditAgentFactory = requires (
        TBanditAgentFactory factory,
        const DeviceParameters& deviceParameters,
        ActionCount nActions)
    {
        TBanditAgentFactory{deviceParameters, nActions};
    };

    /// <summary>
    /// Types that can act as environments in bandit processes.
    /// </summary>
    template <class TBanditEnvironment>
    concept BanditEnvironment = requires (
        TBanditEnvironment environment,
        const Actions & actions)
    {
        { environment.reward(actions) } -> std::same_as<Rewards>;
        { environment.optimal() } -> std::same_as<Actions>;
        { environment.update() } -> std::same_as<void>;
    };

    /// <summary>
    /// Types that can be called with an ActionCount and a RunCount to create an
    /// environment.
    /// </summary>
    template <class TBanditEnvironmentFactory>
    concept BanditEnvironmentFactory = requires (
        TBanditEnvironmentFactory factory,
        ActionCount nActions,
        RunCount nRuns)
    {
        TBanditEnvironmentFactory{nActions, nRuns};
    };

    /// <summary>
    /// Types that can act as a results in bandit processes.
    /// </summary>
    template <class TBanditResult>
    concept BanditResult = requires (
        TBanditResult result,
        const Actions & actions,
        const Actions & optimal,
        const Rewards & rewards)
    {
        result.update(actions, optimal, rewards);
        { result.value() } -> not_void;
    };

    /// <summary>
    /// Types that can be called with a ParameterCount and some ReductionKeys to create a
    /// result.
    /// </summary>
    template <class TBanditResultFactory>
    concept BanditResultFactory = requires (
        TBanditResultFactory factory,
        ParameterCount nParameters,
        const ReductionKeys& reductionKeys)
    {
        TBanditResultFactory{nParameters, reductionKeys};
    };

    /// <summary>
    /// Given appropriate types of each, create an agent, environment, and result for a
    /// bandit process with a given number of actions. Parameters will be duplicated and
    /// distributed to these objects as appropriate.
    /// </summary>
    /// <typeparam name="TAgent">The type of bandit agent to create.</typeparam>
    /// <typeparam name="TEnvironment">
    /// The type of bandit environment to create.
    /// </typeparam>
    /// <typeparam name="TResult">The type of bandit result to create.</typeparam>
    /// <param name="parameters">- The input parameters for the bandit agent.</param>
    /// <param name="nActions">
    /// - The number of actions on each step of the bandit processes.
    /// </param>
    /// <param name="runsPerParam">
    /// - The number of parallel runs to do for each input parameter.
    /// </param>
    /// <returns>A tuple containing the agent, environment, and result.</returns>
    template <
        BanditAgentFactory TAgent,
        BanditEnvironmentFactory TEnvironment,
        BanditResultFactory TResult>
    [[nodiscard]] decltype(auto) make(
        const std::vector<float>& parameters,
        ActionCount nActions,
        RunsPerParameter runsPerParam)
    {
        const ParameterCount nParam{std::ranges::size(parameters)};
        const auto nRuns{nParam * runsPerParam};

        const unsigned uRunsPerParam{runsPerParam.unwrap<RunsPerParameter>()};

        const auto i{af::iota(nRuns.unwrap<RunCount>(), 1, u32)};
        const auto keys{i / uRunsPerParam};

        const auto tiled{af::tile(toArrayFire(parameters), uRunsPerParam)};
        const auto deshuffle{
            ((i % uRunsPerParam) * nParam.unwrap<ParameterCount>()) + keys};

        return std::make_tuple(
            TAgent{DeviceParameters{tiled(deshuffle)}, nActions},
            TEnvironment{nActions, nRuns},
            TResult{nParam, ReductionKeys{keys}});
    }

    /// <summary>
    /// Runs a number of simple bandit algorithms (p.32 Sutton, Barto (2018)) with a
    /// given agent, environment, and result, for a some number of steps, calling the
    /// progress callback each step.
    /// </summary>
    /// <param name="agent">
    /// - The agent responsible for learning to pick the best actions.
    /// </param>
    /// <param name="environment">
    /// - The environment in which the agent has to optimize actions.
    /// </param>
    /// <param name="result">- The final result of the learning process.</param>
    /// <param name="nSteps">- The number of steps to run the process for.</param>
    /// <param name="progressCallback">- The callback to call each step.</param>
    /// <returns>The final value calculated by the result.</returns>
    [[nodiscard]] decltype(auto) run(
        BanditAgent auto&& agent,
        BanditEnvironment auto&& environment,
        BanditResult auto&& result,
        const StepCount nSteps,
        std::function<void(void)> progressCallback)
    {
        for (const auto _ : std::views::iota(0u, nSteps.unwrap<StepCount>()))
        {
            const auto actions{agent.act()};
            const auto rewards{environment.reward(actions)};

            agent.update(actions, rewards);
            result.update(actions, environment.optimal(), rewards);
            environment.update();

            progressCallback();
        }

        return result.value();
    }

    /// <summary>
    /// A bandit process with a specific number of actions, parallel runs per parameter,
    /// and total timesteps.
    /// </summary>
    class Bandits
    {
    public:
        /// <summary>
        /// Create a Bandit with the given options.
        /// </summary>
        /// <param name="nActions">
        /// - The number of actions each agent can pick from each step.
        /// </param>
        /// <param name="runsPerParam">
        /// - The number of parallel runs to run per input parameter.
        /// </param>
        /// <param name="nStep">- The number of timesteps to run the process for.</param>
        Bandits(ActionCount nActions, RunsPerParameter runsPerParam, StepCount nStep) :
            m_nActions{nActions},
            m_runsPerParam{runsPerParam},
            m_nStep{nStep}
        {}

        /// <summary>
        /// Runs parallel bandit processes for a number of input parameters.
        /// </summary>
        /// <typeparam name="TAgent">
        /// The agent type responsible for learning to pick the best actions.
        /// </typeparam>
        /// <typeparam name="TEnvironment">
        /// The environment type in which agents have to optimize actions.
        /// </typeparam>
        /// <typeparam name="TResult">
        /// The final result type of the learning process.
        /// </typeparam>
        /// <param name="parameters">
        /// - The input parameters to duplicate and distribute to a number of parallel
        /// bandit processes.
        /// </param>
        /// <param name="progressCallback">- The callback to call each step.</param>
        /// <returns></returns>
        template <class TAgent, class TEnvironment, class TResult>
        requires
            BanditAgentFactory<TAgent> && BanditAgent<TAgent> &&
            BanditEnvironmentFactory<TEnvironment> && BanditEnvironment<TEnvironment> &&
            BanditResultFactory<TResult> && BanditResult<TResult>
        [[nodiscard]] decltype(auto) learn(
            const std::vector<float>& parameters,
            std::function<void(void)> progressCallback
        ) const
        {
            auto&& [agent, environment, result]{
                make<TAgent, TEnvironment, TResult>(
                    parameters,
                    m_nActions,
                    m_runsPerParam)};

            return run(agent, environment, result, m_nStep, progressCallback);
        }

    private:
        ActionCount m_nActions;
        RunsPerParameter m_runsPerParam;
        StepCount m_nStep;
    };
}