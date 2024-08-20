#pragma once

#include <concepts>
#include <functional>
#include <ranges>
#include <vector>

#include "introRL/math/sparse.hpp"
#include "introRL/td/concepts.hpp"
#include "introRL/types.hpp"

namespace irl::td
{
    /// <summary>
    /// Records and returns an episode of some agent acting according to an action value
    /// table.
    /// </summary>
    /// <typeparam name="MAX_STEPS">The maximum steps to record.</typeparam>
    /// <param name="actions">- The actions available to the agent.</param>
    /// <param name="q">- The action value table.</param>
    /// <param name="environment">- The environment to act in.</param>
    /// <param name="agent">- The acting agent.</param>
    /// <returns>The episode that played out.</returns>
    template <StepCount MAX_STEPS>
    std::vector<State> demo(
        const Actions& actions,
        const Q& q,
        CEnvironment auto& environment,
        CAgent auto& agent)
    {
        std::vector<State> result{};

        auto state{environment.start()};
        result.push_back(state);

        auto action{agent.act(q, state, environment.valid(actions, state))};

        for (auto i : std::views::iota(0U, MAX_STEPS.unwrap<StepCount>())
            | std::views::take_while([&](auto _) { return !environment.done(state); }))
        {
            state = environment.step(state, action);
            action = agent.act(q, state, environment.valid(actions, state));

            result.push_back(state);
        }

        return result;
    }

    /// <summary>
    /// The result of a sarsa process.
    /// </summary>
    struct SarsaResult
    {
        /// <summary>
        /// The action value table that sarsa computed.
        /// </summary>
        Q q{};

        /// <summary>
        /// The timesteps that each episode in the learning process took.
        /// </summary>
        std::vector<StepCount> episodes{};
    };

    /// <summary>
    /// Uses the SARSA algorithm to produce an action value table.
    /// </summary>
    /// <typeparam name="N_STEPS">The number of steps to spend training.</typeparam>
    template <StepCount N_STEPS>
    class SarsaController
    {
    public:
        /// <summary>
        /// Creates a SarsaController.
        /// </summary>
        /// <param name="a">- The step size of the SARSA algorithm.</param>
        /// <param name="actions">- The actions available to the agent.</param>
        SarsaController(Alpha a, const Actions& actions) : m_a{a}, m_actions{actions} {}

        /// <summary>
        /// Uses the SARSA algorithm to produce an estimated action value table that, if
        /// followed, is expected to optimize acting in some environment.
        /// </summary>
        /// <param name="environment">
        /// - The environment in which the agent learns to act.
        /// </param>
        /// <param name="agent">- Controls how decisions are taken.</param>
        /// <param name="stepCallback">
        /// - Called once for each step in the process.
        /// </param>
        /// <returns>
        /// An estimated action value table for a given environment.
        /// </returns>
        [[nodiscard]] SarsaResult sarsa(
            CEnvironment auto& environment,
            CAgent auto& agent,
            std::invocable auto&& stepCallback)
        {
            SarsaResult result{};

            StepCount numSteps{0};

            result.episodes.push_back(numSteps);

            for (auto i : std::views::iota(0U)
                | std::views::take_while([&](auto i) { return numSteps < N_STEPS; }))
            {
                numSteps += episode(
                    result.q,
                    environment,
                    agent,
                    stepCallback,
                    N_STEPS - numSteps);

                result.episodes.push_back(numSteps);
            }

            return result;
        }

    private:
        /// <summary>
        /// Runs one episode in the SARSA algorithm.
        /// </summary>
        /// <param name="q">- The action value estimates to learn.</param>
        /// <param name="environment">- The environment to optimize acting in.</param>
        /// <param name="agent">- Controls how actions are taken.</param>
        /// <param name="stepCallback">
        /// - Called once for each step in the process.
        /// </param>
        /// <param name="maxSteps">- The maximum length of the episode.</param>
        /// <returns>How many steps the episode took.</returns>
        StepCount episode(
            Q& q,
            CEnvironment auto& environment,
            CAgent auto& agent,
            std::invocable auto&& stepCallback,
            StepCount maxSteps)
        {
            auto state{environment.start()};
            auto action{act(q, state, environment, agent)};

            StepCount numSteps{0};

            for (auto i : std::views::iota(0U, maxSteps.unwrap<StepCount>())
                | std::views::take_while(
                    [&](auto _) { return !environment.done(state); }))
            {
                step(q, state, action, environment, agent);

                ++numSteps;

                std::invoke(stepCallback);
            }

            return numSteps;
        }

        /// <summary>
        /// Runs one step in an episode of the SARSA algorithm.
        /// </summary>
        /// <param name="q">- The action value estimates to learn.</param>
        /// <param name="state">- The state the agent was in before this step.</param>
        /// <param name="action">
        /// - The action the agent took in the state before this step.
        /// </param>
        /// <param name="environment">- The environment to optimize acting in.</param>
        /// <param name="agent">- Controls how actions are taken.</param>
        void step(
            Q& q,
            State& state,
            Action& action,
            CEnvironment auto& environment,
            CAgent auto& agent)
        {
            const auto& sPrime{environment.step(state, action)};
            const auto& aPrime{act(q, sPrime, environment, agent)};

            auto& q_sa{q(state, action)};
            q_sa += m_a * (-1 + q.peek(sPrime, aPrime) - q_sa);

            state = sPrime;
            action = aPrime;
        }

        /// <summary>
        /// Produce an action by having the agent decide between valid actions.
        /// </summary>
        /// <param name="q">
        /// - The action value estimates that the agent will follow.
        /// </param>
        /// <param name="state">- The current state of the agent.</param>
        /// <param name="environment">
        /// - The environment in which the agent should act.
        /// </param>
        /// <param name="agent">- Controls how actions are taken.</param>
        /// <returns></returns>
        Action act(
            const Q& q,
            const State& state,
            CEnvironment auto& environment,
            CAgent auto& agent)
        {
            return agent.act(q, state, environment.valid(m_actions, state));
        }

        Alpha m_a{};
        const Actions& m_actions;
    };
}