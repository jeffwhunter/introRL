#pragma once

#include <concepts>
#include <random>
#include <ranges>
#include <utility>

#include "introRL/concepts.hpp"
#include "introRL/ring.hpp"
#include "introRL/td/concepts.hpp"
#include "introRL/td/types.hpp"
#include "introRL/types.hpp"

namespace irl::td
{
    /// <summary>
    /// Estimates state values using an N-step algorithm.
    /// </summary>
    /// <typeparam name="TUpdater">The type to update the estimate with.</typeparam>
    /// <typeparam name="N">The number of states to update for each experience.</typeparam>
    /// <typeparam name="EPISODES_PER_RUN">
    /// The number of episodes to average results over.
    /// </typeparam>
    /// <typeparam name="TStateValues">The type of the value estimates.</typeparam>
    template <
        StepCount N,
        EpisodeCount EPISODES_PER_RUN,
        CIsMap TStateValues,
        class TUpdater>
    requires CUpdatesNStep<TUpdater, TStateValues, N.unwrap<StepCount>()>
    class NStepTD
    {
    protected:
        using State = TStateValues::key_type;
        using Value = TStateValues::mapped_type;

    public:
        /// <summary>
        /// Estimate state values.
        /// </summary>
        /// <typeparam name="E">The type of the environment.</typeparam>
        /// <param name="alpha">- The step size of the model.</param>
        /// <param name="environment">- The environment to learn in.</param>
        /// <param name="updateCallback">- A callback to call with learning progress.</param>
        /// <param name="generator">- A random number generator.</param>
        template <class E>
            requires CNStepEnvironment<E, State, bool>
        void stateValues(
            const Alpha& alpha,
            const E& environment,
            const std::invocable<const TStateValues&> auto& updateCallback,
            std::uniform_random_bit_generator auto& generator)
        {
            TStateValues stateValues{};
            TUpdater updater{};

            for (auto i : std::views::iota(0U, EPISODES_PER_RUN.unwrap<EpisodeCount>()))
            {
                runEpisode(updater, alpha, environment, stateValues, generator);

                std::invoke(updateCallback, stateValues);
            }
        }

    private:
        /// <summary>
        /// Runs one episode of an N-step learning algorithm.
        /// </summary>
        /// <typeparam name="E">The type of the environment.</typeparam>
        /// <param name="updater">- The object that updates the value estimates.</param>
        /// <param name="alpha">- The step size of the model.</param>
        /// <param name="environment">- The environment to learn in.</param>
        /// <param name="stateValues">- The state value estimates.</param>
        /// <param name="generator">- A random number generator.</param>
        template <class E>
            requires CNStepEnvironment<E, State, bool>
        void runEpisode(
            const TUpdater& updater,
            const Alpha& alpha,
            const E& environment,
            TStateValues& stateValues,
            std::uniform_random_bit_generator auto& generator)
        {
            Ring<State, N + 1> states{};
            Ring<float, N + 1> rewards{};

            states[0] = environment.start();

            int tau{};
            int T{std::numeric_limits<int>::max()};
            for (auto t : std::views::iota(0)
                | std::views::take_while(
                    [&](auto t)
                    {
                        return tau < T - 1;
                    }))
            {
                if (t < T)
                {
                    static std::bernoulli_distribution actor{.5f};

                    states[t + 1] = environment.step(states[t], actor(generator));
                    rewards[t + 1] = environment.reward(states[t + 1]);

                    if (environment.done(states[t + 1]))
                    {
                        T = t + 1;
                    }
                }

                tau = t - N + 1;
                if (tau >= 0)
                {
                    updater.update(stateValues, alpha, tau, T, states, rewards);
                }
            }
        }
    };

    /// <summary>
    /// Updates a value estimate with N-step returns.
    /// </summary>
    /// <typeparam name="N">The number of states to update for each experience.</typeparam>
    /// <typeparam name="EPISODES_PER_RUN">
    /// The number of episodes to average results over.
    /// </typeparam>
    /// <typeparam name="TStateValues">The type of the value estimates.</typeparam>
    template <
        StepCount N,
        EpisodeCount EPISODES_PER_RUN,
        CIsMap TStateValues>
    struct UpdatesWithNStepReturn
    {
        using State = TStateValues::key_type;
        using Value = TStateValues::mapped_type;

        /// <summary>
        /// Updates one state estimate at some specific point in time with N-step returns.
        /// </summary>
        /// <param name="stateValues">- The state value estimate.</param>
        /// <param name="alpha">- The step size of the model.</param>
        /// <param name="tau">- The timestep being updated.</param>
        /// <param name="bigT">- The timestep the episode ends at.</param>
        /// <param name="states">- The visited states.</param>
        /// <param name="rewards">- The observed rewards.</param>
        void update(
            TStateValues& stateValues,
            const Alpha& alpha,
            int tau,
            int bigT,
            const Ring<State, N + 1>& states,
            const Ring<float, N + 1>& rewards) const
        {
            Value G{
                std::ranges::fold_left(
                    std::views::iota(tau + 1, std::min(tau + N, bigT) + 1)
                    | std::views::transform([&](int i) { return rewards[i]; }),
                    Value{},
                    std::plus<>{})};

            if (tau + N < bigT)
            {
                G += stateValues[states[tau + N]];
            }

            auto& VSt{stateValues[states[tau]]};

            VSt += alpha * (G - VSt);
        }
    };

    /// <summary>
    /// Updates a value estimate with the sum of TD errors.
    /// </summary>
    /// <typeparam name="N">The number of states to update for each experience.</typeparam>
    /// <typeparam name="EPISODES_PER_RUN">
    /// The number of episodes to average results over.
    /// </typeparam>
    /// <typeparam name="TStateValues">The type of the value estimates.</typeparam>
    template <
        StepCount N,
        EpisodeCount EPISODES_PER_RUN,
        CIsMap TStateValues>
    struct UpdatesWithSumTDErrors
    {
        using State = TStateValues::key_type;
        using Value = TStateValues::mapped_type;

        /// <summary>
        /// Updates one state estimate at some specific point in time with the sum of TD errors.
        /// </summary>
        /// <param name="stateValues">- The state value estimate.</param>
        /// <param name="alpha">- The step size of the model.</param>
        /// <param name="tau">- The timestep being updated.</param>
        /// <param name="bigT">- The timestep the episode ends at.</param>
        /// <param name="states">- The visited states.</param>
        /// <param name="rewards">- The observed rewards.</param>
        void update(
            TStateValues& stateValues,
            const Alpha& alpha,
            int tau,
            int bigT,
            const Ring<State, N + 1>& states,
            const Ring<float, N + 1>& rewards) const
        {
            stateValues[states[tau]] +=
                alpha
                * std::ranges::fold_left(
                    std::views::iota(tau, std::min(tau + N - 1, bigT) + 1)
                    | std::views::transform(
                        [&](auto k) { return delta(k, stateValues, states, rewards); }),
                    Value{},
                    std::plus<>{});
        }

    private:
        /// <summary>
        /// The TD error for some specific timestep.
        /// </summary>
        /// <param name="t">- The timestep of the TD error to calculate.</param>
        /// <param name="stateValues">- The state value estimate.</param>
        /// <param name="states">- The visited states.</param>
        /// <param name="rewards">- The observed rewards.</param>
        /// <returns>The TD error for timestep t.</returns>
        Value delta(
            int t,
            TStateValues& stateValues,
            const Ring<State, N + 1>& states,
            const Ring<float, N + 1>& rewards) const
        {
            return Value{rewards[t + 1]} + stateValues[states[t + 1]] - stateValues[states[t]];
        }
    };
}