#pragma once

#include <concepts>
#include <ranges>
#include <type_traits>

#include "introRL/concepts.hpp"
#include "introRL/ring.hpp"
#include "introRL/td/types.hpp"

namespace irl::td
{
    /// <summary>
    /// Models types that act in a reinforcement learning process.
    /// </summary>
    template <class T>
    concept CSarsaAgent = requires (
        T agent,
        Q& q,
        const GridState& state,
        const GridActions& actions)
    {
        { agent.act(q, state, actions) } -> std::same_as<GridAction>;
    };

    /// <summary>
    /// Models windy grid worlds.
    /// </summary>
    template <class T>
    concept CSarsaEnvironment = requires (
        T environment,
        GridState state,
        GridAction action,
        const GridActions& actions,
        size_t index)
    {
        { environment.start() } -> std::same_as<GridState>;
        { environment.goal() } -> std::same_as<GridState>;
        { environment.valid(actions, state) } -> std::same_as<GridActions>;
        { environment.step(state, action) } -> std::same_as<GridState>;
        { environment.done(state) } -> std::same_as<bool>;
        { environment.wind(index) } -> std::same_as<int>;
    };

    /// <summary>
    /// Models a coin flip walking environment.
    /// </summary>
    template <class T, class TState, class TAction>
    concept CNStepEnvironment = requires(
        const T environment,
        const TState state,
        const TAction action)
    {
        { environment.start() } -> std::same_as<TState>;
        { environment.step(state, action) } -> std::same_as<TState>;
        { environment.reward(state) } -> std::same_as<float>;
        { environment.done(state) } -> std::same_as<bool>;
    };

    /// <summary>
    /// Models types that update an N-step estimate.
    /// </summary>
    template <typename T, typename TStateValues, size_t N>
    concept CUpdatesNStep =
        CIsMap<TStateValues>
        && requires(
            const T t,
            TStateValues& stateValues,
            const Alpha& alpha,
            int tau,
            int bigT,
            const Ring<typename TStateValues::key_type, N + 1>&states,
            const Ring<float, N + 1>&rewards)
    {
        { T{} };
        { t.update(stateValues, alpha, tau, bigT, states, rewards) } -> std::same_as<void>;
    };
}