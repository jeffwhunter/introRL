#pragma once

#include <concepts>

#include "introRL/td/types.hpp"

namespace irl::td
{
    /// <summary>
    /// Models types that act in a reinforcement learning process.
    /// </summary>
    template <class T>
    concept CAgent = requires (
        T agent,
        Q& q,
        const State& state,
        const Actions& actions)
    {
        { agent.act(q, state, actions) } -> std::same_as<Action>;
    };

    /// <summary>
    /// Models windy grid worlds.
    /// </summary>
    template <class T>
    concept CEnvironment = requires (
        T environment,
        State state,
        Action action,
        const Actions& actions,
        size_t index)
    {
        { environment.start() } -> std::same_as<State>;
        { environment.goal() } -> std::same_as<State>;
        { environment.valid(actions, state) } -> std::same_as<Actions>;
        { environment.step(state, action) } -> std::same_as<State>;
        { environment.done(state) } -> std::same_as<bool>;
        { environment.wind(index) } -> std::same_as<int>;
    };
}