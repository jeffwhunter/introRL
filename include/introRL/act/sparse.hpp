#pragma once

#include <map>
#include <random>
#include <ranges>
#include <set>

#include "introRL/math/sparse.hpp"
#include "introRL/stats.hpp"

namespace irl::act
{
    /// <summary>
    /// Pick the best possible action given some action value table.
    /// </summary>
    /// <typeparam name="State">The 'positions' in the environment.</typeparam>
    /// <typeparam name="Action">The actions available in the environment.</typeparam>
    /// <typeparam name="T">The numeric type of the action value estimates.</typeparam>
    /// <param name="q">- The action value estimates.</param>
    /// <param name="state">- The state from which to act.</param>
    /// <param name="actions">- The available actions in that state.</param>
    /// <param name="generator">- A random number generator.</param>
    /// <returns>The best action according to the action value estimates.</returns>
    template <
        std::totally_ordered State,
        std::totally_ordered Action,
        std::semiregular T>
    Action greedy(
        const math::SparseMatrix<State, Action, T>& q,
        const State& state,
        const std::set<Action>& actions,
        std::uniform_random_bit_generator auto& generator)
    {
        if (!q.contains(state))
        {
            return sample(actions, generator);
        }

        const auto& q_s{q(state)};

        const auto peek{
            [](const std::map<Action, T> m, const Action& key) -> T
            {
                auto i{m.find(key)};
                if (i != m.cend())
                {
                    return T{i->second};
                }

                return T{};
            }};

        const auto actionValues{
            std::views::zip(
                actions,
                actions
                | std::views::transform(
                    [&](const Action& a) { return peek(q_s, a); }))
                | std::ranges::to<std::vector>()};

        const auto max{
            std::ranges::max(
                actionValues
                | std::views::transform(
                    [](const auto& t)
                    {
                        return std::get<1>(t);
                    }),
                std::less<T>{})};

        return sample(
            actionValues
            | std::views::filter(
                [&](const auto& t) { return std::get<1>(t) == max; })
            | std::views::transform(
                [](const auto& t) { return std::get<0>(t); })
            | std::ranges::to<std::set>(),
            generator);
    }
}