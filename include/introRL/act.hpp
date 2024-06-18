#pragma once

#include <concepts>

#include <arrayfire.h>

#include "introRL/types.hpp"

namespace irl::act
{
    /// <summary>
    /// Generate random actions suitable for exploring.
    /// </summary>
    /// <param name="nRuns">- The number of agents to generate actions for.</param>
    /// <param name="nActions">- The number of actions available to each agent.</param>
    /// <returns>An array of uniformly sampled actions, one per agent.</returns>
    Actions explore(RunCount nRuns, ActionCount nActions);

    /// <summary>
    /// Pick the best actions given some action value table q with ties randomly broken.
    /// </summary>
    /// <param name="q">
    /// - A matrix of shape (agents, actions) holding values of each action available to
    /// each agent.
    /// </param>
    /// <returns>An array of the highest value actions available to each agent.</returns>
    Actions greedy(const af::array& q);

    /// <summary>
    /// Types that produce an af::array when on the right hand side of > from another
    /// af::array.
    /// </summary>
    template<class T>
    concept arrayComparable = requires (T t, af::array a)
    {
        { a > t } -> std::convertible_to<af::array>;
    };

    /// <summary>
    /// Chooses randomly, with some set ratio, between exploratory and greedy actions.
    /// </summary>
    /// <param name="q">
    /// - A matrix of shape (agents, actions) holding values of each action available to
    /// each agent.
    /// </param>
    /// <param name="epsilon">
    /// - The proportion of actions that should be exploratory, one per agent.
    /// </param>
    /// <returns>
    /// An array of one action per agent, either exploratory or greedy according to
    /// epsilon.
    /// </returns>
    Actions eGreedy(const af::array& q, arrayComparable auto epsilon)
    {
        return Actions(
            af::select(
                af::randu(q.dims(0), f32) > epsilon,
                greedy(q).unwrap<Actions>(),
                explore(RunCount{q.dims(0)}, ActionCount{q.dims(1)}).unwrap<Actions>()),
            false);
    }

    /// <summary>
    /// Chooses actions randomly according to probabilities p.
    /// </summary>
    /// <param name="p">- A matrix of shape (agents, actions) holding the probability
    /// that each agent will select each action. Rows of p must sum to 1.</param>
    /// <returns>An array of one action per agent, drawn from p.</returns>
    Actions choose(const af::array& p);
}