#pragma once

namespace af
{
    class array;
}

namespace introRL::actions
{
    /// <summary>
    /// Generate random actions suitable for exploring.
    /// </summary>
    /// <param name="runs">- The number of agents to generate actions for.</param>
    /// <param name="actions">- The number of actions available to each agent.</param>
    /// <returns>An array of uniformly sampled actions, one per agent.</returns>
    af::array explore(int runs, int actions);

    /// <summary>
    /// Chooses randomly, with some set ratio, between exploratory and greedy actions.
    /// </summary>
    /// <param name="q">
    /// - A matrix of shape (agents, actions) holding values of each action available to
    /// each agent.
    /// </param>
    /// <param name="epsilon">
    /// - The proportion of actions that should be exploratory.
    /// </param>
    /// <returns>
    /// An array of one action per agent, either exploratory or greedy according to
    /// epsilon.
    /// </returns>
    af::array eGreedy(const af::array& q, float epsilon);

    /// <summary>
    /// Pick the best actions given some action value table q.
    /// </summary>
    /// <param name="q">
    /// - A matrix of shape (agents, actions) holding values of each action available to
    /// each agent.
    /// </param>
    /// <returns>An array of the highest value actions available to each agent.</returns>
    af::array greedy(const af::array& q);
}