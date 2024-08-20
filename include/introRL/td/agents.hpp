#pragma once

#include <random>

#include "introRL/td/types.hpp"

namespace irl::td
{
    /// <summary>
    /// An agent that can either explore randomly or pick the best action given some
    /// action value table.
    /// </summary>
    class EGreedy
    {
    public:
        /// <summary>
        /// Creates an EGreedy.
        /// </summary>
        /// <param name="e">- The probability that this agent will explore.</param>
        /// <param name="generator">- A random number generator.</param>
        EGreedy(Epsilon e, std::mt19937& generator);

        /// <summary>
        /// Picks an action.
        /// </summary>
        /// <param name="q">- An action value table.</param>
        /// <param name="state">- The state from which to act.</param>
        /// <param name="actions">- The actions available in that state.</param>
        /// <returns>
        /// Some action the agent has picked, either exploratory or greedy.
        /// </returns>
        Action act(const Q& q, const State& state, const Actions& actions);

    private:
        /// <summary>
        /// Whether or not the agent should explore for some move.
        /// </summary>
        /// <returns>True if the agent should explore, false otherwise.</returns>
        bool shouldExplore() const;

        Epsilon m_e;
        std::mt19937& m_generator;
    };
}