#pragma once

#include <arrayfire.h>

#include "introRL/types.hpp"

namespace irl::bandit::environments
{
    /// <summary>
    /// An environment that simulates a number of slot machines in parallel.
    /// </summary>
    class Stationary
    {
    public:
        /// <summary>
        /// Creates a Stationary with specific number of slot machines for a specific
        /// number of agents.
        /// machines.
        /// </summary>
        /// <param name="nActions">- The number of slot machines to pull from.</param>
        /// <param name="nRuns">
        /// - The number of agents to simulate runs for in parallel.
        /// </param>
        Stationary(ActionCount nActions, RunCount nRuns);

        /// <summary>
        /// Generate rewards for a number of agents making one pull each from a number of
        /// slot machines.
        /// </summary>
        /// <param name="actions">
        /// - An array of actions, one per agent, to return rewards for.
        /// </param>
        /// <returns>
        /// The reward earned from each agent pulling their chosen slot machines.
        /// </returns>
        Rewards reward(const Actions& actions) const;

        /// <summary>
        /// The optimal actions to pick for each agent.
        /// </summary>
        /// <returns>An array of optimal actions, one per agent.</returns>
        Actions optimal() const;

        /// <summary>
        /// Does nothing at all.
        /// </summary>
        void update() const;

    protected:
        af::array m_qStar;
    };

    /// <summary>
    /// An environment that simulates a number of slot machines in parallel, where the
    /// slot machine will randomly change their value each turn.
    /// </summary>
    /// <typeparam name="WALK_SIZE">The average change in slot machine value.</typeparam>
    template <float WALK_SIZE>
    class Walking : public Stationary
    {
    public:
        using Stationary::Stationary;

        /// <summary>
        /// Randomly walk each slot machine's average value.
        /// </summary>
        void update()
        {
            m_qStar += af::randn(m_qStar.dims(), f32) * WALK_SIZE;
        }
    };
}