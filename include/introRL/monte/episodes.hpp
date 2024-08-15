#pragma once

#include <experimental/generator>
#include <optional>
#include <vector>

#include "introRL/monte/types.hpp"
#include "introRL/stats.hpp"

namespace irl::monte
{
    /// <summary>
    /// A record of an agents steps through and environment.
    /// </summary>
    class Episode final
    {
    public:
        /// <summary>
        /// A single step in an episode.
        /// </summary>
        struct Step
        {
            /// <summary>
            /// The state the agent was in at the start of this step.
            /// </summary>
            State state{};

            /// <summary>
            /// The action the agent took while it was in state.
            /// </summary>
            Probable<Action> action{};
        };

        /// <summary>
        /// Appends a new step to this episode.
        /// </summary>
        /// <param name="state">
        /// - The state the agent was in when action was taken.
        /// </param>
        /// <param name="action">- The action taken when the agent was in state.</param>
        void append(State state, Probable<Action> action);

        /// <summary>
        /// Sets the final position of the episode.
        /// </summary>
        /// <param name="position">
        /// - The goal position that triggered the end of the episode.
        /// </param>
        void setFinalPosition(Position position);

        /// <summary>
        /// Returns the number of full steps this episode recorded.
        /// </summary>
        /// <returns>The number of steps contained in this episode.</returns>
        size_t bigT() const;

        /// <summary>
        /// Returns some indexed step.
        /// </summary>
        /// <param name="i">- The index of the step to retrieve.</param>
        /// <returns>The i'th step, starting at 0.</returns>
        const Step& getStep(unsigned i) const;

        /// <summary>
        /// Returns a range of all positions the agent visited during the episode,
        /// including the final one if it exists.
        /// </summary>
        /// <returns>A range over all positions in the episode.</returns>
        std::experimental::generator<Position> getAllPositions() const;

    private:
        std::vector<Step> m_steps{};
        std::optional<Position> m_finalPosition{};
    };
}