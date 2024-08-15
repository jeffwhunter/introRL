# pragma once

#include <map>
#include <random>
#include <set>

#include "introRL/monte/types.hpp"
#include "introRL/stats.hpp"

namespace irl::monte
{
    /// <summary>
    /// Decides when and how to explore.
    /// </summary>
    class Explorer
    {
    public:
        /// <summary>
        /// Makes an Explorer.
        /// </summary>
        /// <param name="minAction">
        /// - The minimum element in any exploring action.
        /// </param>
        /// <param name="maxAction">
        /// - The maximum element in any exploring action.
        /// </param>
        /// <param name="seed">- The random seed for the action sampler.</param>
        /// <returns>An Explorer.</returns>
        static Explorer make(int minAction, int maxAction, unsigned seed);

        /// <summary>
        /// Destroys an Explorer.
        /// </summary>
        virtual ~Explorer() = default;

        /// <summary>
        /// Whether or not something should explore.
        /// </summary>
        /// <param name="pExplore">- The probability to explore.</param>
        /// <returns>True if something should explore, false otherwise.</returns>
        bool should_explore(float pExplore);

        /// <summary>
        /// An exploratory action.
        /// </summary>
        /// <returns>A randomly sampled action.</returns>
        Action explore();

    private:
        struct M
        {
            std::set<Action> actions;
            std::mt19937 generator;
        } m;

        Explorer(M m);
    };

    /// <summary>
    /// An agent that either explores or takes expert hand-coded actions.
    /// </summary>
    class ExpertAgent
    {
    public:
        /// <summary>
        /// Creates an ExpertAgent.
        /// </summary>
        /// <param name="sprintSpeed">- How fast to go forwards before the turn.</param>
        /// <param name="sprintStop">- Where to stop going forwards.</param>
        /// <param name="turnStart">- Where to start turning right.</param>
        /// <param name="explorer">- When and how to explore.</param>
        ExpertAgent(
            size_t sprintSpeed,
            size_t sprintStop,
            size_t turnStart,
            Explorer& explorer);

        /// <summary>
        /// Destroys an ExpertAgent.
        /// </summary>
        virtual ~ExpertAgent() = default;

        /// <summary>
        /// Returns a new action given some state.
        /// </summary>
        /// <param name="state">- The state the agent is in before the decision.</param>
        /// <param name="pExplore">- The probability to explore.</param>
        /// <returns>Some probable action taken in response to state.</returns>
        Probable<Action> act(const State& state, float pExplore);

    private:
        /// <summary>
        /// Returns a hand coded expert action for some state.
        /// </summary>
        /// <param name="state">- The state the agent is in before the decision.</param>
        /// <returns>A hand coded expert action.</returns>
        Action example(const State& state) const;

        size_t m_sprintSpeed;
        size_t m_sprintStop;
        size_t m_turnStart;
        Explorer& m_explorer;
    };

    /// <summary>
    /// An agent that either explores or takes actions according to a given table.
    /// </summary>
    class TableAgent
    {
    public:
        /// <summary>
        /// Creates a TableAgent.
        /// </summary>
        /// <param name="pi">- The table of actions to take in each state.</param>
        /// <param name="explorer">- When and how to explore.</param>
        TableAgent(std::map<State, Action>&& pi, Explorer& explorer);

        /// <summary>
        /// Destroys a TableAgent.
        /// </summary>
        virtual ~TableAgent() = default;

        /// <summary>
        /// Returns a new action given some state.
        /// </summary>
        /// <param name="state">- The state the agent is in before the decision.</param>
        /// <param name="pExplore">- The probability to explore.</param>
        /// <returns>Some probable action taken in response to state.</returns>
        Probable<Action> act(State state, float pExplore) const;

    private:
        std::map<State, Action> m_pi{};
        Explorer& m_explorer;
    };
}