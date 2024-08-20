#pragma once

#include <algorithm>
#include <array>
#include <ranges>
#include <set>
#include <utility>

#include <introRL/td/types.hpp>

namespace irl::td
{
    /// <summary>
    /// A windy grid world.
    /// </summary>
    /// <typeparam name="W">The number of columns in the grid world.</typeparam>
    /// <typeparam name="H">The number of rows in the grid world.</typeparam>
    template <Width W, Height H>
    class Environment
    {
        using Wind = std::array<int, W.unwrap<Width>()>;

    public:
        /// <summary>
        /// Creates a windy grid world.
        /// </summary>
        /// <param name="start">- The starting position.</param>
        /// <param name="goal">- The ending position.</param>
        /// <param name="wind">- The strength of the wind in each column.</param>
        Environment(State start, State goal, Wind wind) :
            m_start{std::move(start)},
            m_goal{std::move(goal)},
            m_wind{std::move(wind)}
        {}

        /// <summary>
        /// Filters out invalid actions.
        /// </summary>
        /// <param name="actions">- The actions to filter.</param>
        /// <param name="state">- The state in which these actions originate.</param>
        /// <returns>- The actions that are valid in state.</returns>
        Actions valid(const Actions& actions, State state) const
        {
            return
                actions
                | std::views::filter([&](auto&& action) { return valid(state, action); })
                | std::ranges::to<std::set>();
        }

        /// <summary>
        /// Takes one step in the world.
        /// </summary>
        /// <param name="state">- The starting position of the step.</param>
        /// <param name="action">- The action controlling the step.</param>
        /// <returns>Where the step ends up.</returns>
        State step(const State& state, const Action& action) const
        {
            return applyWind(state + action, m_wind[state.x()]);
        }

        /// <summary>
        /// The starting position.
        /// </summary>
        /// <returns>The starting position.</returns>
        State start() const
        {
            return m_start;
        }

        /// <summary>
        /// The goal position.
        /// </summary>
        /// <returns>The goal position.</returns>
        State goal() const
        {
            return m_goal;
        }

        /// <summary>
        /// Whether or not a state is at the goal.
        /// </summary>
        /// <param name="state">- The state to test.</param>
        /// <returns>True if state is at the goal, false otherwise.</returns>
        bool done(const State& state) const
        {
            return state == m_goal;
        }

        /// <summary>
        /// The strength of the wind in some column.
        /// </summary>
        /// <param name="column">- The column to check the wind in.</param>
        /// <returns>The strength of the wind in that column.</returns>
        int wind(size_t column) const
        {
            return m_wind[column];
        }

    private:
        /// <summary>
        /// Whether or not some action is valid.
        /// </summary>
        /// <param name="state">- The state in which the action is taken.</param>
        /// <param name="action">- The action to test for validity.</param>
        /// <returns>True if action is available in state, false otherwise.</returns>
        bool valid(State state, Action action) const
        {
            auto newX{static_cast<int>(state.x()) + action.x()};
            auto newY{static_cast<int>(state.y()) + action.y()};

            return newX >= 0 && newX < W && newY >= 0 && newY < H;
        }

        /// <summary>
        /// Applies wind to some state.
        /// </summary>
        /// <param name="state">- The state to apply wind to.</param>
        /// <param name="wind">- The strength of the wind to apply.</param>
        /// <returns>The new state, pushed by the wind.</returns>
        State applyWind(State&& state, int wind) const
        {
            State result{std::move(state)};

            result.y() = static_cast<size_t>(
                std::clamp(static_cast<int>(result.y()) + wind, 0, H - 1));

            return result;
        }

        State m_start;
        State m_goal;
        Wind m_wind;
    };
}