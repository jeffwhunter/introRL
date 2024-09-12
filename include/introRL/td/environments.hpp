#pragma once

#include <algorithm>
#include <array>
#include <random>
#include <ranges>
#include <set>
#include <utility>

#include <introRL/stats.hpp>
#include <introRL/td/types.hpp>

namespace irl::td
{
    /// <summary>
    /// A random walk controlled by a coin flip.
    /// </summary>
    /// <typeparam name="N">The number of states to walk over.</typeparam>
    template <StateCount N>
    class Walk
    {
        static_assert(N.template unwrap<StateCount>() >= 3);
        static_assert(N.template unwrap<StateCount>() % 2 == 1);

    public:
        /// <summary>
        /// Returns the starting point of the walk.
        /// </summary>
        /// <returns>The midpoint between the two ends.</returns>
        WalkState start() const
        {
            return WalkState{1U + N / 2U};
        }

        /// <summary>
        /// Returns the next state for an action taken from some state.
        /// </summary>
        /// <param name="state">- The state to act from.</param>
        /// <param name="action">- The action taken.</param>
        /// <returns>The new state that resulted from taking action in state.</returns>
        WalkState step(WalkState state, bool action) const
        {
            return action
                ? WalkState{static_cast<unsigned>(state + 1)}
                : WalkState{static_cast<unsigned>(state - 1)};
        }

        /// <summary>
        /// Rewards some state.
        /// </summary>
        /// <param name="state">- The reward producing state.</param>
        /// <returns>The reward produced when visiting state.</returns>
        float reward(WalkState state) const
        {
            if (state == WalkState{N + 1})
            {
                return 1.f;
            }

            if (state == WalkState{0})
            {
                return -1.f;
            }

            return .0f;
        }

        /// <summary>
        /// Returns if some state is terminal.
        /// </summary>
        /// <param name="state">- The state to judge for terminality.</param>
        /// <returns>True is state is terminal, false otherwise.</returns>
        bool done(WalkState state) const
        {
            return state == WalkState{0} || state == WalkState{N + 1};
        }
    };

    /// <summary>
    /// A windy grid world.
    /// </summary>
    /// <typeparam name="W">The number of columns in the grid world.</typeparam>
    /// <typeparam name="H">The number of rows in the grid world.</typeparam>
    template <Width W, Height H>
    class Windy
    {
    protected:
        using Wind = std::array<int, W.unwrap<Width>()>;

    public:
        /// <summary>
        /// Creates a windy grid world.
        /// </summary>
        /// <param name="start">- The starting position.</param>
        /// <param name="goal">- The ending position.</param>
        /// <param name="wind">- The strength of the wind in each column.</param>
        Windy(GridState start, GridState goal, Wind wind) :
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
        GridActions valid(const GridActions& actions, GridState state) const
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
        GridState step(const GridState& state, const GridAction& action) const
        {
            return applyWind(state + action, wind(state.x()));
        }

        /// <summary>
        /// The starting position.
        /// </summary>
        /// <returns>The starting position.</returns>
        GridState start() const
        {
            return m_start;
        }

        /// <summary>
        /// The goal position.
        /// </summary>
        /// <returns>The goal position.</returns>
        GridState goal() const
        {
            return m_goal;
        }

        /// <summary>
        /// Whether or not a state is at the goal.
        /// </summary>
        /// <param name="state">- The state to test.</param>
        /// <returns>True if state is at the goal, false otherwise.</returns>
        bool done(const GridState& state) const
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
        bool valid(GridState state, GridAction action) const
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
        virtual GridState applyWind(GridState&& state, int wind) const
        {
            GridState result{std::move(state)};

            result.y() = static_cast<size_t>(
                std::clamp(static_cast<int>(result.y()) + wind, 0, H - 1));

            return result;
        }

        GridState m_start;
        GridState m_goal;
        Wind m_wind;
    };

    /// <summary>
    /// A windy grid world with a randomly varying wind.
    /// </summary>
    /// <typeparam name="W">The number of columns in the grid world.</typeparam>
    /// <typeparam name="H">The number of rows in the grid world.</typeparam>
    template <Width W, Height H>
    class RandomWindy : public Windy<W, H>
    {
    public:
        /// <summary>
        /// Creates a randomly windy grid world.
        /// </summary>
        /// <param name="start">- The starting position.</param>
        /// <param name="goal">- The ending position.</param>
        /// <param name="wind">- The strength of the wind in each column.</param>
        /// <param name="generator">- A random number generator.</param>
        RandomWindy(
            GridState start,
            GridState goal,
            Windy<W, H>::Wind wind,
            std::mt19937& generator)
        :
            Windy<W, H>{start, goal, wind},
            m_generator{generator}
        {}

    private:
        /// <summary>
        /// Applies a wind, that uniformly varies from [-1, 1], to some state.
        /// </summary>
        /// <param name="state">- The state to apply wind to.</param>
        /// <param name="wind">- The strength of the wind to apply.</param>
        /// <returns>The new state, pushed by the varying wind.</returns>
        GridState applyWind(GridState&& state, int wind) const override
        {
            GridState result{std::move(state)};

            if (wind != 0)
            {
                result.y() = static_cast<size_t>(
                    std::clamp(
                        static_cast<int>(result.y()) +
                        wind +
                        sample(m_windVariation, m_generator),
                        0,
                        H - 1));
            }

            return result;
        }

        inline static const std::set<int> m_windVariation{-1, 0, 1};

        std::mt19937& m_generator;
    };
}