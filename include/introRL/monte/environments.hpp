#pragma once

#include <mdspan>
#include <random>
#include <ranges>
#include <set>
#include <utility>

#include "introRL/cartesian.hpp"
#include "introRL/math/array.hpp"
#include "introRL/monte/types.hpp"
#include "introRL/stats.hpp"

namespace irl::monte
{
    /// <summary>
    /// Possible spaces on a discrete race track.
    /// </summary>
    enum TrackTile
    {
        /// <summary>
        /// A road space.
        /// </summary>
        _,
        /// <summary>
        /// A wall space.
        /// </summary>
        X,
        /// <summary>
        /// A starting space.
        /// </summary>
        S,
        /// <summary>
        /// A final space.
        /// </summary>
        F
    };

    /// <summary>
    /// An environment that simulates racing a car around a discrete track.
    /// </summary>
    /// <typeparam name="H">The height of the track.</typeparam>
    /// <typeparam name="W">The width of the track.</typeparam>
    template <size_t H, size_t W>
    class Environment
    {
    public:
        using Track = std::mdspan<const TrackTile, std::extents<size_t, H, W>>;

        /// <summary>
        /// Makes an Environment.
        /// </summary>
        /// <param name="track">
        /// - An mdspan over the tiles that make up the track to simulate.
        /// </param>
        /// <param name="generator">- A random number generator.</param>
        /// <returns>An Environment.</returns>
        static Environment<H, W> make(Track track, std::mt19937& generator)
        {
            return Environment{M{
                .track{track},
                .generator{generator},
                .starts{
                    mdIndices(H, W)
                    | std::views::filter([=](auto&& index) { return track[index] == S; })
                    | std::views::transform([](auto&& index) { return Position{index}; })
                    | std::ranges::to<std::set>()}}};
        }

        /// <summary>
        /// Returns the next state of some race car acting in the environment.
        /// </summary>
        /// <param name="state">- The state the race car is in before acting.</param>
        /// <param name="action">- The action the race car takes in state.</param>
        /// <returns>
        /// The next state the race car finds itself in after doing action in state.
        /// </returns>
        State step(const State& state, const Action& action)
        {
            using namespace irl::math;

            const auto& oldPosition{state.position};
            const auto& oldVelocity{state.velocity};

            auto newVelocity{oldVelocity + action};

            auto path{
                math::interp(
                    oldPosition.unwrap<Position>(),
                    newVelocity.unwrap<Velocity>())
                | std::views::transform([](auto&& p) { return Position(p); })};

            for (auto&& position : path)
            {
                if (!inBounds(position))
                {
                    return reset();
                }

                if (finish(position))
                {
                    return State{.position{position}, .velocity{newVelocity}};
                }
            }

            return State{.position{oldPosition + newVelocity}, .velocity{newVelocity}};
        }

        /// <summary>
        /// Returns a random starting state.
        /// </summary>
        /// <returns>
        /// A state with a random starting position and zero velocity.
        /// </returns>
        State reset()
        {
            return State{.position{sample(m.starts, m.generator)}};
        }

        /// <summary>
        /// Checks if some state is a final state.
        /// </summary>
        /// <param name="state">- The state to check.</param>
        /// <returns>
        /// True if the tile at state's position is F, false otherwise.
        /// </returns>
        bool done(const State& state)
        {
            return finish(state.position);
        }

        /// <summary>
        /// Returns all starting positions in the environment.
        /// </summary>
        /// <returns>A set of all starting positions.</returns>
        const std::set<Position>& starts() const { return m.starts; }

    private:
        struct M
        {
            Track track;
            std::mt19937& generator;
            std::set<Position> starts;
        } m;

        explicit Environment(M m) : m{std::move(m)} {}

        /// <summary>
        /// Checks if some position is in bounds.
        /// </summary>
        /// <param name="p">- The position to check.</param>
        /// <returns>
        /// True if p is on the track somewhere, and the tile at that location isn't X.
        /// </returns>
        bool inBounds(const Position& p)
        {
            if (p.y() < 0 || p.y() >= H || p.x() < 0 || p.x() >= W)
            {
                return false;
            }

            if (m.track[p.unwrap<Position>()] == X)
            {
                return false;
            }

            return true;
        }

        /// <summary>
        /// Checks if some position is a final position.
        /// </summary>
        /// <param name="p">- The position to check</param>
        /// <returns>True if the tile at p is F, false otherwise.</returns>
        bool finish(const Position& p)
        {
            return m.track[p.unwrap<Position>()] == F;
        }
    };
}