# pragma once

#include <stronk/stronk.h>

#include <array>
#include <concepts>
#include <utility>

#include "introRL/math/array.hpp"

namespace irl::monte
{
    /// <summary>
    /// Generates a skill or mixin that allows some type to return another of itself when
    /// added to some target type. Said another way, the target type is the delta of the
    /// type using the generated skill, so a user of that skill is an antidelta of the
    /// target type.
    /// </summary>
    /// <typeparam name="T">
    /// The target type that is a delta of the type using this skill.
    /// </typeparam>
    template <twig::stronk_like T>
    struct Antidelta
    {
        /// <summary>
        /// A skill that allows some type to return another of itself when added to some
        /// T. Said another way, T is the delta of the type using this skill, so a user
        /// of this skill is an antidelta of T.
        /// </summary>
        /// <typeparam name="StronkT">The type using this skill.</typeparam>
        template <class StronkT>
        struct Skill
        {
            /// <summary>
            /// Adds a StronkT to a stronk_like T to produce a StronkT.
            /// </summary>
            /// <param name="lhs">- Some StronkT.</param>
            /// <param name="rhs">- Some other stronk_like.</param>
            /// <returns>The sum of lhs and rhs as a StronkT.</returns>
            constexpr friend StronkT operator+(const StronkT& lhs, const T& rhs) noexcept
            {
                using namespace irl::math;
                return StronkT{
                    lhs.template unwrap<StronkT>() + rhs.template unwrap<T>()};
            }
        };
    };

    /// <summary>
    /// Generates a skill or mixin that allows some type to be able to be made with a
    /// static functions that take specific numbers of some type of argument.
    /// </summary>
    /// <typeparam name="T">The type of arguments to the make function.</typeparam>
    /// <typeparam name="...Ns">The possible number of arguments.</typeparam>
    template <class T, size_t ... Ns>
    struct Makeable
    {
        /// <summary>
        /// A skill that allows some type to be made with a static function, as described
        /// above.
        /// </summary>
        /// <typeparam name="StronkT">
        /// The type that will be allowed to be made with static functions.
        /// </typeparam>
        template <class StronkT>
        struct Skill
        {
            /// <summary>
            /// Makes a StronkT out of a number of T.
            /// </summary>
            /// <typeparam name="...Ts">
            /// The actual types of the passed parameters.
            /// </typeparam>
            /// <param name="...ts">
            /// - The arguments that will be passed to construction.
            /// </param>
            /// <returns>A StronkT.</returns>
            template <class ... Ts>
            [[nodiscard]] static constexpr StronkT make(Ts ... ts)
            {
                constexpr auto nTs{sizeof...(ts)};
                static_assert(
                    ((Ns == nTs) || ...),
                    "Wrong number of args");

                return StronkT{{static_cast<T>(ts)...}};
            }
        };
    };

    /// <summary>
    /// A change in velocity in a grid world.
    /// </summary>
    struct Action :
        twig::stronk<
            Action,
            std::array<int, 2>,
            twig::can_equate,
            twig::can_index,
            twig::can_iterate,
            twig::can_order,
            Makeable<int, 0, 2>::template Skill>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// A change in position in a grid world.
    /// </summary>
    struct Velocity :
        twig::stronk<
            Velocity,
            std::array<int, 2>,
            twig::can_equate,
            twig::can_index,
            twig::can_order,
            Antidelta<Action>::template Skill,
            Makeable<int, 0, 2>::template Skill>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// A position in a grid world.
    /// </summary>
    struct Position :
        twig::stronk<
            Position,
            std::array<size_t, 2>,
            twig::can_equate,
            twig::can_index,
            twig::can_order,
            Antidelta<Velocity>::template Skill,
            Makeable<size_t, 0, 2>::template Skill>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// The state of a race car in a grid world.
    /// </summary>
    struct State
    {
        Position position{Position::make()};
        Velocity velocity{Velocity::make()};

        auto operator<=>(const State&) const = default;
    };

}