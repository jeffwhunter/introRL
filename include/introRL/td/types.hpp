#pragma once

#include <concepts>
#include <set>

#include <mp++/mp++.hpp>
#include <mp++/real.hpp>
#include <stronk/prefabs.h>
#include <stronk/stronk.h>

#include "introRL/math/sparse.hpp"
#include "introRL/types.hpp"

namespace irl::td
{
    /// <summary>
    /// Generates a skill or mixin that allows some type to multiply into another type,
    /// producing a result with type equal to the other type.
    /// </summary>
    /// <typeparam name="T">The other type.</typeparam>
    template <class T>
    struct MultipliesUnder
    {
        /// <summary>
        /// A skill or mixin that allows some type to multiply with Ts, producing a T.
        /// </summary>
        /// <typeparam name="StronkT">The type using this skill.</typeparam>
        template <class StronkT>
        struct Skill
        {
            /// <summary>
            /// Multiplies a StronkT with a T.
            /// </summary>
            /// <param name="lhs">- The left side of the multiplication.</param>
            /// <param name="rhs">- The right side of the multiplication.</param>
            /// <returns>The product of lhs and rhs.</returns>
            [[nodiscard]] friend T operator*(const StronkT& lhs, T rhs)
            {
                return lhs.template unwrap<StronkT>() * rhs;
            }

            /// <summary>
            /// Multiplies a StronkT with a T.
            /// </summary>
            /// <param name="lhs">- The left side of the multiplication.</param>
            /// <param name="rhs">- The right side of the multiplication.</param>
            /// <returns>The product of lhs and rhs.</returns>
            [[nodiscard]] friend T operator*(T lhs, const StronkT& rhs)
            {
                return rhs * lhs;
            }
        };
    };

    /// <summary>
    /// Generates a skill or mixin that allows some type to subtract into another type,
    /// producing a result with type equal to the other type.
    /// </summary>
    /// <typeparam name="T">The other type.</typeparam>
    template <class T>
    struct SubtractsUnder
    {
        /// <summary>
        /// A skill or mixin that allows some type to subtract with Ts, producing a T.
        /// </summary>
        /// <typeparam name="StronkT">The type using this skill.</typeparam>
        template <class StronkT>
        struct Skill
        {
            /// <summary>
            /// Subtracts a T from a StronkT.
            /// </summary>
            /// <param name="lhs">- The left side of the subtraction.</param>
            /// <param name="rhs">- The right side of the subtraction.</param>
            /// <returns>The difference between lhs and rhs.</returns>
            [[nodiscard]] friend T operator-(const StronkT& lhs, T rhs)
            {
                return lhs.template unwrap<StronkT>() - rhs;
            }

            /// <summary>
            /// Subtracts a StronkT from a T.
            /// </summary>
            /// <param name="lhs">- The left side of the subtraction.</param>
            /// <param name="rhs">- The right side of the subtraction.</param>
            /// <returns>The difference between lhs and rhs.</returns>
            [[nodiscard]] friend T operator-(T lhs, const StronkT& rhs)
            {
                return lhs - rhs.template unwrap<StronkT>();
            }
        };
    };

    /// <summary>
    /// Generates a skill or mixin that allows some type to order with some other type.
    /// </summary>
    /// <typeparam name="T">The other type.</typeparam>
    template <class T>
    struct OrdersWith
    {
        /// <summary>
        /// A skill or mixin that allows some type to order with Ts.
        /// </summary>
        /// <typeparam name="StronkT">The type using this skill.</typeparam>
        template <class StronkT>
        struct Skill
        {
            /// <summary>
            /// Compares a StronkT with a T.
            /// </summary>
            /// <param name="lhs">- The left side of the comparison.</param>
            /// <param name="rhs">- The right side of the comparison.</param>
            /// <returns>How lhs compares to rhs.</returns>
            constexpr friend auto operator<=>(const StronkT& lhs, T rhs) noexcept
            {
                return static_cast<T>(lhs.template unwrap<StronkT>()) <=> rhs;
            }
        };
    };

    /// <summary>
    /// A change in position in a grid world.
    /// </summary>
    struct Action :
        twig::stronk<
            Action,
            std::array<int, 2>,
            twig::can_equate,
            twig::can_order,
            Makeable<int, 0, 2>::Skill,
            XY<0, 1>::Skill>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// A position in a grid world.
    /// </summary>
    struct State :
        twig::stronk<
            State,
            std::array<size_t, 2>,
            twig::can_equate,
            twig::can_order,
            Antidelta<Action>::Skill,
            Makeable<size_t, 0, 2>::Skill,
            XY<0, 1>::Skill>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// The number of rows in a grid world.
    /// </summary>
    struct Height
        : twig::stronk<
            Height,
            size_t,
            OrdersWith<int>::Skill,
            SubtractsUnder<int>::Skill>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// The number of columns in a grid world.
    /// </summary>
    struct Width
        : twig::stronk<
            Width,
            size_t,
            twig::can_multiply,
            OrdersWith<int>::Skill>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// The step size of a reinforcement learning process.
    /// </summary>
    struct Alpha :
        twig::stronk<
            Alpha,
            mppp::real,
            MultipliesUnder<mppp::real>::Skill>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// The probability that some agent will take exploratory actions.
    /// </summary>
    struct Epsilon : twig::stronk<Epsilon, mppp::real>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// A set of actions.
    /// </summary>
    using Actions = std::set<Action>;

    /// <summary>
    /// An action value table.
    /// </summary>
    using Q = math::SparseMatrix<State, Action, mppp::real>;
}