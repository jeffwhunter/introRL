#pragma once

namespace irl::td
{
    /// <summary>
    /// Generates a skill or mixin that allows some type to multiply into another type,
    /// producing a result with type equal to the other type.
    /// </summary>
    /// <typeparam name="T">The other type.</typeparam>
    template <class T>
    struct MultipliesInto
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
            [[nodiscard]] constexpr friend T operator*(const StronkT& lhs, T rhs)
            {
                return lhs.template unwrap<StronkT>() * rhs;
            }

            /// <summary>
            /// Multiplies a StronkT with a T.
            /// </summary>
            /// <param name="lhs">- The left side of the multiplication.</param>
            /// <param name="rhs">- The right side of the multiplication.</param>
            /// <returns>The product of lhs and rhs.</returns>
            [[nodiscard]] constexpr friend T operator*(T lhs, const StronkT& rhs)
            {
                return rhs * lhs;
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
}