#pragma once

#include <arrayfire.h>
#include <stronk/prefabs.h>
#include <stronk/stronk.h>
#include <stronk/unit.h>

#include "introRL/math/array.hpp"

namespace irl
{
    namespace detail
    {
        /// <summary>
        /// Ensures ALinearActions contain linearized column indices.
        /// </summary>
        struct LinearActionsModel : public af::array
        {
            explicit LinearActionsModel(
                const af::array& columnIndices,
                bool linearize = true);
        };
    }

    /// <summary>
    /// A skill that ensures some type can be incremented.
    /// </summary>
    /// <typeparam name="StronkT">The type using the skill.</typeparam>
    template <class StronkT>
    struct CanIncrement
    {
        /// <summary>
        /// Increments a StronkT.
        /// </summary>
        /// <param name="self">- The StronkT being inremented.</param>
        /// <returns>The StronkT being incremented.</returns>
        constexpr StronkT& operator++(this auto&& self)
        {
            ++self.unwrap<StronkT>();
            return self;
        }
    };

    /// <summary>
    /// A skill that ensures two objects can use operators that produce arrayfire arrays.
    /// af::array.
    /// </summary>
    /// <typeparam name="StronkT">The type of objects to operate on.</typeparam>
    template <class StronkT>
    struct AFOperators
    {
        /// <summary>
        /// Equates two objects in some way that produces an af::array.
        /// </summary>
        /// <param name="lhs">The left hand side of the equation.</param>
        /// <param name="rhs">The right hand side of the equation.</param>
        /// <returns>An af::array representing how equal the two sides are.</returns>
        constexpr friend auto operator==(const StronkT& lhs, const StronkT& rhs) noexcept
            -> af::array
        {
            static_assert(!std::is_floating_point_v<class StronkT::underlying_type>);
            return lhs.template unwrap<StronkT>() == rhs.template unwrap<StronkT>();
        }

        /// <summary>
        /// Discerns two objects in some way that produces an af::array.
        /// </summary>
        /// <param name="lhs">The left hand side of the discernment.</param>
        /// <param name="rhs">The right hand side of the discernment.</param>
        /// <returns>An af::array representing how equal the two sides are.</returns>
        constexpr friend auto operator!=(const StronkT& lhs, const StronkT& rhs) noexcept
            -> af::array
        {
            static_assert(!std::is_floating_point_v<class StronkT::underlying_type>);
            return lhs.template unwrap<StronkT>() != rhs.template unwrap<StronkT>();
        }

        /// <summary>
        /// Subtracts two objects in some way that produces an af::array.
        /// </summary>
        /// <param name="lhs">The left hand side of the subtraction.</param>
        /// <param name="rhs">The right hand side of the subtraction.</param>
        /// <returns>An af::array representing how different the two sides are.</returns>
        constexpr friend auto operator-(const StronkT& lhs, const StronkT& rhs) noexcept
            -> af::array
        {
            static_assert(!std::is_floating_point_v<class StronkT::underlying_type>);
            return lhs.template unwrap<StronkT>() - rhs.template unwrap<StronkT>();
        }
    };

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

                return StronkT{StronkT::template underlying_type{static_cast<T>(ts)...}};
            }
        };
    };

    /// <summary>
    /// Generates a skill or mixin that allows some type to have it's elements accessed
    /// by .x() and .y().
    /// </summary>
    /// <typeparam name="X">The index of the x element.</typeparam>
    /// <typeparam name="Y">The index of the y element.</typeparam>
    template <size_t X, size_t Y>
    struct XY
    {
        static_assert(X == 0 && Y == 1 || X == 1 && Y == 0, "<0, 1> or <1, 0> only");

        /// <summary>
        /// A skill that allows some type to have it's elements accessed by .x() and
        /// .y().
        /// </summary>
        /// <typeparam name="StronkT">The type using this skill.</typeparam>
        template <class StronkT>
        struct Skill
        {
            /// <summary>
            /// Returns a reference to the x element.
            /// </summary>
            /// <param name="self">- The StronkT being accessed.</param>
            /// <returns>A reference to the x element in the StronkT.</returns>
            [[nodiscard]] constexpr decltype(auto) x(this auto& self)
            {
                return self.template unwrap<StronkT>()[X];
            }

            /// <summary>
            /// Returns a reference to the y element.
            /// </summary>
            /// <param name="self">- The StronkT being accessed.</param>
            /// <returns>A reference to the y element in the StronkT.</returns>
            [[nodiscard]] constexpr decltype(auto) y(this auto& self)
            {
                return self.template unwrap<StronkT>()[Y];
            }
        };
    };

    /// <summary>
    /// The number of actions available in some learning process.
    /// </summary>
    struct ActionCount : twig::stronk_default_unit<ActionCount, unsigned>
    {
        using stronk_default_unit::stronk_default_unit;
    };

    /// <summary>
    /// The number of episodes to execute in some learning process.
    /// </summary>
    struct EpisodeCount : twig::stronk_default_unit<EpisodeCount, unsigned>
    {
        using stronk_default_unit::stronk_default_unit;
    };

    /// <summary>
    /// The size of some dimension in a tensor.
    /// </summary>
    struct Extent : twig::stronk<Extent, unsigned>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// The index of some sequential data.
    /// </summary>
    struct Index : twig::stronk<Index, unsigned>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// The axis along which some data is indexed.
    /// </summary>
    struct IndexAxis : twig::stronk<IndexAxis, unsigned>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// An array of actions, one per agent, suitable for indexing af::arrays.
    /// </summary>
    struct LinearActions : twig::stronk<
        LinearActions,
        detail::LinearActionsModel,
        AFOperators,
        twig::can_forward_constructor_args>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// The number of input parameters for some parallel learning process.
    /// </summary>
    struct ParameterCount : twig::stronk_default_unit<ParameterCount, unsigned>
    {
        using stronk_default_unit::stronk_default_unit;
    };

    /// <summary>
    /// The width of a progress bar.
    /// </summary>
    struct ProgressWidth : twig::stronk_default_unit<ProgressWidth, unsigned>
    {
        using stronk_default_unit::stronk_default_unit;
    };

    /// <summary>
    /// The number of calls it takes to fill up a progress bar.
    /// </summary>
    struct ProgressTicks : twig::stronk_default_unit<ProgressTicks, unsigned>
    {
        using stronk_default_unit::stronk_default_unit;
    };

    /// <summary>
    /// The rank of a tensor.
    /// </summary>
    struct Rank : twig::stronk<Rank, unsigned>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// An array of values, one per agent, where adjacent equal elements imply the input
    /// parameters at those indices have been duplicated from the same original.
    /// </summary>
    struct ReductionKeys : twig::stronk<ReductionKeys, af::array>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// The total number of runs in some parallel learning process.
    /// </summary>
    struct RunCount : twig::stronk_default_unit<RunCount, unsigned>
    {
        using stronk_default_unit::stronk_default_unit;
    };

    /// <summary>
    /// The number of states available in some learning process.
    /// </summary>
    struct StateCount : twig::stronk_default_unit<StateCount, unsigned>
    {
        using stronk_default_unit::stronk_default_unit;
    };

    /// <summary>
    /// The number of timesteps to execute in some learning process.
    /// </summary>
    struct StepCount : twig::stronk_default_unit<StepCount, unsigned, CanIncrement>
    {
        using stronk_default_unit::stronk_default_unit;
    };

    /// <summary>
    /// The ratio of RunCount / ParameterCount; that is, the number of runs to execute
    /// for each input parameter in some parallel learning process.
    /// </summary>
    using RunsPerParameter = decltype(RunCount{} / ParameterCount{});
}