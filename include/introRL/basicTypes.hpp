#pragma once

#include <arrayfire.h>
#include <stronk/prefabs.h>
#include <stronk/stronk.h>
#include <stronk/unit.h>

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
    /// The number of actions available in some learning process.
    /// </summary>
    struct ActionCount : twig::stronk_default_unit<ActionCount, unsigned>
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
    struct StepCount : twig::stronk_default_unit<StepCount, unsigned>
    {
        using stronk_default_unit::stronk_default_unit;
    };

    /// <summary>
    /// The ratio of RunCount / ParameterCount; that is, the number of runs to execute
    /// for each input parameter in some parallel learning process.
    /// </summary>
    using RunsPerParameter = decltype(RunCount{} / ParameterCount{});
}