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
        /// Ensures Actions contain linearized column indices.
        /// </summary>
        struct ActionsModel : public af::array
        {
            explicit ActionsModel(const af::array& columnIndices, bool linearize = true);
        };
    }

    /// <summary>
    /// A skill that ensures two objects of some type can be equated to produce an
    /// af::array.
    /// </summary>
    /// <typeparam name="StronkT"></typeparam>
    template <typename StronkT>
    struct can_arrayfire_equate
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
            static_assert(!std::is_floating_point_v<typename StronkT::underlying_type>);
            return lhs.template unwrap<StronkT>() == rhs.template unwrap<StronkT>();
        }
    };

    /// <summary>
    /// An array of actions, one per agent, suitable for indexing af::arrays.
    /// </summary>
    struct Actions : twig::stronk<
        Actions,
        detail::ActionsModel,
        can_arrayfire_equate,
        twig::can_forward_constructor_args>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// The number of actions available in some parallel learning process.
    /// </summary>
    struct ActionCount : twig::stronk_default_unit<ActionCount, unsigned>
    {
        using stronk_default_unit::stronk_default_unit;
    };

    /// <summary>
    /// An array of input parameters, one per agent.
    /// </summary>
    struct DeviceParameters : twig::stronk<DeviceParameters, af::array>
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
    /// An array of values, one per agent, where adjacent equal elements imply the input
    /// parameters at those indices have been duplicated from the same original.
    /// </summary>
    struct ReductionKeys : twig::stronk<ReductionKeys, af::array>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// An array of rewards, one per agent.
    /// </summary>
    struct Rewards : twig::stronk<Rewards, af::array>
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