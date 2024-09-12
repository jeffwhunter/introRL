#pragma once

#include <arrayfire.h>
#include <stronk/prefabs.h>
#include <stronk/stronk.h>
#include <stronk/unit.h>

#include "introRL/math/array.hpp"
#include <introRL/skills.hpp>

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
    /// The number of actions available in some learning process.
    /// </summary>
    struct ActionCount : twig::stronk_default_unit<ActionCount, unsigned>
    {
        using stronk_default_unit::stronk_default_unit;
    };

    /// <summary>
    /// Inclusive interval which defines possible actions.
    /// </summary>
    struct ActionLimits
    {
        int low;
        int high;
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
    /// The size of some overall chart.
    /// </summary>
    struct PlotSize
    {
        unsigned width;
        unsigned height;
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
    /// The total number of runs in some learning process.
    /// </summary>
    struct RunCount : twig::stronk_default_unit<RunCount, unsigned>
    {
        using stronk_default_unit::stronk_default_unit;
    };

    /// <summary>
    /// The number of states available in some learning process.
    /// </summary>
    struct StateCount : twig::stronk_default_unit<StateCount, unsigned, AddsInto<int>::Skill, DivsInto<unsigned>::Skill>
    {
        using stronk_default_unit::stronk_default_unit;
    };

    /// <summary>
    /// Units counting steps in a some learning process.
    /// </summary>
    struct StepCount : twig::stronk_default_unit<
        StepCount,
        unsigned,
        AddsInto<int>::Skill,
        CanIncrement,
        SubtractsInto<int>::Skill>
    {
        using stronk_default_unit::stronk_default_unit;
    };

    /// <summary>
    /// The ratio of RunCount / ParameterCount; that is, the number of runs to execute
    /// for each input parameter in some parallel learning process.
    /// </summary>
    using RunsPerParameter = decltype(RunCount{} / ParameterCount{});
}