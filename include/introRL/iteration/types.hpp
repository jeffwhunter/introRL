#pragma once

#include <arrayfire.h>
#include <stronk/stronk.h>

#include "introRL/types.hpp"

namespace irl::iteration
{
    /// <summary>
    /// An array of action indices.
    /// </summary>
    struct ActionIndices : twig::stronk<ActionIndices, af::array>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// An array of action indices, one per state, which represent each action to
    /// take in each state.
    /// </summary>
    struct Policy : twig::stronk<Policy, af::array, AFOperators>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// An array of values, one per state, which represent the future reward available in
    /// each state.
    /// </summary>
    struct StateValue : twig::stronk<StateValue, af::array, AFOperators>
    {
        using stronk::stronk;
    };
}