#pragma once

#include <arrayfire.h>
#include <stronk/stronk.h>

#include "introRL/basicTypes.hpp"

namespace irl::bandit
{
    /// <summary>
    /// An array of input parameters, one per agent.
    /// </summary>
    struct DeviceParameters : twig::stronk<DeviceParameters, af::array>
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
}