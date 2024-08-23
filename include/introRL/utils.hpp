#pragma once

#include <string>

#include <indicators/progress_bar.hpp>

#include "introRL/types.hpp"

namespace irl
{
    /// <summary>
    /// Makes a progress bar showing time remaining.
    /// </summary>
    /// <param name="title">- The title to show to the left of the bar.</param>
    /// <param name="colour">- The colour of the bar.</param>
    /// <param name="progressWidth">- The width of the bar on the screen.</param>
    /// <param name="proasgressTicks">- How many ticks it takes the bar to fill.</param>
    /// <returns>The progress bar itself.</returns>
    indicators::ProgressBar makeBar(
        std::string_view title,
        indicators::Color colour,
        ProgressWidth progressWidth,
        ProgressTicks proasgressTicks);
}