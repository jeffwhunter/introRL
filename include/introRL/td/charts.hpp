#pragma once

#include <string>
#include <vector>

#include "introRL/td/algorithm.hpp"

namespace irl::td
{
    /// <summary>
    /// The dimensions of a chart showing SARSA results.
    /// </summary>
    struct ChartDimensions
    {
        /// <summary>
        /// The width of the chart.
        /// </summary>
        unsigned width{};

        /// <summary>
        /// The height of the chart.
        /// </summary>
        unsigned height{};
    };

    /// <summary>
    /// Renders a chart and saves the results to a file.
    /// </summary>
    /// <param name="dimensions">- The dimensions of the chart.</param>
    /// <param name="results">- The SARSA results to render.</param>
    /// <param name="names">- The names of each SARSA result.</param>
    /// <param name="fileName">- The file to render into.</param>
    /// <param name="xLim">- The maximimum timestep to chart, or 0 to chart all.</param>
    void chartSarsaToFile(
        ChartDimensions dimensions,
        const std::vector<SarsaResult>& results,
        const std::vector<std::string>& names,
        const std::string_view fileName,
        double xLim = 0);
}