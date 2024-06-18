#pragma once

#include <string>
#include <vector>

namespace irl::charts
{
    /// <summary>
    /// Make two charts for a number of runs of a learning algorithm: one for the average
    /// reward, and another for the average chance of taking the best action, on each
    /// step; both will be subplots of some larger plot.
    /// </summary>
    /// <param name="title">- The title of these evaluations.</param>
    /// <param name="rewards">- The reward sets to plot.</param>
    /// <param name="optimality">- The optimality sets to plot.</param>
    /// <param name="names">- The name of each set.</param>
    /// <param name="columns">- The number of columns in the overall plot.</param>
    /// <param name="rewardIndex">- The index of the reward subplot.</param>
    /// <param name="optimalityIndex">- The index of the optimality subplot.</param>
    /// <param name="fontSize">- The plot's font size.</param>
    /// <param name="xTicks">- Ticks on the x-axis.</param>
    /// <param name="rewardTicks">- Ticks on the reward axis.</param>
    /// <param name="optimalTicks">- Ticks on the optimality axis.</param>
    void subplotRewardAndOptimal(
        std::string title,
        const std::vector<std::vector<float>>& rewards,
        const std::vector<std::vector<float>>& optimality,
        const std::vector<std::string>& names,
        unsigned columns,
        unsigned rewardIndex,
        unsigned optimalityIndex,
        float fontSize,
        const std::vector<double>& xTicks,
        const std::vector<double>& rewardTicks,
        const std::vector<double>& optimalTicks);
}