#pragma once

#include <vector>

namespace introRL::reinforcement
{
    struct Evaluation;
}

namespace introRL::charts
{
    /// <summary>
    /// Make two charts for a number of runs of a learning algorithm: one for the average
    /// reward, and another for the average chance of taking the best action, on each
    /// step. These charts will be subplots of some larger plot.
    /// </summary>
    /// <param name="title">- The title of these evaluations.</param>
    /// <param name="columns">- The number of columns in the larger plot.</param>
    /// <param name="rewardIndex">- The subplot index of the rewards chart.</param>
    /// <param name="optimalityIndex">
    /// - The subplot index of the optimality chart.
    /// </param>
    /// <param name="fontSize">- The font size of the axes.</param>
    /// <param name="steps">- How many steps the evaluations went on for.</param>
    /// <param name="evaluations">
    /// - Evaluations measuring the performance of some agents.
    /// </param>
    void evaluations(
        std::string title,
        unsigned columns,
        unsigned rewardIndex,
        unsigned optimalityIndex,
        float fontSize,
        unsigned steps,
        const std::vector<reinforcement::Evaluation>& evaluations);
}