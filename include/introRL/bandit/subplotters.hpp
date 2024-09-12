#pragma once

#include <ranges>
#include <string>
#include <vector>

#include <matplot/matplot.h>

#include "introRL/types.hpp"
#include "introRL/bandit/results.hpp"

namespace af { class array; }

namespace irl::bandit
{
    /// <summary>
    /// A subplotter that plots the rewards and chance of optimal action for a number of
    /// runs of a simple bandit algorithm.
    /// </summary>
    class RewardOptimalitySubplotter
    {
        using BanditRewards = bandit::RewardsAndOptimality::RewardsResult;
        using BanditOptimality = bandit::RewardsAndOptimality::OptimalityResult;

    public:

        /// <summary>
        /// Makes a RewardOptimalitySubplotter.
        /// </summary>
        /// <param name="plotSize">- The width and height of the overall plot.</param>
        /// <param name="columns">- The number of columns in the plot.</param>
        /// <param name="fontSize">- The font size of the plot.</param>
        /// <param name="names">- The names of the runs to be plotted.</param>
        /// <param name="xTicks">- The location of ticks on the x axis.</param>
        /// <param name="rewardTicks">
        /// - The location of ticks on the y axis of the rewards plots.
        /// </param>
        /// <param name="optimalityTicks">
        /// - The location of ticks on the y axis of the optimality plots,
        /// </param>
        /// <returns>A RewardOptimalitySubplotter.</returns>
        static RewardOptimalitySubplotter make(
            PlotSize plotSize,
            unsigned columns,
            float fontSize,
            std::ranges::range auto&& names,
            std::ranges::range auto&& xTicks,
            std::ranges::range auto&& rewardTicks,
            std::ranges::range auto&& optimalityTicks)
        {
            auto hFigure{matplot::figure(true)};
            hFigure->size(plotSize.width, plotSize.height);

            auto hLegend = matplot::legend(matplot::subplot(2, columns, 0), {});
            hLegend->box(false);
            hLegend->font_size(fontSize);
            hLegend->location(matplot::legend::general_alignment::topleft);

            return RewardOptimalitySubplotter{M{
                .columns{columns},
                .fontSize{fontSize},
                .names{
                    std::forward<decltype(names)>(names)
                    | std::ranges::to<std::vector<std::string>>()},
                .xTicks{
                    std::forward<decltype(xTicks)>(xTicks)
                    | std::ranges::to<std::vector<double>>()},
                .rewardTicks{
                    std::forward<decltype(rewardTicks)>(rewardTicks)
                    | std::ranges::to<std::vector<double>>()},
                .optimalityTicks{
                    std::forward<decltype(optimalityTicks)>(optimalityTicks)
                    | std::ranges::to<std::vector<double>>()}
            }};
        }

        /// <summary>
        /// Plots two subplots in one column of the overall plot.
        /// </summary>
        /// <param name="title">- The title of the column.</param>
        /// <param name="rewards">- The rewards of the runs to plot.</param>
        /// <param name="optimalities">
        /// - The chance of optimal action for the runs to plot.
        /// </param>
        void plot(
            std::string title,
            const BanditRewards& rewards,
            const BanditOptimality& optimalities);

        /// <summary>
        /// Shows the final plot.
        /// </summary>
        void show();

    private:
        struct M
        {
            const unsigned columns{};
            const float fontSize{};

            const std::vector<std::string> names{};

            const std::vector<double> xTicks{};
            const std::vector<double> rewardTicks{};
            const std::vector<double> optimalityTicks{};

            unsigned column{};
        } m;

        explicit RewardOptimalitySubplotter(M m);

        /// <summary>
        /// Plots some rewards on an upper subplot.
        /// </summary>
        /// <param name="title">- The title of the current subplot column.</param>
        /// <param name="rewards">- The rewards of the runs to plot.</param>
        void plot(std::string title, const BanditRewards& rewards);

        /// <summary>
        /// Plots some changes of optimal action on a lower subplot.
        /// </summary>
        /// <param name="optimalities">
        /// - The chance of optimal action for the runs to plot.
        /// </param>
        void plot(const BanditOptimality& optimalities);

        /// <summary>
        /// Sets up some common properties for both subplots.
        /// </summary>
        /// <param name="yTicks">
        /// - The positions of ticks on the y-axis of the subplot.
        /// </param>
        /// <param name="yLabel">- The label of the y-axis of the subplot.</param>
        void setupAxes(const std::vector<double>& yTicks, std::string_view yLabel);
    };
}