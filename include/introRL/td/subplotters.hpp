#pragma once

#include <array>
#include <string>
#include <vector>

#include <matplot/matplot.h>

#include "introRL/types.hpp"

namespace af { class array; }

namespace irl::td
{
    /// <summary>
    /// A subplotter that plots the error in some state value estimates for some runs of a coin
    /// flipping random walk.
    /// </summary>
    class NStepSubplotter
    {
    public:
        /// <summary>
        /// Makes an NStepSubplotter.
        /// </summary>
        /// <param name="plotSize">- The width and height of the overall plot.</param>
        /// <param name="columns">- The number of columns in the plot.</param>
        /// <param name="xTitle">- The title of the x axis.</param>
        /// <param name="yTitle">- The title of the y axis.</param>
        /// <param name="xLimits">- The limits of the x axis.</param>
        /// <param name="yLimits">- The limits of the y axis.</param>
        /// <param name="legendFontSize">- The font size of the legend.</param>
        /// <returns>A NStepSubplotter.</returns>
        static NStepSubplotter make(
            PlotSize plotSize,
            unsigned columns,
            std::string_view xTitle,
            std::string_view yTitle,
            std::array<double, 2> xLimits,
            std::array<double, 2> yLimits,
            double legendFontSize);

        /// <summary>
        /// Sets up some common properties for both subplots.
        /// </summary>
        /// <param name="title">- The title of the subplot.</param>
        void setupAxes(std::string_view title);

        /// <summary>
        /// Plots the error of some state value estimate.
        /// </summary>
        /// <param name="x">- The alpha of the estimators.</param>
        /// <param name="y">- The error of those estimators.</param>
        /// <param name="title">- The title of this plot.</param>
        void plot(
            const std::vector<double>& x,
            const std::vector<double>& y,
            std::string_view title);

    private:
        struct M
        {
            const unsigned columns{};

            const std::string_view xTitle{};
            const std::string_view yTitle{};
            const std::array<double, 2> xLimits{};
            const std::array<double, 2> yLimits{};
            const double legendFontSize{};

            int column{-1};
            unsigned count{};
        } m;

        explicit NStepSubplotter(M m);

        /// <summary>
        /// Gets the axes for the current plot.
        /// </summary>
        /// <returns></returns>
        matplot::axes_handle axes() const;
    };
}