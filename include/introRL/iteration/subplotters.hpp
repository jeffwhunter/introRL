#pragma once

#include <array>
#include <functional>
#include <set>
#include <span>
#include <string>
#include <vector>

#include <matplot/matplot.h>

#include "introRL/iteration/types.hpp"
#include "introRL/types.hpp"

namespace af { class array; }

namespace irl::iteration
{
    /// <summary>
    /// A subplotter that plots the policy and state value estimates for a run of the
    /// rental policy problem.
    /// </summary>
    class PolicyValueSubplotter
    {
        using PolicyActionFn = std::function<af::array(const iteration::Policy&)>;

    public:

        /// <summary>
        /// Makes a PolicyValueSubplotter
        /// </summary>
        /// <param name="plotSize">- The width and height of the overall plot.</param>
        /// <param name="columns">- The number of columns in the plot.</param>
        /// <param name="lotSize">
        /// - The side length of the policy and state values.
        /// </param>
        /// <param name="lotTickInterval">
        /// - How often to display ticks on the x and y axes.
        /// </param>
        /// <param name="actionLimits">- The bounds of possible actions.</param>
        /// <param name="policyActionFn">
        /// - A function that turns action indices into a number of cars moved between
        /// lots.
        /// </param>
        /// <returns>A PolicyValueSubplotter.</returns>
        static PolicyValueSubplotter make(
            PlotSize plotSize,
            unsigned columns,
            unsigned lotSize,
            unsigned lotTickInterval,
            ActionLimits actionLimits,
            PolicyActionFn policyActionFn);

        /// <summary>
        /// Plots a policy on an upper subplot.
        /// </summary>
        /// <param name="policy">- The policy to plot.</param>
        void plot(const iteration::Policy& policy);

        /// <summary>
        /// Plots an state value estimate on a lower subplot.
        /// </summary>
        /// <param name="stateValue">- The state value estimate to plot.</param>
        void plot(const iteration::StateValue& stateValue);

        /// <summary>
        /// Shows the final plot.
        /// </summary>
        void show();

    private:
        struct M
        {
            const unsigned columns{};
            const unsigned lotSize{};

            const std::array<double, 2> actionLimits{};
            const PolicyActionFn policyActionFn{};

            const std::vector<double> actionTickLocations{};
            const std::vector<double> lotTickLocations{};
            const std::vector<std::string> lotTickLabels{};

            unsigned column{};
        } m;

        explicit PolicyValueSubplotter(M m);

        /// <summary>
        /// Sets up some common properties for both subplots.
        /// </summary>
        /// <param name="title">- The y axis title.</param>
        void setupAxes(std::string_view title);

        /// <summary>
        /// Reshapes an array from {N*N} to {N, N}.
        /// </summary>
        /// <param name="data">The shape {N*N} array to reshape.</param>
        /// <returns>The shape {N, N} array that was reshaped.</returns>
        af::array reshape(const af::array& data);
    };

    /// <summary>
    /// Plots the optimal policy and state value estimates for a run of the coin flip problem.
    /// </summary>
    class ValueIterationSubplotter
    {
        using PolicyActionFn = std::function<af::array(const iteration::Policy&)>;

    public:
        /// <summary>
        /// Makes a ValueIterationSubplotter.
        /// </summary>
        /// <param name="plotSize">- The width and height of the overall plot.</param>
        /// <param name="columns">- The number of columns in the plot.</param>
        /// <param name="nStates">- The number of states.</param>
        /// <param name="plotIterations">- The iterations to plot.</param>
        /// <param name="valueXTicks">- The x ticks to plot.</param>
        /// <param name="policyActionFn">
        /// - A function that converts from indices to actions.
        /// </param>
        /// <returns>A ValueIterationSubplotter.</returns>
        static ValueIterationSubplotter make(
            PlotSize plotSize,
            unsigned columns,
            StateCount nStates,
            std::span<const unsigned> plotIterations,
            std::span<const unsigned> valueXTicks,
            PolicyActionFn policyActionFn);

        /// <summary>
        /// Sets up some common properties for both subplots.
        /// </summary>
        /// <param name="title">- The subplot's title.</param>
        /// <param name="policyXTicks">- The x axis ticks to plot for the policy.</param>
        /// <param name="policyYTicks">- The y axis ticks to plot for the policy.</param>
        void setupAxes(
            std::string_view title,
            std::span<const double> policyXTicks,
            std::span<const double> policyYTicks);

        /// <summary>
        /// Plots a state value iteration. Can handle multiple calls.
        /// </summary>
        /// <param name="stateValue">- A state value estimate for the coin flip problem.</param>
        void plot(const iteration::StateValue& stateValue);

        /// <summary>
        /// Plots an estimate of the optimal policy on the bottom plot, then moves the subplotter
        /// to the next column.
        /// </summary>
        /// <param name="policy">- The estimate of the optimal policy to plot.</param>
        void plot(const iteration::Policy& policy);

        /// <summary>
        /// Shows the final plot.
        /// </summary>
        void show();

    private:
        struct M
        {
            const unsigned columns{};
            const unsigned nStates{};
            const std::set<unsigned> plotIterations{};
            const std::vector<double> valueXTicks{};
            const PolicyActionFn policyActionFn{};

            unsigned column{};
            unsigned count{};
        } m;

        explicit ValueIterationSubplotter(M m);

        /// <summary>
        /// Gets the current axes for the policy plot.
        /// </summary>
        /// <returns></returns>
        matplot::axes_handle policyAx() const;

        /// <summary>
        /// Gets the current axes for the value plot.
        /// </summary>
        /// <returns></returns>
        matplot::axes_handle valueAx() const;
    };
}