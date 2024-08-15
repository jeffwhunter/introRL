#pragma once

#include <functional>
#include <ranges>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <matplot/matplot.h>

#include "introRL/types.hpp"
#include "introRL/iteration/types.hpp"
#include "introRL/bandit/results.hpp"

namespace af { class array; }

namespace irl
{
    struct Size
    {
        unsigned width;
        unsigned height;
    };

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
        /// <param name="size">- The width and height of the overall plot.</param>
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
            Size size,
            unsigned columns,
            float fontSize,
            std::ranges::range auto&& names,
            std::ranges::range auto&& xTicks,
            std::ranges::range auto&& rewardTicks,
            std::ranges::range auto&& optimalityTicks)
        {
            auto hFigure{matplot::figure(true)};
            hFigure->size(size.width, size.height);

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

    struct Limits
    {
        int low;
        int high;
    };

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
        /// <param name="size">- The width and height of the overall plot.</param>
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
            Size size,
            unsigned columns,
            unsigned lotSize,
            unsigned lotTickInterval,
            Limits actionLimits,
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

    class ValueIterationSubplotter
    {
        using PolicyActionFn = std::function<af::array(const iteration::Policy&)>;

    public:

        static ValueIterationSubplotter make(
            Size size,
            unsigned columns,
            StateCount nStates,
            std::span<const unsigned> plotIterations,
            std::span<const unsigned> valueXTicks,
            PolicyActionFn policyActionFn);

        void setupAxes(
            std::string_view title,
            std::span<const double> policyXTicks,
            std::span<const double> policyYTicks);

        void plot(const iteration::StateValue& stateValue);
        void plot(const iteration::Policy& policy);

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

        matplot::axes_handle policyAx() const;
        matplot::axes_handle valueAx() const;
    };
}