#include <format>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include <matplot/matplot.h>

#include "introRL/afUtils.hpp"
#include "introRL/iteration/subplotters.hpp"
#include "introRL/iteration/types.hpp"

namespace irl::iteration
{
    PolicyValueSubplotter PolicyValueSubplotter::make(
        PlotSize plotSize,
        unsigned columns,
        unsigned lotSize,
        unsigned lotTickInterval,
        ActionLimits actionLimits,
        PolicyActionFn policyActionFn)
    {
        auto lotTickIndices{
            std::views::iota(unsigned{0})
            | std::views::take(lotSize)
            | std::views::filter(
                [=](unsigned i)
                {
                    return i % lotTickInterval == 0;
                })};

        auto lotTickLocations{
            lotTickIndices
            | std::views::transform(
                [](unsigned i)
                {
                    return i + 1;
                })};

        auto lotTickLabels{
            lotTickIndices
            | std::views::transform(
                [](auto index)
                {
                    return std::format("{}", index);
                })};

        auto f{matplot::figure(true)};
        f->size(plotSize.width, plotSize.height);

        return PolicyValueSubplotter{M{
            .columns{columns},
            .lotSize{lotSize},
            .actionLimits{
                static_cast<double>(actionLimits.low),
                static_cast<double>(actionLimits.high)},
            .policyActionFn{policyActionFn},
            .actionTickLocations{
                std::views::iota(actionLimits.low, actionLimits.high + 1)
                | std::ranges::to<std::vector<double>>()},
            .lotTickLocations{lotTickLocations | std::ranges::to<std::vector<double>>()},
            .lotTickLabels{lotTickLabels | std::ranges::to<std::vector>()}
        }};
    }

    PolicyValueSubplotter::PolicyValueSubplotter(M m) : m{std::move(m)} {}

    void PolicyValueSubplotter::show()
    {
        matplot::show();
    }

    void PolicyValueSubplotter::plot(const iteration::Policy& policy)
    {
        matplot::subplot(2, m.columns, m.column);

        matplot::image(toMatrix<int>(reshape(m.policyActionFn(policy))));

        matplot::colorbar()
            .limits(m.actionLimits)
            .tick_values(m.actionTickLocations);

        setupAxes("policy");
    }

    void PolicyValueSubplotter::plot(const iteration::StateValue& stateValue)
    {
        matplot::subplot(2, m.columns, size_t{m.columns} + m.column);

        matplot::image(
            toMatrix<float>(reshape(stateValue.unwrap<iteration::StateValue>())));

        matplot::colorbar()
            .limits_mode_auto(true)
            .ticklabels_mode(true);

        if (m.column == 0)
        {
            matplot::ylabel("Cars at lot A");
            matplot::xlabel("Cars at lot B");
        }

        setupAxes("state value");

        ++m.column;
    }

    void PolicyValueSubplotter::setupAxes(std::string_view title)
    {
        auto ax{matplot::gca()};

        auto oldWidth{ax->width()};

        ax->width(oldWidth * 1.2);

        const auto& oldPosition{ax->position()};

        ax->position({
            oldPosition[0] - oldWidth * .1f,
            oldPosition[1] - oldWidth * .1f,
            oldPosition[2],
            oldPosition[3]});

        ax->title_enhanced(false);
        matplot::title(
            std::format("{} {}", title, m.column + 1) +
            (m.column == m.columns - 1 ? " (optimal)" : ""));
        matplot::xticks(m.lotTickLocations);
        matplot::xticklabels(m.lotTickLabels);
        matplot::yticks(m.lotTickLocations);
        matplot::yticklabels(m.lotTickLabels);

        ax->y_axis().reverse(false);
    }

    af::array PolicyValueSubplotter::reshape(const af::array& data)
    {
        return af::moddims(data, af::dim4{m.lotSize, m.lotSize});
    }

    ValueIterationSubplotter ValueIterationSubplotter::make(
        PlotSize plotSize,
        unsigned columns,
        StateCount nStates,
        std::span<const unsigned> plotIterations,
        std::span<const unsigned> valueXTicks,
        PolicyActionFn policyActionFn)
    {
        auto f{matplot::figure(true)};
        f->size(plotSize.width, plotSize.height);

        return ValueIterationSubplotter{M{
            .columns{columns},
            .nStates{nStates.unwrap<StateCount>()},
            .plotIterations{std::from_range, plotIterations},
            .valueXTicks{std::from_range, valueXTicks},
            .policyActionFn{policyActionFn}
        }};
    }

    ValueIterationSubplotter::ValueIterationSubplotter(M m) : m{std::move(m)} {}

    void ValueIterationSubplotter::setupAxes(
        std::string_view title,
        std::span<const double> policyXTicks,
        std::span<const double> policyYTicks)
    {
        auto vAx{valueAx()};

        vAx->title(title);
        vAx->title_enhanced(false);
        vAx->x_axis().tick_values(m.valueXTicks);
        vAx->x_axis().limits({0., static_cast<double>(m.nStates + 1)});

        auto legend{matplot::legend(vAx)};
        legend->box(false);
        legend->location(matplot::legend::general_alignment::bottomright);
        legend->font_size(legend->font_size() * 0.8);

        if (m.column == 0)
        {
            vAx->y_axis().label("state values");
        }

        auto pAx{policyAx()};

        pAx->x_axis().tick_values(policyXTicks | std::ranges::to<std::vector>());
        pAx->y_axis().tick_values(policyYTicks | std::ranges::to<std::vector>());

        if (m.column == 0)
        {
            pAx->y_axis().label("optimal policy");
        }
    }

    void ValueIterationSubplotter::plot(const iteration::StateValue& stateValue)
    {
        auto prePlot{toVector<double>(stateValue.unwrap<iteration::StateValue>())};

        auto ax{valueAx()};

        if (m.plotIterations.contains(m.count))
        {
            auto stairs{matplot::stairs(ax, prePlot)};
            stairs->display_name(std::format("value iteration {}", m.count));
            stairs->stair_style(matplot::stair::stair_style::histogram);
        }

        if (++m.count == 1)
        {
            matplot::hold(ax, matplot::on);
        }
    }

    void ValueIterationSubplotter::plot(const iteration::Policy& policy)
    {
        matplot::hold(valueAx(), matplot::off);

        matplot::bar(
            policyAx(),
            toVector<double>(m.policyActionFn(policy).as(f64)),
            0.5);

        m.count = 0;
        ++m.column;
    }

    void ValueIterationSubplotter::show()
    {
        matplot::show();
    }

    matplot::axes_handle ValueIterationSubplotter::policyAx() const
    {
        return matplot::subplot(2, m.columns, size_t{m.columns} + m.column);
    }

    matplot::axes_handle ValueIterationSubplotter::valueAx() const
    {
        return matplot::subplot(2, m.columns, m.column);
    }
}