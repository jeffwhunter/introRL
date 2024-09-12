#include <array>
#include <string>
#include <utility>
#include <vector>

#include <matplot/matplot.h>

#include "introRL/td/subplotters.hpp"
#include "introRL/types.hpp"

namespace irl::td
{
    NStepSubplotter NStepSubplotter::make(
        PlotSize plotSize,
        unsigned columns,
        std::string_view xTitle,
        std::string_view yTitle,
        std::array<double, 2> xLimits,
        std::array<double, 2> yLimits,
        double legendFontSize)
    {
        matplot::figure(true)->size(plotSize.width, plotSize.height);

        return NStepSubplotter{M{
            .columns{columns},
            .xTitle{xTitle},
            .yTitle{yTitle},
            .xLimits{xLimits},
            .yLimits{yLimits},
            .legendFontSize{legendFontSize}
        }};
    }

    NStepSubplotter::NStepSubplotter(M m) : m{std::move(m)} {}

    void NStepSubplotter::setupAxes(std::string_view title)
    {
        ++m.column;
        m.count = 0;

        auto ax{axes()};

        ax->title(title);

        ax->xlim(m.xLimits);
        ax->ylim(m.yLimits);

        auto legend{matplot::legend(ax)};
        legend->box(false);
        legend->font_size(m.legendFontSize);

        if (m.column == 0)
        {
            ax->x_axis().label(m.xTitle);
            ax->y_axis().label(m.yTitle);
        }
    }

    void NStepSubplotter::plot(
        const std::vector<double>& x,
        const std::vector<double>& y,
        std::string_view title)
    {
        auto ax{axes()};

        matplot::plot(ax, x, y)->display_name(title);

        if (++m.count == 1)
        {
            matplot::hold(ax, matplot::on);
        }
    }

    matplot::axes_handle NStepSubplotter::axes() const
    {
        return matplot::subplot(1, m.columns, m.column);
    }
}