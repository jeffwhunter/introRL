#include <ranges>
#include <string>
#include <vector>

#include <matplot/matplot.h>

#include "introRL/td/algorithm.hpp"
#include "introRL/td/charts.hpp"

namespace irl::td
{
    void chartSarsaToFile(
        ChartDimensions dimensions,
        const std::vector<SarsaResult>& results,
        const std::vector<std::string>& names,
        const std::string_view fileName,
        double xLim)
    {
        auto hFigure{matplot::figure(true)};
        hFigure->size(dimensions.width, dimensions.height);

        if (xLim != 0)
        {
            matplot::xlim({0, xLim});
        }
        matplot::xlabel("Time Steps");
        matplot::ylabel("Episodes");

        matplot::hold(matplot::on);

        matplot::legend(names);

        for (const auto& result : results)
        {
            matplot::plot(
                result.episodes
                | std::views::transform(
                    [](const StepCount& s) { return s.unwrap<StepCount>(); })
                | std::ranges::to<std::vector<double>>(),
                std::views::iota(0U, result.episodes.size())
                | std::ranges::to<std::vector<double>>());
        }

        matplot::hold(matplot::off);

        // Matplot has a bug where it won't close file handles until the next op.
        // This is needed to close tempfile.png so it can be read into a BLImage.
        // As a consequence of failure it shows the chart.
        matplot::save(fileName.data());
        matplot::save("");
    }
}