#include <ranges>
#include <vector>

#include <matplot/matplot.h>

#include "introRL/charts.hpp"

namespace irl::charts
{
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
        const std::vector<double>& optimalTicks)
    {
        auto hRewardAxes{matplot::subplot(2, columns, rewardIndex)};
        hRewardAxes->font_size(fontSize);
        matplot::title(hRewardAxes, title);
        matplot::hold(hRewardAxes, matplot::on);
        matplot::ylim(hRewardAxes, {0, rewardTicks.back()});
        matplot::yticks(hRewardAxes, rewardTicks);
        matplot::xticks(hRewardAxes, xTicks);

        auto hOptimalityAxes{matplot::subplot(2, columns, optimalityIndex)};
        hOptimalityAxes->font_size(fontSize);
        matplot::hold(hOptimalityAxes, matplot::on);
        matplot::ylim(hOptimalityAxes, {0, optimalTicks.back()});
        matplot::yticks(hOptimalityAxes, optimalTicks);
        matplot::xticks(hOptimalityAxes, xTicks);

        const auto x{matplot::iota(1, xTicks.back())};

        for (const auto& [name, r, o] : std::views::zip(names, rewards, optimality))
        {
            matplot::plot(hRewardAxes, x, r)->display_name(name);
            matplot::plot(hOptimalityAxes, x, o);
        }
    }
}