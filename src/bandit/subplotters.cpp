#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include <matplot/matplot.h>

#include "introRL/bandit/subplotters.hpp"

namespace irl::bandit
{
    RewardOptimalitySubplotter::RewardOptimalitySubplotter(M m) : m{std::move(m)} {}

    void RewardOptimalitySubplotter::plot(
        std::string title,
        const BanditRewards& rewards,
        const BanditOptimality& optimalities)
    {
        plot(title, rewards);
        plot(optimalities);

        ++m.column;
    }

    void RewardOptimalitySubplotter::show()
    {
        matplot::show();
    }

    void RewardOptimalitySubplotter::plot(
        std::string title,
        const BanditRewards& rewards)
    {
        matplot::subplot(2, m.columns, m.column);
        matplot::title(title);
        setupAxes(m.rewardTicks, "reward");

        const auto x{matplot::iota(1, m.xTicks.back())};
        for (const auto& [name, reward] : std::views::zip(m.names, rewards))
        {
            matplot::plot(x, reward)->display_name(name);
        }
    }

    void RewardOptimalitySubplotter::plot(const BanditOptimality& optimalities)
    {
        matplot::subplot(2, m.columns, size_t{m.columns} + m.column);
        setupAxes(m.optimalityTicks, "optimal");

        const auto x{matplot::iota(1, m.xTicks.back())};
        for (const auto& optimality : optimalities)
        {
            matplot::plot(x, optimality);
        }
    }
    
    void RewardOptimalitySubplotter::setupAxes(
        const std::vector<double>& yTicks,
        std::string_view yLabel)
    {
        auto ax{matplot::gca()};

        ax->font_size(m.fontSize);
        matplot::hold(ax, matplot::on);
        matplot::ylim(ax, {0, yTicks.back()});
        matplot::yticks(ax, yTicks);
        matplot::xticks(ax, m.xTicks);

        if (m.column == 0)
        {
            matplot::ylabel(ax, yLabel);
        }
    }
}