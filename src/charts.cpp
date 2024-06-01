#include <algorithm>
#include <vector>

#include <matplot/matplot.h>

#include "introRL/charts.hpp"
#include "introRL/reinforcement.hpp"

const unsigned X_TICKS{5u};
const double REWARD_MAX{2.5};
const double OPTIMALITY_MAX{1.};
const std::vector<double> REWARD_Y_TICKS{0, .5, 1, 1.5, 2, 2.5};
const std::vector<double> OPTIMALITY_Y_TICKS{0, .2, .4, .6, .8, 1};

namespace introRL::charts
{
    void evaluations(
        std::string title,
        unsigned columns,
        unsigned rewardIndex,
        unsigned optimalityIndex,
        float fontSize,
        unsigned steps,
        const std::vector<reinforcement::Evaluation>& evaluations)
    {
        auto increment{steps / (X_TICKS - 1)};

        std::vector<double> xTicks(X_TICKS);
        std::generate(
            xTicks.begin(),
            xTicks.end(),
            [&, n = 0]() mutable { return std::clamp(n++ * increment, 1u, steps); });

        auto hRewardAxes{matplot::subplot(2, columns, rewardIndex)};
        hRewardAxes->font_size(fontSize);
        matplot::title(hRewardAxes, title);
        matplot::hold(hRewardAxes, matplot::on);
        matplot::ylim(hRewardAxes, {0, REWARD_MAX});
        matplot::yticks(hRewardAxes, REWARD_Y_TICKS);
        matplot::xticks(hRewardAxes, xTicks);

        auto hOptimalityAxes{matplot::subplot(2, columns, optimalityIndex)};
        hOptimalityAxes->font_size(fontSize);
        matplot::hold(hOptimalityAxes, matplot::on);
        matplot::ylim(hOptimalityAxes, {0, OPTIMALITY_MAX});
        matplot::yticks(hOptimalityAxes, OPTIMALITY_Y_TICKS);
        matplot::xticks(hOptimalityAxes, xTicks);

        const auto x{matplot::iota(1, steps)};

        for (const auto& [name, rewards, optimality] : evaluations)
        {
            matplot::plot(hRewardAxes, x, rewards)->display_name(name);
            matplot::plot(hOptimalityAxes, x, optimality);
        }
    }
}