#include <array>
#include <format>
#include <functional>
#include <random>
#include <ranges>

#include <arrayfire.h>
#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>
#include <matplot/matplot.h>

#include <introRL/agents.hpp>
#include <introRL/algorithm.hpp>
#include <introRL/charts.hpp>
#include <introRL/environments.hpp>
#include <introRL/results.hpp>

using namespace irl;
using namespace irl::bandit;
using namespace irl::bandit::agents;
using namespace irl::bandit::algorithm;
using namespace irl::bandit::environments;

constexpr unsigned FIGURE_WIDTH{1'000};
constexpr unsigned FIGURE_HEIGHT{500};
constexpr unsigned FONT_SIZE{5};
constexpr unsigned N_X_TICKS{5};

constexpr unsigned N_ACTIONS{10};
constexpr unsigned N_RUNS_PER_PARAMETER{2'000};
constexpr unsigned N_STEPS{10'000};

constexpr unsigned PROGRESS_WIDTH{50};
constexpr unsigned PROGRESS_TICKS{10};
constexpr unsigned PROGRESS_FREQ{N_STEPS / PROGRESS_TICKS};

constexpr float ALPHA{.1};
constexpr float WALK_SIZE{.01};

const std::vector<float> EPSILONS{0, .01, .1};
const auto EPSILON_NAMES{EPSILONS
    | std::views::transform(
        [](const auto e)
        {
            return std::format("e={}", e);
        })
    | std::ranges::to<std::vector>()};

const std::vector<double> X_TICKS{
    std::views::iota(0)
    | std::views::take(N_X_TICKS)
    | std::views::transform(
        [](const double tIndex)
        {
            return std::max(N_STEPS * tIndex / (N_X_TICKS - 1), 1.);
        })
    | std::ranges::to<std::vector>()};

const std::vector<double> REWARD_Y_TICKS{0, .5, 1, 1.5, 2, 2.5};
const std::vector<double> OPTIMALITY_Y_TICKS{0, .2, .4, .6, .8, 1};

using Result = results::RewardsAndOptimality;

struct ExperimentSetup
{
    std::string title;

    using LearnFunction =
        decltype(&Bandits::learn<EpsilonGreedyAverage, Stationary, Result>);

    LearnFunction learn;
};

const auto SETUPS{std::to_array<ExperimentSetup>({
    {
        "1/N step",
        &Bandits::learn<EpsilonGreedyAverage, Stationary, Result>
    }, {
        std::format("{} step", ALPHA),
        &Bandits::learn<EpsilonGreedy<ALPHA>, Stationary, Result>
    }, {
        "Walk, 1/N step",
        &Bandits::learn<EpsilonGreedyAverage, Walking<WALK_SIZE>, Result>
    }, {
        std::format("Walk, {} step", ALPHA),
        &Bandits::learn<EpsilonGreedy<ALPHA>, Walking<WALK_SIZE>, Result>
    }})};

int main()
{
    af::getDefaultRandomEngine().setSeed(std::random_device{}());

    auto hFigure{matplot::figure(true)};
    hFigure->size(FIGURE_WIDTH, FIGURE_HEIGHT);

    const auto nSetups{SETUPS.size()};

    matplot::tiledlayout(2, nSetups);

    matplot::ylabel(matplot::subplot(2, nSetups, 0), "reward");
    matplot::ylabel(matplot::subplot(2, nSetups, nSetups), "optimal");

    size_t tCount{0};
    size_t bCount{nSetups};

    indicators::show_console_cursor(false);

    const Bandits learner{
        ActionCount{N_ACTIONS},
        RunsPerParameter{N_RUNS_PER_PARAMETER},
        StepCount{N_STEPS}};

    for (const auto& setup : SETUPS)
    {
        indicators::ProgressBar bar{
            indicators::option::MaxProgress{PROGRESS_TICKS},
            indicators::option::BarWidth{PROGRESS_WIDTH},
            indicators::option::Start{"["},
            indicators::option::Fill{"="},
            indicators::option::Lead{">"},
            indicators::option::Remainder{" "},
            indicators::option::End{"]"},
            indicators::option::PrefixText{setup.title},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true}};

        bar.set_progress(0);

        unsigned stepCounter{0};
        const auto score{
            std::mem_fn(setup.learn)(
                learner,
                EPSILONS,
                [&]
                {
                    if (++stepCounter % PROGRESS_FREQ == 0)
                    {
                        bar.tick();
                    }
                })};

        charts::subplotRewardAndOptimal(
            setup.title,
            score.rewards,
            score.optimality,
            EPSILON_NAMES,
            nSetups,
            tCount++,
            bCount++,
            FONT_SIZE,
            X_TICKS,
            REWARD_Y_TICKS,
            OPTIMALITY_Y_TICKS);
    }

    indicators::show_console_cursor(true);

    auto hLegend = matplot::legend(matplot::subplot(2, nSetups, 0), {});
    hLegend->box(false);
    hLegend->font_size(FONT_SIZE);
    hLegend->location(matplot::legend::general_alignment::topleft);

    matplot::show();

    return 0;
}