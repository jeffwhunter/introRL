#include <array>
#include <format>
#include <functional>
#include <random>
#include <ranges>

#include <arrayfire.h>
#include <indicators/color.hpp>
#include <indicators/cursor_control.hpp>
#include <matplot/matplot.h>

#include <introRL/bandit/agents.hpp>
#include <introRL/bandit/algorithm.hpp>
#include <introRL/bandit/environments.hpp>
#include <introRL/bandit/results.hpp>
#include <introRL/bandit/subplotters.hpp>
#include <introRL/ticker.hpp>
#include <introRL/utils.hpp>
#include <introRL/types.hpp>

using namespace irl;
using namespace irl::bandit;

PlotSize PLOT_SIZE{.width{1'500}, .height{750}};
constexpr unsigned FONT_SIZE{5};
constexpr unsigned N_X_TICKS{5};

constexpr unsigned N_ACTIONS{10};
constexpr unsigned N_RUNS_PER_PARAMETER{2'000};
constexpr unsigned N_STEPS{10'000};

constexpr ProgressWidth PROGRESS_WIDTH{50};
constexpr ProgressTicks PROGRESS_TICKS{10};
constexpr unsigned PROGRESS_FREQ{N_STEPS / PROGRESS_TICKS.unwrap<ProgressTicks>()};

constexpr float ALPHA{.1};
constexpr float WALK_SIZE{.01};

constexpr auto EPSILONS{std::to_array({0.f, .01f, .1f})};

constexpr auto EPSILON_NAMES{
    EPSILONS
    | std::views::transform(
        [](const auto e)
        {
            return std::format("e={}", e);
        })};

constexpr auto X_TICKS{
    std::views::iota(0)
    | std::views::take(N_X_TICKS)
    | std::views::transform(
        [](const double tIndex)
        {
            return std::max(N_STEPS * tIndex / (N_X_TICKS - 1), 1.);
        })};

constexpr auto REWARD_Y_TICKS{std::to_array({0., .5, 1., 1.5, 2., 2.5})};
constexpr auto OPTIMALITY_Y_TICKS{std::to_array({0., .2, .4, .6, .8, 1.})};

using Result = RewardsAndOptimality;

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

    indicators::show_console_cursor(false);

    const Bandits learner{
        ActionCount{N_ACTIONS},
        RunsPerParameter{N_RUNS_PER_PARAMETER},
        StepCount{N_STEPS}};

    auto plotter{
        RewardOptimalitySubplotter::make(
            PLOT_SIZE,
            SETUPS.size(),
            FONT_SIZE,
            EPSILON_NAMES,
            X_TICKS,
            REWARD_Y_TICKS,
            OPTIMALITY_Y_TICKS)};

    for (const auto& setup : SETUPS)
    {
        auto bar{
            makeBar(
                setup.title,
                indicators::Color::unspecified,
                PROGRESS_WIDTH,
                PROGRESS_TICKS)};

        bar.set_progress(0);

        const auto score{
            std::mem_fn(setup.learn)(
                learner,
                EPSILONS | std::ranges::to<std::vector<float>>(),
                Ticker<PROGRESS_FREQ>{[&] { bar.tick(); }})};

        plotter.plot(setup.title, score.rewards, score.optimality);
    }

    indicators::show_console_cursor(true);

    plotter.show();

    return 0;
}