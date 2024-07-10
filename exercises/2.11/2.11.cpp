#include <array>
#include <cmath>
#include <format>
#include <random>
#include <ranges>
#include <set>
#include <type_traits>

#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>
#include <matplot/matplot.h>

#include <introRL/agents.hpp>
#include <introRL/algorithm.hpp>
#include <introRL/environments.hpp>
#include <introRL/results.hpp>

using namespace indicators::option;
using namespace irl;
using namespace irl::bandit;
using namespace irl::bandit::agents;
using namespace irl::bandit::algorithm;

constexpr unsigned FIGURE_WIDTH{1'000};
constexpr unsigned FIGURE_HEIGHT{500};
constexpr unsigned FONT_SIZE{7};

constexpr unsigned N_ACTIONS{10};
constexpr unsigned N_RUNS_PER_PARAMETER{2'000};
constexpr unsigned N_STEPS{200'000};
constexpr unsigned START_MEASURE_STEP{100'000};

constexpr unsigned PROGRESS_WIDTH{50};
constexpr unsigned PROGRESS_TICKS{10};
constexpr unsigned PROGRESS_FREQ{N_STEPS / PROGRESS_TICKS};

constexpr float ALPHA{.1};
constexpr float WALK_SIZE{.01};
constexpr float PARAMETER_BASE{2};

constexpr auto X_SCALE_POWERS{std::views::iota(-7, 3)};

template <typename T>
requires std::is_floating_point_v<T>
constexpr auto expParameters(float parameterBase)
{
    return std::views::transform(
        [=](int power) { return static_cast<T>(std::pow(parameterBase, power)); }
    );
}

template <typename T>
constexpr auto makeParameters(
    float parameterBase,
    std::pair<T, T> exponentRange)
{
    const auto sorted{std::minmax(exponentRange.first, exponentRange.second)};

    return
        std::views::iota(sorted.first, sorted.second)
        | expParameters<float>(parameterBase)
        | std::ranges::to<std::vector>();
}

constexpr auto toFractions(float parameterBase)
{
    return std::views::transform(
        [=](int power)
        {
            return power >= 0 ?
                std::format("{}", std::pow(parameterBase, power)) :
                std::format("1/{}", std::pow(parameterBase, -power));
        });
}

using Environment = environments::Walking<WALK_SIZE>;
using Result = results::RollingRewards<START_MEASURE_STEP>;

struct ExperimentSetup
{
    std::string barTitle;
    indicators::Color barColour;

    std::string plotTitle;
    std::pair<double, double> plotTitlePos;
    matplot::color plotColour;

    std::string parameterSymbol;

    using LearnFunction =
        decltype(&Bandits::learn<EpsilonGreedyAverage, Environment, Result>);

    LearnFunction learn;

    std::pair<int, int> exponentRange;
};

const auto SETUPS{std::to_array<ExperimentSetup>({
    {
        "ega",
        indicators::Color::magenta,
        "e-greedy<1/N>",
        {std::pow(PARAMETER_BASE, -6.75), 4.25},
        matplot::color::magenta,
        "e",
        &Bandits::learn<EpsilonGreedyAverage, Environment, Result>,
        { -7, -1 }
    }, {
        "egc",
        indicators::Color::red,
        "e-greedy<.1>",
        {std::pow(PARAMETER_BASE, -6.75), 5.5},
        matplot::color::red,
        "e",
        &Bandits::learn<EpsilonGreedy<ALPHA>, Environment, Result>,
        {-7, -1}
    }, {
        " op",
        indicators::Color::unspecified,
        "op-greedy<.1>",
        {std::pow(PARAMETER_BASE, 0), 5.5},
        matplot::color::black,
        "q0",
        &Bandits::learn<Optimistic<ALPHA>, Environment, Result>,
        {-2, 3}
    }, {
        "ucb",
        indicators::Color::cyan,
        "UCB",
        {std::pow(PARAMETER_BASE, -2), 4.25},
        matplot::color::blue,
        "c",
        &Bandits::learn<UpperConfidence, Environment, Result>,
        {-4, 3}
    }, {
        " gb",
        indicators::Color::green,
        "gradient",
        {std::pow(PARAMETER_BASE, 0), 3.25},
        matplot::color::green,
        "a",
        &Bandits::learn<GradientBaseline, Environment, Result>,
        {-5, 3}
    }})};

int main()
{
    af::getDefaultRandomEngine().setSeed(std::random_device{}());

    auto hFigure{matplot::figure(true)};
    hFigure->size(FIGURE_WIDTH, FIGURE_HEIGHT);
    hFigure->font_size(FONT_SIZE);

    const auto xScale{
        X_SCALE_POWERS
        | expParameters<double>(PARAMETER_BASE)
        | std::ranges::to<std::vector>()};

    matplot::xlim({xScale.front(), xScale.back()});

    matplot::xticks(xScale);

    matplot::xticklabels(
        X_SCALE_POWERS
        | toFractions(PARAMETER_BASE)
        | std::ranges::to<std::vector>());

    matplot::hold(matplot::on);

    indicators::show_console_cursor(false);

    const Bandits learner{
        ActionCount{N_ACTIONS},
        RunsPerParameter{N_RUNS_PER_PARAMETER},
        StepCount{N_STEPS}};

    for (const auto& setup : SETUPS)
    {
        indicators::ProgressBar bar{
            MaxProgress{PROGRESS_TICKS},
            ForegroundColor{setup.barColour},
            BarWidth{PROGRESS_WIDTH},
            Start{"["},
            Fill{"="},
            Lead{">"},
            Remainder{" "},
            End{"]"},
            PrefixText{setup.barTitle},
            ShowElapsedTime{true},
            ShowRemainingTime{true}};

        bar.set_progress(0);

        const auto parameters{makeParameters<int>(PARAMETER_BASE, setup.exponentRange)};

        unsigned stepCounter{0};
        const auto score{
            std::mem_fn(setup.learn)(
                learner,
                parameters,
                [&]
                {
                    if (++stepCounter % PROGRESS_FREQ == 0)
                    {
                        bar.tick();
                    }
                })};

        auto hPlot{matplot::semilogx(parameters, score)};

        hPlot->color(setup.plotColour);

        auto hText{
            matplot::text(
                setup.plotTitlePos.first,
                setup.plotTitlePos.second,
                std::format("{} ({})", setup.plotTitle, setup.parameterSymbol))};

        hText->color(setup.plotColour);
    }

    indicators::show_console_cursor(true);

    matplot::xlabel(
        SETUPS
        | std::views::transform(
            [](const ExperimentSetup& setup)
            {
                return setup.parameterSymbol;
            })
        | std::ranges::to<std::set>()
        | std::views::join_with(std::string{", "})
        | std::ranges::to<std::string>());

    matplot::ylabel(
        std::format("Average reward over steps [{}, {})", START_MEASURE_STEP, N_STEPS));

    hFigure->current_axes()->font_size(FONT_SIZE);

    matplot::hold(matplot::off);

    matplot::show();

    return 0;
}