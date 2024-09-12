#include <algorithm>
#include <array>
#include <cmath>
#include <experimental/generator>
#include <format>
#include <random>
#include <ranges>
#include <unordered_map>
#include <utility>
#include <vector>

#include <print>

#include <indicators/color.hpp>
#include <indicators/cursor_control.hpp>
#include <matplot/matplot.h>

#include <introRL/math/error.hpp>
#include <introRL/ring.hpp>
#include <introRL/td/environments.hpp>
#include <introRL/td/nStep.hpp>
#include <introRL/td/subplotters.hpp>
#include <introRL/td/types.hpp>
#include <introRL/utils.hpp>

using namespace irl;

PlotSize PLOT_SIZE{.width{1'500}, .height{750}};
constexpr std::array<double, 2> X_LIMITS{.0, 1.};
constexpr std::array<double, 2> Y_LIMITS{.25, .55};
constexpr double LEGEND_FONT_SIZE{10.};

constexpr RunCount N_RUNS{100};
constexpr EpisodeCount N_EPISODES{10};

constexpr size_t NS[]{1, 2, 4, 8, 16};

constexpr StateCount N_STATES{19};

using Environment = td::Walk<N_STATES>;

constexpr Environment ENVIRONMENT{};

constexpr size_t N_ALPHAS{11};
constexpr size_t ALPHA_DENOMINATOR{N_ALPHAS - 1};

constexpr ProgressWidth PROGRESS_WIDTH{50};
constexpr ProgressTicks PROGRESS_TICKS{N_ALPHAS};

using StateValueMap = std::unordered_map<td::WalkState, mppp::real>;

template <StepCount N>
using NStepReturn = td::UpdatesWithNStepReturn<N, N_EPISODES, StateValueMap>;

template <StepCount N>
using SumTDError = td::UpdatesWithSumTDErrors<N, N_EPISODES, StateValueMap>;

const auto ALPHAS{
    std::views::iota(0U, N_ALPHAS)
    | std::views::transform([](unsigned n) { return td::Alpha{n, ALPHA_DENOMINATOR}; })
    | std::ranges::to<std::vector>()};

const auto ANSWERS{
    std::views::iota(0U, N_STATES.unwrap<StateCount>())
    | std::views::transform(
        [](unsigned n)
        {
            return std::pair{
                irl::td::WalkState{n + 1},
                td::rat1_t{n + 1, (N_STATES + 1) / 2} - 1};
        })
    | std::ranges::to<StateValueMap>()};

template <StepCount N_STEPS, template <StepCount N> class TUpdater>
double runExperiments(
    const td::Alpha& alpha,
    std::uniform_random_bit_generator auto& generator)
{
    td::NStepTD<N_STEPS, N_EPISODES, StateValueMap, TUpdater<N_STEPS>> nSteps{};

    double error{.0f};

    for (int _{0}; _ < N_RUNS.unwrap<RunCount>(); ++_)
    {
        nSteps.stateValues(
            alpha,
            ENVIRONMENT,
            [&](const auto& stateValues) { error += irl::math::rmse(stateValues, ANSWERS); },
            generator);
    }

    return
        error
        / (N_EPISODES.unwrap<EpisodeCount>() * N_RUNS.unwrap<RunCount>());
}

template <StepCount N_STEPS, template <StepCount N> class TUpdater>
std::vector<double> calculateResults(
    std::uniform_random_bit_generator auto& generator)
{
    auto bar{
        makeBar(
            std::format("N = {:3}", N_STEPS.unwrap<StepCount>()),
            indicators::Color::unspecified,
            PROGRESS_WIDTH,
            PROGRESS_TICKS)};

    bar.set_progress(0);

    auto result{
        ALPHAS
        | std::views::transform(
            [&](const td::Alpha& alpha)
            {
                bar.tick();
                return runExperiments<N_STEPS, TUpdater>(alpha, generator);
            })
        | std::ranges::to<std::vector>()};

    return result;
}

template <size_t ... Is>
void plotResults(
    std::uniform_random_bit_generator auto& generator,
    std::index_sequence<Is ...> const&)
{
    auto x{matplot::linspace(0, 1, N_ALPHAS)};

    auto plotter{
        td::NStepSubplotter::make(
            PLOT_SIZE,
            2,
            "alpha",
            std::format(
                "avg RMS error over {} states and {} episodes",
                N_STATES.unwrap<StateCount>(),
                N_EPISODES.unwrap<EpisodeCount>()),
            X_LIMITS,
            Y_LIMITS,
            LEGEND_FONT_SIZE)};

    plotter.setupAxes("nstep return");

    const auto returnResults{
        std::views::zip(
            std::vector{calculateResults<StepCount{NS[Is]}, NStepReturn>(generator)...},
            NS)};

    for (const auto& [y, n] : returnResults)
    {
        plotter.plot(x, y, std::format("n = {}", n));
    }

    plotter.setupAxes("sum td error");

    const auto errorResults{
        std::views::zip(
            std::vector{calculateResults<StepCount{NS[Is]}, SumTDError>(generator)...},
            NS)};

    for (const auto& [y, n] : errorResults)
    {
        plotter.plot(x, y, std::format("n = {}", n));
    }

    matplot::hold(matplot::off);

    matplot::show();
}

int main()
{
    std::mt19937 generator{std::random_device{}()};

    indicators::show_console_cursor(false);
    plotResults(generator, std::make_index_sequence<std::size(NS)>());
    indicators::show_console_cursor(true);

    return 0;
}