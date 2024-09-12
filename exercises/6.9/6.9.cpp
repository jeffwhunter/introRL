#include <array>
#include <random>
#include <string>
#include <vector>

#include <indicators/color.hpp>

#include <introRL/td/agents.hpp>
#include <introRL/td/charts.hpp>
#include <introRL/td/environments.hpp>
#include <introRL/td/renderers.hpp>
#include <introRL/td/types.hpp>
#include <introRL/td/utils.hpp>
#include <introRL/types.hpp>

using namespace irl;
using namespace irl::td;

constexpr Width WIDTH{10};
constexpr Height HEIGHT{7};

const Alpha A{1, 2};
const Epsilon E{.1};

constexpr StepCount N_STEPS{100'000};

constexpr ProgressWidth PROGRESS_WIDTH{50};
constexpr ProgressTicks PROGRESS_TICKS{50};
constexpr auto TICK_RATE{
    N_STEPS.unwrap<StepCount>() / PROGRESS_TICKS.unwrap<ProgressTicks>()};

constexpr unsigned FIGURE_WIDTH{1'000};
constexpr unsigned FIGURE_HEIGHT{500};

constexpr double X_LIM{12'000};
constexpr size_t TILE{30};
constexpr float FONT_SIZE{15.f};
constexpr auto CHART_NAME{"chart6.10.jpeg"};

struct Setup
{
    std::string name{};
    GridActions actions{};
    indicators::Color barColour{};
    BLRgba32 pathColour{};
};

const auto SETUPS{
    std::to_array<Setup>({
        {
            "nesw",
            {
                GridAction::make(0, -1),
                GridAction::make(1, 0),
                GridAction::make(0, 1),
                GridAction::make(-1, 0)
            },
            indicators::Color::blue,
            BLRgba32{0xFF0000FF}
        },
        {
            "king's moves",
            {
                GridAction::make(0, -1),
                GridAction::make(1, 0),
                GridAction::make(0, 1),
                GridAction::make(-1, 0),
                GridAction::make(1, -1),
                GridAction::make(1, 1),
                GridAction::make(-1, 1),
                GridAction::make(-1, -1)
            },
            indicators::Color::red,
            BLRgba32{0xFFFF0000}
        },
        {
            "king's + wait",
            {
                GridAction::make(0, -1),
                GridAction::make(1, 0),
                GridAction::make(0, 1),
                GridAction::make(-1, 0),
                GridAction::make(1, -1),
                GridAction::make(1, 1),
                GridAction::make(-1, 1),
                GridAction::make(-1, -1),
                GridAction::make()
            },
            indicators::Color::yellow,
            BLRgba32{0xFFCCCC00}
        }})};

int main()
{
    std::mt19937 generator{std::random_device{}()};

    EGreedy agent{E, generator};

    Windy<WIDTH, HEIGHT> environment{
        GridState::make(0, 3),
        GridState::make(7, 3),
        {0, 0, 0, -1, -1, -1, -2, -2, -1, 0}};

    auto results{
        SETUPS
        | std::views::transform(
            [&](const Setup& setup)
            {
                return learnSarsa<N_STEPS>(
                    A,
                    environment,
                    agent,
                    setup.actions,
                    setup.name,
                    setup.barColour);
            })
        | std::ranges::to<std::vector>()};

    const auto names{
        SETUPS
        | std::views::transform(
            [](const Setup& setup) { return setup.name; })
        | std::ranges::to<std::vector>()};

    EGreedy demoAgent{Epsilon{0}, generator};
    const auto episodes{
        std::views::zip(SETUPS, results)
        | std::views::transform(
            [&](auto&& setupResult)
            {
                auto&& [setup, result] {setupResult};

                return episode<StepCount{100}>(
                    setup.actions,
                    result.q,
                    environment,
                    demoAgent);
            })};

    const auto colours{
        SETUPS
        | std::views::transform([](const Setup& s) { return s.pathColour; })};

    makeImage(
        "6.9.png",
        environment,
        results,
        names,
        episodes,
        colours,
        ChartDimensions{.width{FIGURE_WIDTH}, .height{FIGURE_HEIGHT}},
        Layout{
            .grid{TILE},
            .columns{WIDTH.unwrap<Width>()},
            .rows{HEIGHT.unwrap<Height>()},
            .xTextOffset{.325},
            .yTextOffset{.65}},
        X_LIM,
        FONT_SIZE);

    return 0;
}