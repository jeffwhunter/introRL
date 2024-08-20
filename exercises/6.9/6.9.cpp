#include <array>
#include <format>
#include <print>
#include <random>
#include <ranges>
#include <set>
#include <string>
#include <vector>

#include <blend2d.h>
#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>
#include <matplot/matplot.h>

#include <introRL/td/agents.hpp>
#include <introRL/td/algorithm.hpp>
#include <introRL/td/concepts.hpp>
#include <introRL/td/environments.hpp>
#include <introRL/td/renderers.hpp>
#include <introRL/td/types.hpp>
#include <introRL/ticker.hpp>

using namespace irl;
using namespace irl::td;

constexpr Width WIDTH{10};
constexpr Height HEIGHT{7};

const Alpha A{.5};
const Epsilon E{.1};

constexpr StepCount N_STEPS{100'000};

constexpr size_t PROGRESS_WIDTH{50};
constexpr size_t PROGRESS_TICKS{50};
constexpr auto TICK_RATE{N_STEPS.unwrap<StepCount>() / PROGRESS_TICKS};

constexpr unsigned FIGURE_WIDTH{1'000};
constexpr unsigned FIGURE_HEIGHT{500};

constexpr double X_LIM{12'000};
constexpr size_t TILE{30};
constexpr float FONT_S{15.f};
constexpr auto CHART_NAME{"chart.jpeg"};

struct Setup
{
    std::string name{};
    std::set<Action> actions{};
    indicators::Color barColour{};
    BLRgba32 pathColour{};
};

const auto SETUPS{
    std::to_array<Setup>({
        {
            "nesw",
            {
                Action::make(0, -1),
                Action::make(1, 0),
                Action::make(0, 1),
                Action::make(-1, 0)
            },
            indicators::Color::blue,
            BLRgba32{0xFF0000FF}
        },
        {
            "king's moves",
            {
                Action::make(0, -1),
                Action::make(1, 0),
                Action::make(0, 1),
                Action::make(-1, 0),
                Action::make(1, -1),
                Action::make(1, 1),
                Action::make(-1, 1),
                Action::make(-1, -1)
            },
            indicators::Color::red,
            BLRgba32{0xFFFF0000}
        },
        {
            "king's + wait",
            {
                Action::make(0, -1),
                Action::make(1, 0),
                Action::make(0, 1),
                Action::make(-1, 0),
                Action::make(1, -1),
                Action::make(1, 1),
                Action::make(-1, 1),
                Action::make(-1, -1),
                Action::make()
            },
            indicators::Color::yellow,
            BLRgba32{0xFFCCCC00}
        }})};

static indicators::ProgressBar makeBar(indicators::Color colour, std::string_view title)
{
    using namespace indicators::option;

    return indicators::ProgressBar{
        MaxProgress{PROGRESS_TICKS},
        ForegroundColor{colour},
        BarWidth{PROGRESS_WIDTH},
        Start{"["},
        Fill{"="},
        Lead{">"},
        Remainder{" "},
        End{"]"},
        PrefixText{std::format("{:>20}", title)},
        ShowRemainingTime{true}};
}

static auto run(
    CEnvironment auto& environment,
    CAgent auto& agent,
    const Setup& setup)
{
    auto bar{makeBar(setup.barColour, setup.name)};

    bar.set_progress(0);

    SarsaController<N_STEPS> controller{A, setup.actions};

    auto result{
        controller.sarsa(
            environment,
            agent,
            Ticker<TICK_RATE>{[&] { bar.tick(); }})};

    if (!bar.is_completed())
    {
        bar.mark_as_completed();
    }

    return result;
}

static void chartResults(
    const std::vector<SarsaResult>& results,
    const std::vector<std::string>& names)
{
    auto hFigure{matplot::figure(true)};
    hFigure->size(FIGURE_WIDTH, FIGURE_HEIGHT);

    matplot::xlim({0, X_LIM});
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
    matplot::save(CHART_NAME);
    matplot::save("");
}

int main()
{
    std::mt19937 generator{std::random_device{}()};

    EGreedy agent{E, generator};

    indicators::show_console_cursor(false);

    Environment<WIDTH, HEIGHT> environment{
        State::make(0, 3),
        State::make(7, 3),
        {0, 0, 0, -1, -1, -1, -2, -2, -1, 0}};

    auto results{
        SETUPS
        | std::views::transform(
            [&](const Setup& setup)
            {
                return run(environment, agent, setup);
            })
        | std::ranges::to<std::vector>()};

    indicators::show_console_cursor(true);

    chartResults(
        results,
        SETUPS
        | std::views::transform(
            [](const Setup& setup) { return setup.name; })
        | std::ranges::to<std::vector>());

    BLImage chart{};
    if (chart.readFromFile(CHART_NAME) != BL_SUCCESS)
    {
        std::println(std::cerr, "Can't read chart file");
        return 0;
    }

    BLImage image{FIGURE_WIDTH, FIGURE_HEIGHT, BL_FORMAT_PRGB32};
    BLContext context{image};

    context.clearAll();

    context.blitImage(BLPointI{0, 0}, chart);

    BLFontFace face{};
    if (face.createFromFile("font.ttf") != BL_SUCCESS)
    {
        std::println(std::cerr, "Can't find font.ttf");
        return 1;
    }

    EGreedy demoAgent{Epsilon{0}, generator};

    BLFont font{};
    font.createFromFace(face, FONT_S);

    renderDemos(
        context,
        BLPoint{200, 75},
        Layout{
            .grid{TILE},
            .columns{WIDTH.unwrap<Width>()},
            .rows{HEIGHT.unwrap<Height>()},
            .xTextOffset{.325},
            .yTextOffset{.65}},
        environment,
        std::views::zip(SETUPS, results)
        | std::views::transform(
            [&](auto&& setupResult)
            {
                auto&& [setup, result] {setupResult};

                return demo<StepCount{100}>(
                    setup.actions,
                    result.q,
                    environment,
                    demoAgent);
            })
        | std::ranges::to<std::vector>(),
        font,
        SETUPS
        | std::views::transform([](const Setup& s) { return s.pathColour; })
        | std::ranges::to<std::vector>());

    context.end();

    image.writeToFile("6.9.png");

    return 0;
}