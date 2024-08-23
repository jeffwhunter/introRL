#include <iostream>
#include <memory>
#include <print>
#include <random>
#include <ranges>

#include <indicators/color.hpp>
#include <indicators/cursor_control.hpp>

#include <introRL/types.hpp>
#include <introRL/monte/agents.hpp>
#include <introRL/monte/algorithm.hpp>
#include <introRL/monte/renderers.hpp>
#include <introRL/ticker.hpp>
#include <introRL/types.hpp>
#include <introRL/utils.hpp>

using namespace irl;
using namespace irl::monte;

constexpr size_t N_GENERATIONS{8};
constexpr auto RENDER_GENERATIONS{std::to_array({0, 1, 7})};

constexpr EpisodeCount N_EPISODES{10'000'000};
constexpr StepCount MAX_EPISODE_STEPS{1'000};

auto PATH_COLOURS{
    std::to_array({
        std::to_array({BLRgba32{0xCCFFFF00}, BLRgba32{0xCC888800}}),
        std::to_array({BLRgba32{0xCCFF00FF}, BLRgba32{0xCC880088}}),
        std::to_array({BLRgba32{0xCC00FFFF}, BLRgba32{0xCC008888}}),
        std::to_array({BLRgba32{0xCC0000FF}, BLRgba32{0xCC000088}}),
        std::to_array({BLRgba32{0xCC00FF00}, BLRgba32{0xCC008800}}),
        std::to_array({BLRgba32{0xCCFF0000}, BLRgba32{0xCC880000}})
    })};

auto BAR_COLOURS{
    std::to_array({
        indicators::Color::red,
        indicators::Color::blue,
        indicators::Color::green})};

constexpr ProgressWidth PROGRESS_WIDTH{50};
constexpr ProgressTicks PROGRESS_TICKS{50};
constexpr auto tickRate{
    N_EPISODES.unwrap<EpisodeCount>()
    / PROGRESS_TICKS.unwrap<ProgressTicks>()};

constexpr auto T_DATA{
    std::to_array({
        X, X, X, _, _, _, _, _, _, _, _, _, _, _, _, _, F,
        X, X, _, _, _, _, _, _, _, _, _, _, _, _, _, _, F,
        X, X, _, _, _, _, _, _, _, _, _, _, _, _, _, _, F,
        X, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, F,
        _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, F,
        _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, F,
        _, _, _, _, _, _, _, _, _, _, X, X, X, X, X, X, X,
        _, _, _, _, _, _, _, _, _, X, X, X, X, X, X, X, X,
        _, _, _, _, _, _, _, _, _, X, X, X, X, X, X, X, X,
        _, _, _, _, _, _, _, _, _, X, X, X, X, X, X, X, X,
        _, _, _, _, _, _, _, _, _, X, X, X, X, X, X, X, X,
        _, _, _, _, _, _, _, _, _, X, X, X, X, X, X, X, X,
        _, _, _, _, _, _, _, _, _, X, X, X, X, X, X, X, X,
        _, _, _, _, _, _, _, _, _, X, X, X, X, X, X, X, X,
        X, _, _, _, _, _, _, _, _, X, X, X, X, X, X, X, X,
        X, _, _, _, _, _, _, _, _, X, X, X, X, X, X, X, X,
        X, _, _, _, _, _, _, _, _, X, X, X, X, X, X, X, X,
        X, _, _, _, _, _, _, _, _, X, X, X, X, X, X, X, X,
        X, _, _, _, _, _, _, _, _, X, X, X, X, X, X, X, X,
        X, _, _, _, _, _, _, _, _, X, X, X, X, X, X, X, X,
        X, _, _, _, _, _, _, _, _, X, X, X, X, X, X, X, X,
        X, _, _, _, _, _, _, _, _, X, X, X, X, X, X, X, X,
        X, X, _, _, _, _, _, _, _, X, X, X, X, X, X, X, X,
        X, X, _, _, _, _, _, _, _, X, X, X, X, X, X, X, X,
        X, X, _, _, _, _, _, _, _, X, X, X, X, X, X, X, X,
        X, X, _, _, _, _, _, _, _, X, X, X, X, X, X, X, X,
        X, X, _, _, _, _, _, _, _, X, X, X, X, X, X, X, X,
        X, X, _, _, _, _, _, _, _, X, X, X, X, X, X, X, X,
        X, X, _, _, _, _, _, _, _, X, X, X, X, X, X, X, X,
        X, X, X, _, _, _, _, _, _, X, X, X, X, X, X, X, X,
        X, X, X, _, _, _, _, _, _, X, X, X, X, X, X, X, X,
        X, X, X, S, S, S, S, S, S, X, X, X, X, X, X, X, X})};

constexpr size_t T_WIDTH{17};
constexpr size_t T_HEIGHT{32};

constexpr size_t T_SIZE{20};

constexpr static size_t imageSize(size_t nTiles, size_t tileSize)
{
    return (nTiles + 2) * tileSize;
}

constexpr float TEXT_S{15.f};

constexpr int SPRINT_SPEED{1};
constexpr int SPRINT_STOP{3};
constexpr int TURN_START{5};

constexpr int MAX_ACTION{1};
constexpr int MIN_ACTION{-1};

constexpr float P_EXPLORE{.01f};

constexpr Layout LAYOUT{
    .single{imageSize(T_WIDTH, T_SIZE), imageSize(T_HEIGHT, T_SIZE)},
    .tileSize{T_SIZE},
    .titleLocation{225, 375},
    .lengthLocation{225, 400},
    .pathOffset{.8, .8},
    .strokeWidth{2}};

int main()
{
    auto seed{std::random_device{}()};
    std::mt19937 generator{seed};

    af::getDefaultRandomEngine().setSeed(seed);

    using TEnv = Environment<T_HEIGHT, T_WIDTH>;

    TEnv::Track track{T_DATA.data()};

    auto environment{TEnv::make(track, generator)};

    auto explorer{Explorer::make(MIN_ACTION, MAX_ACTION, generator)};

    ExpertAgent teacher{SPRINT_SPEED, SPRINT_STOP, TURN_START, explorer};

    BLFontFace face{};
    BLResult result{face.createFromFile("font.ttf")};

    if (result != BL_SUCCESS)
    {
        std::println(std::cerr, "Can't find font.ttf");
        return 1;
    }

    BLFont font{};
    font.createFromFace(face, TEXT_S);

    std::vector<std::vector<Episode>> demos{};

    demos.emplace_back(demoAllStarts<MAX_EPISODE_STEPS>(teacher, environment));

    indicators::show_console_cursor(false);

    std::optional<TableAgent> student{};

    std::set<size_t> renderGenerations{std::from_range, RENDER_GENERATIONS};
    for (size_t i : std::views::iota(0U, N_GENERATIONS))
    {
        auto bar{
            makeBar(
                std::format("generation {}", i + 1),
                BAR_COLOURS[i % BAR_COLOURS.size()],
                PROGRESS_WIDTH,
                PROGRESS_TICKS)};

        bar.set_progress(0);

        const auto makeStudent{
            [&](CAgent auto&& agent)
            {
                return control<N_EPISODES, MAX_EPISODE_STEPS, P_EXPLORE>(
                    std::forward<decltype(agent)>(agent),
                    environment,
                    explorer,
                    generator,
                    Ticker<tickRate>{[&] { bar.tick(); }});
            }};

        student.emplace(
            student.has_value() ? makeStudent(student.value()) : makeStudent(teacher));

        if (renderGenerations.contains(i))
        {
            demos.emplace_back(
                demoAllStarts<MAX_EPISODE_STEPS>(student.value(), environment));
        }
    }

    indicators::show_console_cursor(true);

    std::vector<std::string> titles{"expert"};

    titles.append_range(
        renderGenerations
        | std::views::transform(
            [](size_t generation)
            {
                return std::format("generation {}", generation + 1);
            }));

    renderDemos(
        "5.12.png",
        LAYOUT,
        track,
        std::views::zip(demos, titles),
        std::span{PATH_COLOURS},
        font);

    return 0;
}