#include <experimental/generator>
#include <random>
#include <string>
#include <vector>

#include <introRL/td/agents.hpp>
#include <introRL/td/charts.hpp>
#include <introRL/td/concepts.hpp>
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

constexpr StepCount N_STEPS{10'000'000};

constexpr ProgressWidth PROGRESS_WIDTH{50};

constexpr ProgressTicks PROGRESS_TICKS{50};
constexpr auto TICK_RATE{
    N_STEPS.unwrap<StepCount>() / PROGRESS_TICKS.unwrap<ProgressTicks>()};

constexpr unsigned FIGURE_WIDTH{1'000};
constexpr unsigned FIGURE_HEIGHT{500};

constexpr double X_LIM{30'000};
constexpr size_t TILE{30};
constexpr float FONT_SIZE{15.f};
constexpr auto CHART_NAME{"chart6.9.jpeg"};

constexpr size_t N_EPISODES{100};
constexpr BLRgba32 PATH_COLOUR{0x110000FF};

const GridActions ACTIONS{
    GridAction::make(0, -1),
    GridAction::make(1, 0),
    GridAction::make(0, 1),
    GridAction::make(-1, 0),
    GridAction::make(1, -1),
    GridAction::make(1, 1),
    GridAction::make(-1, 1),
    GridAction::make(-1, -1)};

static std::experimental::generator<Episode> sarsaEpisodes(
    const Q& q,
    CSarsaEnvironment auto& environment,
    CSarsaAgent auto& agent)
{
    while (true)
    {
        co_yield episode<StepCount{100}>(ACTIONS, q, environment, agent);
    }
}

int main()
{
    std::mt19937 generator{std::random_device{}()};

    EGreedy agent{E, generator};

    RandomWindy<WIDTH, HEIGHT> environment{
        GridState::make(0, 3),
        GridState::make(7, 3),
        {0, 0, 0, -1, -1, -1, -2, -2, -1, 0},
        generator};

    auto result{
        learnSarsa<N_STEPS>(
            A,
            environment,
            agent,
            ACTIONS,
            "Running SARSA")};

    EGreedy demoAgent{Epsilon{0}, generator};

    result.episodes = result.episodes
        | std::views::filter(
            [](const StepCount& step) { return step.unwrap<StepCount>() < X_LIM; })
        | std::ranges::to<std::vector>();

    makeImage(
        "6.10.png",
        environment,
        std::vector{result},
        std::vector{std::string{"sarsa"}},
        sarsaEpisodes(result.q, environment, demoAgent)
        | std::views::take(N_EPISODES),
        std::views::repeat(PATH_COLOUR, N_EPISODES),
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