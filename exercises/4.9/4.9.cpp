#include <array>
#include <random>
#include <ranges>
#include <vector>

#include <arrayfire.h>
#include <indicators/cursor_control.hpp>
#include <indicators/indeterminate_progress_bar.hpp>

#include <introRL/afUtils.hpp>
#include <introRL/iteration/algorithm.hpp>
#include <introRL/iteration/subplotters.hpp>

using namespace indicators::option;
using namespace irl;
using namespace irl::iteration;

PlotSize PLOT_SIZE{.width{1'500}, .height{750}};

constexpr unsigned PROGRESS_WIDTH{50};

constexpr unsigned N_STATES{99};
constexpr unsigned MAX_N_ACTIONS{(N_STATES + 1) / 2};

constexpr auto VALUE_X_TICKS{std::to_array<unsigned>({1, 25, 50, 75, 99})};
constexpr auto PLOT_ITERATIONS{std::to_array<unsigned>({1, 2, 3, 5, 30, 100})};

struct ExperimentSetup
{
    double probHeads{};
    double threshold{};
    unsigned rounding{};
    std::vector<double> policyXTicks{};
    std::vector<double> policyYTicks{};
    indicators::Color barColour{};
};

const auto SETUPS{std::to_array<ExperimentSetup>({
    {
        .probHeads{.4},
        .threshold{1e-9},
        .rounding{5},
        .policyXTicks{1, 25, 50, 75, 99},
        .policyYTicks{1, 12, 25, 50},
        .barColour{indicators::Color::blue}
    }, {
        .probHeads{.55},
        .threshold{2e-3},
        .rounding{5},
        .policyXTicks{1, 9, 18, 32, 69, 83, 92, 99},
        .policyYTicks{1, 2, 3, 4},
        .barColour{indicators::Color::red}
    }})};

constexpr unsigned LIMIT{1'000};

struct GamblersExpecter
{
public:
    GamblersExpecter(
        double probHeads,
        StateCount nStates
    ) :
        m_stateDollars{af::range(af::dim4{nStates.unwrap<StateCount>()}, 0, u32) + 1},
        m_betLimit{
            af::min(m_stateDollars, nStates.unwrap<StateCount>() + 1 - m_stateDollars)},
        m_cLoseWin{af::moddims(af::array{-1, 1}, af::dim4{1, 2})},
        m_pLoseWin{af::moddims(af::array{1. - probHeads, probHeads}, af::dim4{1, 2})}
    {}

    static af::array indicesToActions(const ActionIndices& actionIndices)
    {
        return actionIndices.unwrap<ActionIndices>() + 1;
    }

    af::array expectedReturn(
        const ActionIndices& actionIndices,
        const StateValue& stateValue)
    {
        const auto actions{indicesToActions(actionIndices)};

        const auto stateChange{m_cLoseWin * af::min(m_betLimit, actions).as(s32)};

        const auto nextValue{
            at(
                af::join(
                    0,
                    af::array{0.},
                    stateValue.unwrap<StateValue>(),
                    af::array{1.}),
                (m_stateDollars.as(s32) + stateChange).as(u32))};

        return
            af::select(
                af::tile(actions, m_betLimit.dims()) >
                af::tile(m_betLimit, actions.dims()),
                -af::Inf,
                af::sum(m_pLoseWin * nextValue, 1));
    }

private:
    const af::array m_stateDollars{};
    const af::array m_betLimit{};
    const af::array m_cLoseWin{};
    const af::array m_pLoseWin{};
};

int main()
{
    af::getDefaultRandomEngine().setSeed(std::random_device{}());

    using Expecter = GamblersExpecter;

    indicators::show_console_cursor(false);

    auto plotter{ValueIterationSubplotter::make(
        PLOT_SIZE,
        SETUPS.size(),
        StateCount{N_STATES},
        PLOT_ITERATIONS,
        VALUE_X_TICKS,
        [](const Policy& policy)
        {
            return Expecter::indicesToActions(ActionIndices{policy.unwrap<Policy>()});
        })};

    ValueIteration valueIteration{ActionCount{MAX_N_ACTIONS}, StateCount{N_STATES}};

    for (auto&& [column, setup] : std::views::enumerate(SETUPS))
    {
        const auto title{std::format("p(Heads) = {}", setup.probHeads)};

        indicators::IndeterminateProgressBar bar{
            BarWidth{PROGRESS_WIDTH},
            Start{"["},
            Fill{"-"},
            Lead{"<=>"},
            End{"]"},
            PostfixText{std::format("value iterating {}", title)},
            ForegroundColor{setup.barColour}};

        Expecter expecter{setup.probHeads, StateCount{N_STATES}};

        plotter.setupAxes(
            title,
            setup.policyXTicks,
            setup.policyYTicks);

        valueIteration.iterate(
            [&](const ActionIndices& actions, const StateValue& stateValue)
            {
                return expecter.expectedReturn(actions, stateValue);
            },
            plotter,
            [&] { bar.tick(); },
            setup.threshold,
            setup.rounding,
            LIMIT);

        bar.mark_as_completed();
    }

    indicators::show_console_cursor(true);

    plotter.show();

    return 0;
}