#include <array>
#include <functional>
#include <limits>
#include <random>
#include <ranges>

#include <arrayfire.h>
#include <indicators/color.hpp>
#include <indicators/cursor_control.hpp>
#include <matplot/matplot.h>
#include <stronk/prefabs.h>
#include <stronk/stronk.h>

#include "introRL/afUtils.hpp"
#include "introRL/types.hpp"
#include "introRL/cartesian.hpp"
#include "introRL/iteration/algorithm.hpp"
#include "introRL/iteration/types.hpp"
#include "introRL/stats.hpp"
#include "introRL/subplotters.hpp"
#include "introRL/utils.hpp"

using namespace irl;
using namespace irl::iteration;

constexpr unsigned FIGURE_WIDTH{1'500};
constexpr unsigned FIGURE_HEIGHT{750};

constexpr unsigned LOT_SIZE{21};
constexpr unsigned LOT_TICK_INTERVAL{4};

constexpr int MAX_MOVES{5};
constexpr unsigned N_ACTIONS{2 * MAX_MOVES + 1};

constexpr unsigned E_REQ_A{3};
constexpr unsigned E_REQ_B{4};

constexpr unsigned E_RET_A{3};
constexpr unsigned E_RET_B{2};

constexpr unsigned DEAL_SIZE{11};

constexpr unsigned POLICY_ITERATIONS{4};

constexpr ProgressWidth PROGRESS_WIDTH{50};

constexpr float MOVE_COST{2};
constexpr float FREE_MOVES_A_TO_B{1};
constexpr float FREE_MOVES_B_TO_A{0};

constexpr float HOLD_COST{4};
constexpr float HOLD_LIMIT{10};

constexpr float RENTAL_REWARD{10};
constexpr float DISCOUNT{.9};

template <size_t N>
af::array totalPoisson(
    std::array<af::array, N> counts,
    std::array<unsigned, N> expected)
{
    af::array result{af::constant(1, 1, f32)};
    for (auto&& [c, e] : std::views::zip(counts, expected))
    {
        result *= poisson(e, c);
    }

    return result;
}

af::array multiClamp(const af::array& m, const af::array& l, const af::array& h)
{
    return af::min(af::max(m, l), h);
}

struct LotSize : twig::stronk_default_unit<LotSize, Extent::underlying_type>
{
    using stronk_default_unit::stronk_default_unit;
};

struct DealSize : twig::stronk_default_unit<DealSize, Extent::underlying_type>
{
    using stronk_default_unit::stronk_default_unit;
};

template <
    LotSize LOT_SIZE,
    DealSize DEAL_SIZE,
    unsigned ... EXPECTED_DEALS>
class RentalExpecter
{
    static_assert(
        sizeof...(EXPECTED_DEALS) == 4,
        "Need four expectations for requests and returns at A and B");

    template <unsigned INDEX>
    using Cars =
        CartesianPower<
            Extent{LOT_SIZE.unwrap<LotSize>()},
            Rank{2},
            IndexAxis{0},
            Index{INDEX}>;

    template <unsigned INDEX>
    using Deals =
        CartesianPower<
            Extent{DEAL_SIZE.unwrap<DealSize>()},
            Rank{4},
            IndexAxis{1},
            Index{INDEX}>;

public:
    RentalExpecter() :
        m_nCarsA{Cars<0>::elements()},
        m_nCarsB{Cars<1>::elements()},
        m_nRequestsA{Deals<0>::elements()},
        m_nRequestsB{Deals<1>::elements()},
        m_nReturnsA{Deals<2>::elements()},
        m_nReturnsB{Deals<3>::elements()},
        m_pDeal{
            totalPoisson(
                std::to_array({m_nRequestsA, m_nRequestsB, m_nReturnsA, m_nReturnsB}),
                std::to_array({EXPECTED_DEALS...}))}
    {}

    af::array expectedReturn(
        const ActionIndices& actionIndices,
        const StateValue& stateValue)
    {
        const auto actions{indicesToActions(actionIndices)};
        const auto validActions{multiClamp(actions, -m_nCarsB, m_nCarsA)};

        constexpr auto lotSize{LOT_SIZE.unwrap<LotSize>()};

        const auto postActionCarsA{af::clamp(m_nCarsA - validActions, 0, lotSize - 1)};
        const auto postActionCarsB{af::clamp(m_nCarsB + validActions, 0, lotSize - 1)};

        const auto validRequestsA{af::min(postActionCarsA, m_nRequestsA)};
        const auto validRequestsB{af::min(postActionCarsB, m_nRequestsB)};

        const auto postReturnedCarsA{
            af::clamp(
                postActionCarsA - validRequestsA + m_nReturnsA,
                0,
                lotSize - 1)};
        const auto postReturnedCarsB{
            af::clamp(
                postActionCarsB - validRequestsB + m_nReturnsB,
                0,
                lotSize - 1)};

        return
            af::select(
                invalidActions(actions, m_nCarsA, m_nCarsB),
                -af::Inf,
                af::sum(
                    m_pDeal * (
                        RENTAL_REWARD * (validRequestsA + validRequestsB) +
                        DISCOUNT * at(
                            stateValue.unwrap<StateValue>(),
                            postReturnedCarsB * lotSize + postReturnedCarsA)),
                    1)) -
            moveCost(validActions) -
            holdCost(postActionCarsA, postActionCarsB);
    }

    af::array indicesToActions(const ActionIndices& actionIndices)
    {
        return actionIndices.unwrap<ActionIndices>().as(s32) - MAX_MOVES;
    }

private:
    af::array invalidActions(
        const af::array& actions,
        const af::array& posLimit,
        const af::array& negLimit)
    {
        const auto tiledActions{af::tile(actions, posLimit.dims(0) / actions.dims(0))};

        return
            tiledActions > af::tile(posLimit, 1, 1, actions.dims(2)) ||
            -tiledActions > af::tile(negLimit, 1, 1, actions.dims(2));
    }

    af::array moveCost(const af::array& validActions)
    {
        constexpr auto infinity{std::numeric_limits<int>::max()};

        return
            (
                af::clamp(validActions - FREE_MOVES_A_TO_B, 0, infinity) +
                af::clamp(-validActions - FREE_MOVES_B_TO_A, 0, infinity)
            ) * MOVE_COST;
    }

    af::array holdCost(const af::array& postActionCarsA, const af::array& postActionCarsB)
    {
        return
            (
                (postActionCarsA > HOLD_LIMIT) + 
                (postActionCarsB > HOLD_LIMIT)
            ) * HOLD_COST;
    }

    const af::array m_nCarsA;
    const af::array m_nCarsB;
    const af::array m_nRequestsA;
    const af::array m_nRequestsB;
    const af::array m_nReturnsA;
    const af::array m_nReturnsB;

    const af::array m_pDeal;
};

int main()
{
    af::getDefaultRandomEngine().setSeed(std::random_device{}());

    using Expecter = RentalExpecter<
        LotSize{LOT_SIZE},
        DealSize{DEAL_SIZE},
        E_REQ_A, E_REQ_B, E_RET_A, E_RET_B>;

    Expecter expecter{};

    indicators::show_console_cursor(false);

    auto bar{
        irl::makeBar(
            "policy iterating",
            indicators::Color::unspecified,
            PROGRESS_WIDTH,
            ProgressTicks{POLICY_ITERATIONS})};

    bar.set_progress(0);

    auto plotter{
        PolicyValueSubplotter::make(
            Size{.width{FIGURE_WIDTH}, .height{FIGURE_HEIGHT}},
            POLICY_ITERATIONS,
            LOT_SIZE,
            LOT_TICK_INTERVAL,
            Limits{.low{-MAX_MOVES}, .high{MAX_MOVES}},
            [&](const Policy& policy)
            {
                return expecter.indicesToActions(ActionIndices{policy.unwrap<Policy>()});
            })};

    iteration::PolicyIteration policyIteration{
        ActionCount{N_ACTIONS},
        StateCount{LOT_SIZE * LOT_SIZE},
        [&](const ActionIndices& actionIndices, const StateValue& stateValue)
        {
            return expecter.expectedReturn(actionIndices, stateValue);
        },
        [&] { bar.tick(); }};

    policyIteration.iterate(plotter);

    indicators::show_console_cursor(true);

    plotter.show();

    return 0;
}