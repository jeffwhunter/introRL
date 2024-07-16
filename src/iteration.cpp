#include <cmath>
#include <limits>
#include <ranges>

#include <arrayfire.h>

#include "introRL/basicTypes.hpp"
#include "introRL/iteration.hpp"
#include "introRL/iterationTypes.hpp"
#include "introRL/reduce.hpp"

namespace irl::iteration
{
    [[nodiscard]] StateValue evaluate(
        const ActionIndices& actionIndices,
        const StateValue& initialValue,
        const detail::ExpectedReturnFn& expectedReturnFn,
        const detail::ActionReductionFn& actionReductionFn,
        const detail::ProgressFn<const StateValue&>& progressFn,
        double threshold,
        size_t nMaxIterations)
    {
        StateValue newValue{initialValue};
        StateValue oldValue{};

        using namespace std::literals;

        double delta{std::numeric_limits<double>::max()};
        for (const auto i : std::views::iota(0U, nMaxIterations)
            | std::views::take_while([&](unsigned) { return delta > threshold; }))
        {
            progressFn(newValue);

            oldValue = newValue;
            newValue = actionReductionFn(expectedReturnFn(actionIndices, newValue));
            delta = af::max<double>(af::abs(oldValue - newValue));
        }

        return newValue;
    }

    PolicyIteration::PolicyIteration(
        ActionCount nActions,
        StateCount nStates,
        const detail::ExpectedReturnFn& expectedReturnFn,
        const detail::ProgressFn<>& progressFn
    ) :
        m_allActions{af::range(af::dim4{1, 1, nActions.unwrap<ActionCount>()}, 2, u32)},
        m_initialState{af::constant(0., nStates.unwrap<StateCount>(), f32)},
        m_initialPolicy{
            af::constant(
                std::floor(nActions.unwrap<ActionCount>() / 2),
                nStates.unwrap<StateCount>(),
                u32)},
        m_expectedReturnFn{expectedReturnFn},
        m_progressFn{progressFn}
    {}

    Policy PolicyIteration::improve(const StateValue& stateValue) const
    {
        return Policy{
            reduce::argMax<2>(
                m_expectedReturnFn(
                    m_allActions,
                    stateValue))};
    }

    ValueIteration::ValueIteration(
        ActionCount nActions,
        StateCount nStates
    ) :
        m_allActions{af::range(af::dim4{1, 1, nActions.unwrap<ActionCount>()}, 2, u32)},
        m_initialState{af::constant(0, nStates.unwrap<StateCount>(), f64)}
    {}
}