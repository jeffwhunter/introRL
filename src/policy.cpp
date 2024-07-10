#include <ranges>

#include <arrayfire.h>

#include "introRL/basicTypes.hpp"
#include "introRL/policy.hpp"
#include "introRL/policyTypes.hpp"
#include "introRL/reduce.hpp"

namespace irl::policy
{
    Iteration::Iteration(
        ActionCount nActions,
        StateCount nStates,
        ExpectedReturnFn expectedReturnFn,
        ProgressFn progressFn
    ) :
        m_nActions{nActions},
        m_nStates{nStates},
        m_expectedReturnFn{expectedReturnFn},
        m_progressFn{progressFn}
    {}

    StateValue Iteration::evaluate(
        const Policy& policy,
        const StateValue& stateValue,
        double threshold)
    {
        StateValue resultStateValue{stateValue};
        StateValue newStateValue{};

        double delta{1.};
        while (delta > threshold)
        {
            newStateValue = m_expectedReturnFn(policy, resultStateValue);

            delta = af::max<double>(af::abs(resultStateValue - newStateValue));
            resultStateValue = newStateValue;
        }

        return resultStateValue;
    }

    Policy Iteration::improve(const StateValue& stateValue)
    {
        return Policy{
            reduce::argMax<2>(
                m_expectedReturnFn(
                    Policy{
                        af::range(
                            af::dim4{1, 1, m_nActions.unwrap<ActionCount>()},
                            2,
                            u32)},
                    stateValue)
                .unwrap<StateValue>())};
    }
}