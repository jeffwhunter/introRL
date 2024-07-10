#pragma once

#include <cmath>
#include <functional>
#include <ranges>

#include <arrayfire.h>

#include "introRL/basicTypes.hpp"
#include "introRL/policyTypes.hpp"

namespace irl::policy
{
    /// <summary>
    /// Types that can plot a policy and state value estimate.
    /// </summary>
    template <typename TPlotter>
    concept Plotter = requires (
        TPlotter plotter,
        const Policy& policy,
        const StateValue& stateValue)
    {
        { plotter.plot(policy, stateValue) };
    };

    /// <summary>
    /// Implements policy iteration, which swaps between policy evaluation and
    /// improvement, each slowly improving the other until an optimal policy and state
    /// values have been reached.
    /// </summary>
    class Iteration
    {
        using ExpectedReturnFn =
            std::function<StateValue(const Policy&, const StateValue&)>;

        using ActionFn = std::function<af::array(const Policy&)>;

        using ProgressFn = std::function<void(void)>;

    public:

        /// <summary>
        /// Creates an Iteration.
        /// </summary>
        /// <param name="nActions">- The number of possible actions.</param>
        /// <param name="nStates">- The number of possible states.</param>
        /// <param name="expectedReturnFn">
        /// - Returns the expected return of some policy given some state value estimate.
        /// </param>
        /// <param name="progressFn">
        /// - An update callback called after each iteration.
        /// </param>
        Iteration(
            ActionCount nActions,
            StateCount nStates,
            ExpectedReturnFn expectedReturnFn,
            ProgressFn progressFn = [] {});

        /// <summary>
        /// One run of policy iteration.
        /// </summary>
        /// <param name="plotter">- The plotter in which to plot the results.</param>
        void iterate(Plotter auto& plotter)
        {
            Policy policy{
                af::constant(
                    std::floor(m_nActions.unwrap<ActionCount>() / 2),
                    m_nStates.unwrap<StateCount>(),
                    u32)};

            StateValue stateValue{af::constant(0., m_nStates.unwrap<StateCount>(), f32)};

            bool policyUnstable{true};
            for (const unsigned i : std::views::iota(unsigned{0})
                | std::views::take_while([&](unsigned) { return policyUnstable; }))
            {
                stateValue = evaluate(policy, stateValue);

                plotter.plot(policy, stateValue);

                auto newPolicy{improve(stateValue)};

                policyUnstable = af::anyTrue<bool>(policy != newPolicy);

                policy = newPolicy;

                m_progressFn();
            }
        }

    private:

        /// <summary>
        /// Evaluates a policy, feeding the state value through the expected return
        /// function until the state value stabilizes.
        /// </summary>
        /// <param name="policy">- The policy to evaluate.</param>
        /// <param name="stateValue">
        /// - The state value to start the evaluation with.
        /// </param>
        /// <param name="threshold">- The threshold for state value convergence.</param>
        /// <returns>The state value estimate given some policy.</returns>
        StateValue evaluate(
            const Policy& policy,
            const StateValue& stateValue,
            double threshold = 1e-4);

        /// <summary>
        /// Returns the best policy given some state value estimate.
        /// </summary>
        /// <param name="stateValue">
        /// - The state value to use as the basis for picking a policy.
        /// </param>
        /// <returns>The best policy given some state value estimate.</returns>
        Policy improve(const StateValue& stateValue);

        ActionCount m_nActions;
        StateCount m_nStates;

        ExpectedReturnFn m_expectedReturnFn;
        ProgressFn m_progressFn;
    };
}