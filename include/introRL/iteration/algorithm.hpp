#pragma once

#include <functional>
#include <ranges>

#include <arrayfire.h>

#include "introRL/types.hpp"
#include "introRL/iteration/types.hpp"
#include "introRL/math/af.hpp"

namespace irl::iteration
{
    namespace detail
    {
        using ExpectedReturnFn =
            std::function<af::array(const ActionIndices&, const StateValue&)>;

        using ActionReductionFn = std::function<StateValue(const af::array&)>;

        template <class ... TArgs>
        using ProgressFn = std::function<void(TArgs ...)>;

        template <class TSubplotter>
        concept IterationSubplotter = requires (
            TSubplotter subplotter,
            const Policy& policy,
            const StateValue& stateValue)
        {
            { subplotter.plot(policy) };
            { subplotter.plot(stateValue) };
        };
    }

    /// <summary>
    /// Iteratively estimates a state value until convergence into an optimal value.
    /// </summary>
    /// <param name="actionIndices">
    /// - The future actions the state value will be estimated through.
    /// </param>
    /// <param name="initialValue">- The initial state value of the iteration.</param>
    /// <param name="expectedReturnFn">
    /// - The expected return of some actions given a state value estimate.
    /// </param>
    /// <param name="actionReductionFn">
    /// - How to reduce the state values of future actions into one current estimate.
    /// </param>
    /// <param name="progressFn">
    /// - A callback that receives iterations of the state value.
    /// </param>
    /// <param name="threshold">
    /// - Iteration stops when the change between subsequent state value estimates drops
    /// below this threshold.
    /// </param>
    /// <param name="nMaxIterations">- The maximum number of iterations.</param>
    /// <returns>The optimal state value given some expectations.</returns>
    [[nodiscard]] StateValue evaluate(
        const ActionIndices& actionIndices,
        const StateValue& initialValue,
        const detail::ExpectedReturnFn& expectedReturnFn,
        const detail::ActionReductionFn& actionReductionFn,
        const detail::ProgressFn<const StateValue&>& progressFn =
            [](const StateValue&) {},
        double threshold = 1e-9,
        size_t nMaxIterations = 1e3);

    /// <summary>
    /// Implements policy iteration, which swaps between policy evaluation and
    /// improvement, each slowly improving the other until an optimal policy and state
    /// values have been reached.
    /// </summary>
    class PolicyIteration
    {
    public:

        /// <summary>
        /// Creates a PolicyIteration.
        /// </summary>
        /// <param name="nActions">- The number of possible actions.</param>
        /// <param name="nStates">- The number of possible states.</param>
        /// <param name="expectedReturnFn">
        /// - The expected return of some actions given a state value estimate.
        /// </param>
        /// <param name="progressFn">
        /// - An update callback called during each iteration.
        /// </param>
        PolicyIteration(
            ActionCount nActions,
            StateCount nStates,
            const detail::ExpectedReturnFn& expectedReturnFn,
            const detail::ProgressFn<>& progressFn = [] {});

        /// <summary>
        /// One run of policy iteration.
        /// </summary>
        /// <param name="plotter">- The plotter in which to plot the results.</param>
        void iterate(detail::IterationSubplotter auto&& plotter) const
        {
            StateValue stateValue{m_initialState};
            Policy policy{m_initialPolicy};

            bool policyUnstable{true};
            for (const unsigned i : std::views::iota(unsigned{0})
                | std::views::take_while([&](unsigned) { return policyUnstable; }))
            {
                m_progressFn();

                stateValue = evaluate(
                    ActionIndices{policy.unwrap<Policy>()},
                    stateValue,
                    m_expectedReturnFn,
                    [](af::array expectedReturnPerAction)
                    {
                        return StateValue{expectedReturnPerAction};
                    });

                plotter.plot(policy);
                plotter.plot(stateValue);

                auto newPolicy{improve(stateValue)};

                policyUnstable = af::anyTrue<bool>(policy != newPolicy);

                policy = newPolicy;
            }
        }

    private:
        /// <summary>
        /// Returns the best policy given some state value estimate.
        /// </summary>
        /// <param name="stateValue">
        /// - The state value to use as the basis for picking a policy.
        /// </param>
        /// <returns>The best policy given some state value estimate.</returns>
        Policy improve(const StateValue& stateValue) const;

        const ActionIndices m_allActions;
        const StateValue m_initialState;
        const Policy m_initialPolicy;

        const detail::ExpectedReturnFn m_expectedReturnFn;
        const detail::ProgressFn<> m_progressFn;
    };

    /// <summary>
    /// Implements policy iteration, which iteratively improves a state value estimate
    /// with the results of the best available action, then outputs a greedy policy from
    /// that state value estimate.
    /// </summary>
    class ValueIteration
    {
    public:

        /// <summary>
        /// Creates a ValueIteration
        /// </summary>
        /// <param name="nActions">- The number of possible actions.</param>
        /// <param name="nStates">- The number of possible states.</param>
        ValueIteration(
            ActionCount nActions,
            StateCount nStates);

        /// <summary>
        /// One run of value iteration.
        /// </summary>
        /// <param name="expectedReturnFn">
        /// - The expected return of some actions given a state value estimate.
        /// </param>
        /// <param name="plotter">- The plotter in which to plot the results.</param>
        /// <param name="progressFn">
        /// - An update callback called during each iteration.
        /// </param>
        /// <param name="threshold">
        /// - Iteration stops when the change between subsequent state value estimates drops
        /// below this threshold.
        /// </param>
        /// <param name="rounding">
        /// - The number of digits to round state values to before picking a policy from
        /// them.
        /// </param>
        /// <param name="nMaxIterations">- The maximum number of iterations.</param>
        void iterate(
            const detail::ExpectedReturnFn& expectedReturnFn,
            detail::IterationSubplotter auto&& plotter,
            const detail::ProgressFn<>& progressFn = [] {},
            double threshold = 1e-9,
            unsigned rounding = 5,
            unsigned limit = 100) const
        {
            const auto stateValue{
                evaluate(
                    m_allActions,
                    m_initialState,
                    expectedReturnFn,
                    [](af::array expectedReturnPerAction)
                    {
                        return StateValue{af::max(expectedReturnPerAction, 2)};
                    },
                    [&](const StateValue& stateValue)
                    {
                        plotter.plot(stateValue);
                        progressFn();
                    },
                    threshold)};

            plotter.plot(
                Policy{
                    math::argMax<2>(
                        math::round(
                            expectedReturnFn(m_allActions, stateValue),
                            rounding))});
        }

    private:
        const ActionIndices m_allActions;
        const StateValue m_initialState;
    };
}