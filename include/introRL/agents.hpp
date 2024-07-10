#pragma once

#include <ranges>

#include <arrayfire.h>

#include "introRL/act.hpp"
#include "introRL/afUtils.hpp"
#include "introRL/banditTypes.hpp"
#include "introRL/basicTypes.hpp"

namespace irl::bandit::agents
{
    /// <summary>
    /// A bandit agent that tracks the average value of each action, and picks the best
    /// one, or explores, according to some probability.
    /// </summary>
    class EpsilonGreedyAverage
    {
    public:
        /// <summary>
        /// Creates an EpsilonGreedyAverage with different epsilons for a problem with
        /// some number of bandits.
        /// </summary>
        /// <param name="epsilons">
        /// - An array of floats, one per agent, with the probability that each agent
        /// will spend steps exploring.
        /// </param>
        /// <param name="nActions">
        /// - How many bandits can be chosen from each step.
        /// </param>
        EpsilonGreedyAverage(const DeviceParameters& epsilons, ActionCount nActions) :
            m_e{epsilons.unwrap<DeviceParameters>()},
            m_q{af::constant(0, m_e.dims(0), nActions.unwrap<ActionCount>(), f32)},
            m_n{af::constant(0, m_q.dims(), u32)}
        {}

        /// <summary>
        /// Returns the actions with the best action value estimates, or explores with
        /// some probability.
        /// </summary>
        /// <returns>An array of selected actions, one per agent.</returns>
        Actions act() const
        {
            return act::eGreedy(m_q, m_e);
        }

        /// <summary>
        /// Updates the action values so they maintain averages of the experienced
        /// rewards.
        /// </summary>
        /// <param name="actions">
        /// - An array of floats, one per agent, with the actions each chose last.
        /// </param>
        /// <param name="rewards">
        /// - An array of floats, one per agent, with the rewards that resulted from the
        /// last chosen actions.
        /// </param>
        void update(const Actions& actions, const Rewards& rewards)
        {
            const auto a{actions.unwrap<Actions>()};

            m_n(a) += 1;

            auto increment{(rewards.unwrap<Rewards>() - m_q(a)) / m_n(a)};
            m_q(a) += increment;
        }

    private:
        af::array m_e;
        af::array m_q;
        af::array m_n;
    };

    /// <summary>
    /// A bandit agent that tracks a weighted average value of each action (preferring
    /// more recent actions according to STEP_SIZE), and picks the best one, or explores,
    /// according to some probability.
    /// </summary>
    /// <typeparam name="STEP_SIZE">The size of the action value updates.</typeparam>
    template <float STEP_SIZE>
    class EpsilonGreedy
    {
    public:
        /// <summary>
        /// Creates an EpsilonGreedy with different epsilons for a problem with
        /// some number of bandits.
        /// </summary>
        /// <param name="epsilons">
        /// - A array of floats, one per agent, with the probability that each agent will
        /// spend steps exploring.
        /// </param>
        /// <param name="nActions">
        /// - How many bandits can be chosen from each step.
        /// </param>
        EpsilonGreedy(const DeviceParameters& epsilons, ActionCount nActions) :
            m_e{epsilons.unwrap<DeviceParameters>()},
            m_q{af::constant(0, m_e.dims(0), nActions.unwrap<ActionCount>(), f32)}
        {}

        /// <summary>
        /// Returns the actions with the best action value estimates, or explores with
        /// some probability.
        /// </summary>
        /// <returns>An array of selected actions, one per agent.</returns>
        Actions act() const
        {
            return act::eGreedy(m_q, m_e);
        }

        /// <summary>
        /// Updates the action values with a constant step size, so that they maintain a
        /// weighted average of recent experiences.
        /// </summary>
        /// <param name="actions">
        /// - An array of floats, one per agent, with the actions each chose last.
        /// </param>
        /// <param name="rewards">
        /// - An array of floats, one per agent, with the rewards that resulted from the
        /// last chosen actions.
        /// </param>
        void update(const Actions& actions, const Rewards& rewards)
        {
            const auto a{actions.unwrap<Actions>()};

            auto increment{(rewards.unwrap<Rewards>() - m_q(a)) * STEP_SIZE};
            m_q(a) += increment;
        }

    private:
        af::array m_e;
        af::array m_q;
    };

    /// <summary>
    /// A bandit agent that tracks a weighted average value of each action (preferring
    /// more recent actions according to STEP_SIZE), and always picks the best one.
    /// Initial action values can be set, and optimistic ones will delude the agent into
    /// overvaluing undervisited states, which enforces exploration.
    /// </summary>
    /// <typeparam name="STEP_SIZE">The size of the action value updates.</typeparam>
    template <float STEP_SIZE>
    class Optimistic
    {
    public:
        /// <summary>
        /// Creates an Optimistic with different qZeros for a problem with some number of
        /// bandits.
        /// </summary>
        /// <param name="qZeros">
        /// - An array of floats, one per agent, with the initial value that will be used
        /// for action value estimates.
        /// </param>
        /// <param name="nActions">
        /// - How many bandits can be chosen from each step.
        /// </param>
        Optimistic(const DeviceParameters& qZeros, ActionCount nActions) :
            m_q{
                af::tile(
                    qZeros.unwrap<DeviceParameters>(),
                    1,
                    nActions.unwrap<ActionCount>())}
        {}

        /// <summary>
        /// Returns the actions with the best action value estimates.
        /// </summary>
        /// <returns>An array of selected actions, one per agent.</returns>
        Actions act() const
        {
            return act::greedy(m_q);
        }

        /// <summary>
        /// Updates the action values with a constant step size, so that they maintain a
        /// weighted average of recent experiences.
        /// </summary>
        /// <param name="actions">
        /// - An array of floats, one per agent, with the actions each chose last.
        /// </param>
        /// <param name="rewards">
        /// - An array of floats, one per agent, with the rewards that resulted from the
        /// last chosen actions.
        /// </param>
        void update(const Actions& actions, const Rewards& rewards)
        {
            const auto a{actions.unwrap<Actions>()};

            auto increment{(rewards.unwrap<Rewards>() - m_q(a)) * STEP_SIZE};
            m_q(a) += increment;
        }

    private:
        af::array m_epsilons;
        af::array m_q;
    };

    /// <summary>
    /// A bandit agent that tracks the average value of each action, and picks actions
    /// with the highest expected value modified by how uncertain the agent is about each
    /// action.
    /// </summary>
    class UpperConfidence
    {
    public:
        /// <summary>
        /// Creates an UpperConfidence with different cees for a problem with some number
        /// of bandits.
        /// </summary>
        /// <param name="cees">
        /// - An array of floats, one per agent, with the coefficient of the uncertainty
        /// modifier.
        /// </param>
        /// <param name="nActions">
        /// - How many bandits can be chosen from each step.
        /// </param>
        UpperConfidence(const DeviceParameters& cees, ActionCount nActions) :
            m_cees{cees.unwrap<DeviceParameters>()},
            m_q{af::constant(0, m_cees.dims(0), nActions.unwrap<ActionCount>(), f32)},
            m_n{af::constant(0, m_cees.dims(0), nActions.unwrap<ActionCount>(), u32)}
        {}

        /// <summary>
        /// Returns the actions with the best action value estimates modified by the
        /// uncertainty in each action.
        /// </summary>
        /// <returns>An array of selected actions, one per agent.</returns>
        Actions act() const
        {
            return act::greedy(m_q + mod(m_t));
        }

        /// <summary>
        /// Updates the action values so they maintain averages of the experienced
        /// rewards.
        /// </summary>
        /// <param name="actions">
        /// - An array of floats, one per agent, with the actions each chose last.
        /// </param>
        /// <param name="rewards">
        /// - An array of floats, one per agent, with the rewards that resulted from the
        /// last chosen actions.
        /// </param>
        void update(const Actions& actions, const Rewards& rewards)
        {
            const auto a{actions.unwrap<Actions>()};

            m_n(a) += 1;

            auto increment{(rewards.unwrap<Rewards>() - m_q(a)) / m_n(a)};
            m_q(a) += increment;

            ++m_t;
        }

    private:
        /// <summary>
        /// A measure of uncertainty over actions, which increases as actions are chosen
        /// less often.
        /// </summary>
        /// <param name="timestep">The current timestep.</param>
        /// <returns>
        /// An array of shape (agents, actions) that represents the uncertainty in each
        /// action.
        /// </returns>
        af::array mod(unsigned timestep) const
        {
            return m_cees * af::sqrt(std::log(timestep) / (m_n + 1E-5));
        }

        unsigned m_t{1};
        af::array m_cees;
        af::array m_q;
        af::array m_n;
    };

    /// <summary>
    /// A bandit agent that tracks preferences instead of exact values from each action,
    /// and then uses a softmax distribution to pick between them.
    /// </summary>
    class GradientBaseline
    {
    public:
        /// <summary>
        /// Creates a GradientBaseline with different alphas for a problem with some
        /// number of bandits.
        /// </summary>
        /// <param name="alphas">
        /// - An array of floats, one per agent, with the coefficient of the preference
        /// update.
        /// </param>
        /// <param name="nActions">
        /// - The rewards that resulted from the chosen actions.
        /// </param>
        GradientBaseline(const DeviceParameters& alphas, ActionCount nActions) :
            m_alphas{alphas.unwrap<DeviceParameters>()},
            m_h{af::constant(0, m_alphas.dims(0), nActions.unwrap<ActionCount>(), f32)},
            m_rBar{af::constant(0, m_alphas.dims(), f32)}
        {}

        /// <summary>
        /// Selects randomly from actions, preferring those with higher preferences.
        /// </summary>
        /// <returns>An array of selected actions, one per agent.</returns>
        Actions act() const
        {
            return act::choose(pi());
        }

        /// <summary>
        /// Updates the actions preferences with a constant step size.
        /// </summary>
        /// <param name="actions">
        /// - An array of floats, one per agent, with the actions each chose last.
        /// </param>
        /// <param name="rewards">
        /// - An array of floats, one per agent, with the rewards that resulted from the
        /// last chosen actions.
        /// </param>
        void update(const Actions& actions, const Rewards& rewards)
        {
            const auto r{rewards.unwrap<Rewards>()};

            auto rDiff{r - m_rBar};
            m_rBar += rDiff / ++m_t;

            auto oneHot{af::constant(0, m_h.dims(), u8)};
            oneHot(actions.unwrap<Actions>()) = 1;

            auto increment{m_alphas * (r - m_rBar) * (oneHot - pi())};
            m_h += increment;
        }

    private:
        /// <summary>
        /// The probability of selecting each action given some action preferences.
        /// </summary>
        /// <returns>
        /// An array of shape (agents, actions) with the probability of selecting each
        /// action.
        /// </returns>
        af::array pi() const
        {
            auto eH{af::exp(m_h)};
            return eH / af::sum(eH, 1);
        }

        unsigned m_t{0};
        af::array m_alphas;
        af::array m_h;
        af::array m_rBar;
    };
}