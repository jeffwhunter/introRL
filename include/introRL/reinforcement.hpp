#pragma once

#include <memory>
#include <vector>

#include <arrayfire.h>

namespace introRL::reinforcement
{
    /// <summary>
    /// The results of a simple bandit (slot machines!) experiment.
    /// </summary>
    struct Evaluation
    {
        std::string name;
        std::vector<float> rewards;
        std::vector<float> optimality;
    };

    /// <summary>
    /// An object capable of calculating the step sizes of action value update for a
    /// number of runs in parallel.
    /// </summary>
    class IStepSize
    {
    public:
        IStepSize() = default;
        IStepSize(const IStepSize&) = delete;
        IStepSize(IStepSize&&) = delete;
        IStepSize& operator=(const IStepSize&) = delete;
        IStepSize& operator=(IStepSize&&) = delete;
        virtual ~IStepSize() = default;

        /// <summary>
        /// Returns the step sizes for some action value updates.
        /// </summary>
        /// <param name="linearActionIndices">
        /// - The linear indices of the actions to update in each run.
        /// </param>
        /// <returns>The step size of the action value update in each run.</returns>
        virtual const af::array stepSize(const af::array& linearActionIndices) = 0;
    };

    /// <summary>
    /// Calculates step sizes appropriate for maintaining some averages. Keeps a running
    /// total of actions it has seen so it can calculate appropriate step sizes for
    /// averages over occurrences of those actions.
    /// </summary>
    class AveragingStep : public IStepSize
    {
    public:
        AveragingStep() = delete;

        /// <summary>
        /// Creates a new AveragingStep appropriate for handling a specific number of
        /// runs and actions. 
        /// </summary>
        /// <param name="runs">
        /// - The number of parallel runs to calculate step sizes for.
        /// </param>
        /// <param name="actions">- The number of actions to track.</param>
        AveragingStep(unsigned runs, unsigned actions);

        /// <summary>
        /// Calculates step sizes that will allow action value functions to maintain
        /// averages over the occurrences of the most recent actions.
        /// </summary>
        /// <param name="linearActionIndices">
        /// - Linear indices of the actions to calculate step sizes for, one per run.
        /// </param>
        /// <returns>The step size of the action value update in each run.</returns>
        const af::array stepSize(const af::array& linearActionIndices) override;

    private:
        af::array m_n;
    };

    /// <summary>
    /// Does no calculation at all and simply returns a constant scalar step size.
    /// </summary>
    class ConstantStep : public IStepSize
    {
    public:
        ConstantStep() = delete;

        /// <summary>
        /// Creates a new ConstantStep that always returns a step size of alpha.
        /// </summary>
        /// <param name="alpha">- The step size this will always return.</param>
        ConstantStep(float alpha);

        /// <summary>
        /// The constant scalar step size.
        /// </summary>
        /// <param name="linearActionIndices">
        /// - Linear action indices, which will be completely ignored.
        /// </param>
        /// <returns>The constant step size alpha.</returns>
        const af::array stepSize(const af::array& linearActionIndices) override;

    private:
        af::array m_alpha;
    };

    /// <summary>
    /// Runs a number of simple bandit algorithms from p32 of Sutton, Barto (2018). Some
    /// agents will attempt to optimize their behaviour when playing some slot machines.
    /// They will maintain values associated with each action and try to take the actions
    /// they think is best at each step. Every turn they each have some chance to take an
    /// exploratory action instead of a greedy one.
    /// </summary>
    /// <param name="runs">- The number of agents to run in parallel.</param>
    /// <param name="steps">- The number of steps to run each agent for.</param>
    /// <param name="actions">- The number of actions each agent can take.</param>
    /// <param name="epsilon">
    /// - The probability, from zero to one, that the agents will spend a turn exploring.
    /// </param>
    /// <param name="walk">
    /// - If true, the slow machines will slowly change their average payout.
    /// </param>
    /// <param name="pStepSize">
    /// - Calculates the action value updates' step sizes.
    /// </param>
    /// <returns>An evaluation of how well the agent performed.</returns>
    Evaluation simpleBandit(
        unsigned runs,
        unsigned steps,
        unsigned actions,
        float epsilon,
        bool walk,
        const std::shared_ptr<IStepSize>& pStepSize);
}