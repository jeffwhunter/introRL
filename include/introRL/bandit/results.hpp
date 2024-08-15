#pragma once

#include <vector>

#include <arrayfire.h>

#include "introRL/afUtils.hpp"
#include "introRL/bandit/types.hpp"
#include "introRL/types.hpp"

namespace irl::bandit
{
    /// <summary>
    /// Calculates the average reward and optimal probaility per parameter per timestep.
    /// </summary>
    class RewardsAndOptimality
    {
        using ResultVector = std::vector<std::vector<float>>;

        struct Result;

    public:
        using RewardsResult = ResultVector;
        using OptimalityResult = ResultVector;

        /// <summary>
        /// Creates a RewardsAndOptimality for a specific number of parameters, organized
        /// according to some reduction keys.
        /// </summary>
        /// <param name="nParameters">
        /// - How many parameters these results are tracking the performance of.
        /// </param>
        /// <param name="reductionKeys">
        /// - An array of indices showing how the parameters have been distributed among
        /// parallel runs. Two adjacent equal indices imply the parameters at those
        /// indices are the same, and results will be combined over them.
        /// </param>
        RewardsAndOptimality(
            ParameterCount nParameters,
            const ReductionKeys& reductionKeys);

        /// <summary>
        /// Calculates the average reward and optimal action probability for a number of
        /// parallel runs, appending them to the results.
        /// </summary>
        /// <param name="actions">
        /// - An array of the most recent actions, one per agent.
        /// </param>
        /// <param name="optimalActions">
        /// - An array of optimal actions, one per agent.
        /// </param>
        /// <param name="rewards">
        /// - An array of the most recent rewards, one per agent.
        /// </param>
        void update(
            const LinearActions& actions,
            const LinearActions& optimalActions,
            const Rewards& rewards);

        /// <summary>
        /// Returns the recorded series of average rewards and optimal action chance.
        /// </summary>
        /// <returns>
        /// A pair of vectors holding average rewards and chance of optimal action on
        /// each timestep.
        /// </returns>
        Result value();

    private:
        struct Result
        {
            RewardsResult rewards;
            OptimalityResult optimality;
        };

        /// <summary>
        /// Creates a vector of vectors appropriate for recording a number of sets of
        /// results.
        /// </summary>
        /// <param name="nParameters">- The number of result vectors to create.</param>
        /// <returns>A vector of result vectors, one per parameter.</returns>
        ResultVector makeResultVector(unsigned nParameters);

        /// <summary>
        /// Splits up an arrayfire array, appending each element to a different result
        /// vector.
        /// </summary>
        /// <param name="newResult">- The arrayfire array to split and append.</param>
        /// <param name="resultVector">- The vector of vectors to append onto.</param>
        void appendResultVector(
            const af::array& newResult,
            ResultVector& resultVector);

        ReductionKeys m_keys;
        RewardsResult m_rewards;
        OptimalityResult m_optimality;
    };

    /// <summary>
    /// Calculates the average reward per parameter after some wait.
    /// </summary>
    /// <typeparam name="START_MEASURE_STEP">
    /// Wait this number of timesteps before tracking results.
    /// </typeparam>
    template <unsigned START_MEASURE_STEP>
    class RollingRewards
    {
    public:
        /// <summary>
        /// Creates a RollingRewards for a specific number of parameters, organized
        /// according to some reduction keys.
        /// </summary>
        /// <param name="nParameters">
        /// - How many parameters these results are tracking the performance of.
        /// </param>
        /// <param name="reductionKeys">
        /// - An array of indices showing how the parameters have been distributed among
        /// parallel runs. Two adjacent equal indices imply the parameters at those
        /// indices are the same, and results will be combined over them.
        /// </param>
        RollingRewards(
            ParameterCount nParameters,
            const ReductionKeys& reductionKeys
        ) :
            m_keys{reductionKeys},
            m_rewards{af::constant(0, nParameters.unwrap<ParameterCount>(), f32)}
        {}

        /// <summary>
        /// Updates the rolling average reward for each parameter in parallel.
        /// </summary>
        /// <param name="rewards">
        /// - An array of the most recent rewards, one per agent.
        /// </param>
        void update(const LinearActions&, const LinearActions&, const Rewards& rewards)
        {
            if (m_t++ < START_MEASURE_STEP)
            {
                return;
            }

            const af::array& rKeys{m_keys.unwrap<ReductionKeys>()};

            af::array outKeys;
            af::array outSums;

            af::sumByKey(outKeys, outSums, rKeys, rewards.unwrap<Rewards>());
            const auto nRunsPerKey{rKeys.dims(0) / outKeys.dims(0)};

            auto rewardIncrement{
                (outSums / nRunsPerKey - m_rewards) /
                (m_t - START_MEASURE_STEP)};

            m_rewards += rewardIncrement;
        }

        /// <summary>
        /// Returns the rolling average reward per parameter.
        /// </summary>
        /// <returns>A vector of rewards, one per parameter.</returns>
        std::vector<float> value()
        {
            return toVector<float>(m_rewards);
        }

    private:
        unsigned m_t{0};
        ReductionKeys m_keys;
        af::array m_rewards;
    };
}