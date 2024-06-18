#include <ranges>
#include <vector>

#include <arrayfire.h>

#include "introRL/afUtils.hpp"
#include "introRL/results.hpp"
#include "introRL/types.hpp"

namespace irl::bandit::results
{
    RewardsAndOptimality::RewardsAndOptimality(
        ParameterCount nParameters,
        const ReductionKeys & reductionKeys
    ) :
        m_keys{reductionKeys},
        m_rewards{makeResultVector(nParameters.unwrap<ParameterCount>())},
        m_optimality{makeResultVector(nParameters.unwrap<ParameterCount>())}
    {}

    void RewardsAndOptimality::update(
        const Actions& actions,
        const Actions& optimalActions,
        const Rewards& rewards)
    {
        const af::array& rKeys{m_keys.unwrap<ReductionKeys>()};

        af::array outKeys;
        af::array outScan;

        af::sumByKey(outKeys, outScan, rKeys, rewards.unwrap<Rewards>());
        const auto nRunsPerKey{rKeys.dims(0) / outKeys.dims(0)};
        appendResultVector(outScan / nRunsPerKey, m_rewards);

        af::countByKey(outKeys, outScan, rKeys, actions == optimalActions);
        appendResultVector(outScan.as(f32) / nRunsPerKey, m_optimality);
    }

    RewardsAndOptimality::Result RewardsAndOptimality::value()
    {
        return { m_rewards, m_optimality };
    }

    RewardsAndOptimality::ResultVector RewardsAndOptimality::makeResultVector(
        unsigned nParameters)
    {
        return std::views::repeat(std::vector<float>{}, nParameters)
            | std::ranges::to<std::vector>();
    }

    void RewardsAndOptimality::appendResultVector(
        const af::array& newResult,
        ResultVector& resultVector)
    {
        std::vector<float> hostResult{afu::toHost<float>(newResult)};
        for (auto&& [r, v] : std::views::zip(hostResult, resultVector))
        {
            v.push_back(r);
        }
    }
}