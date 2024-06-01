#include <format>
#include <memory>
#include <vector>

#include <arrayfire.h>

#include "introRL/actions.hpp"
#include "introRL/environments.hpp"
#include "introRL/linear.hpp"
#include "introRL/reinforcement.hpp"

namespace introRL::reinforcement
{
    AveragingStep::AveragingStep(unsigned runs, unsigned actions) :
        m_n{af::constant(0, runs, actions, u32)}
    {}

    const af::array AveragingStep::stepSize(const af::array& linearActionIndices)
    {
        m_n(linearActionIndices) += 1;
        return 1.f / m_n(linearActionIndices);
    }

    ConstantStep::ConstantStep(float alpha) : m_alpha{alpha} {}

    const af::array ConstantStep::stepSize(const af::array& linearActionIndices)
    {
        return m_alpha;
    }

    Evaluation simpleBandit(
        unsigned runs,
        unsigned steps,
        unsigned actions,
        float epsilon,
        bool walk,
        const std::shared_ptr<IStepSize>& pStepSize)
    {
        auto averageRewards{std::vector<float>()};
        auto optimality{std::vector<float>()};

        auto qStar{af::randn(runs, actions, f32)};

        auto q{af::constant(0, runs, actions, f32)};

        for (unsigned i{0}; i < steps; ++i)
        {
            auto iActions{actions::eGreedy(q, epsilon)};

            auto a{linear::index(iActions)};

            auto rewards{environments::bandit(qStar, a)};

            auto increment{(rewards - q(a)) * pStepSize->stepSize(a)};

            q(a) += increment;

            averageRewards.push_back(af::mean<float>(rewards));
            optimality.push_back(af::mean<float>(iActions == actions::greedy(qStar)));

            if (walk)
            {
                qStar += af::randn(runs, actions, f32) / 100.0;
            }
        }

        return {std::format("e={}", epsilon), averageRewards, optimality};
    }
}