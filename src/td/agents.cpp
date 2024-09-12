#pragma once

#include <random>

#include "introRL/act/sparse.hpp"
#include "introRL/stats.hpp"
#include "introRL/td/agents.hpp"
#include "introRL/td/types.hpp"

namespace irl::td
{
    EGreedy::EGreedy(Epsilon e, std::mt19937& generator) :
        m_e{e},
        m_generator{generator}
    {}

    GridAction EGreedy::act(const Q& q, const GridState& state, const GridActions& actions)
    {
        if (shouldExplore())
        {
            return irl::sample(actions, m_generator);
        }

        return act::greedy(q, state, actions, m_generator);
    }

    bool EGreedy::shouldExplore() const
    {
        return
            std::bernoulli_distribution{double{m_e.unwrap<Epsilon>()}}(m_generator);
    }
}