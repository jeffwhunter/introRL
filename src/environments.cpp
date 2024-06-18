#include <arrayfire.h>

#include "introRL/act.hpp"
#include "introRL/environments.hpp"
#include "introRL/types.hpp"

namespace irl::bandit::environments
{
    Stationary::Stationary(ActionCount nActions, RunCount nRuns)
        : m_qStar{
            af::randn(nRuns.unwrap<RunCount>(), nActions.unwrap<ActionCount>(), f32)}
    {}

    Rewards Stationary::reward(const Actions& actions) const
    {
        return irl::Rewards{
            af::randn(m_qStar.dims(0), f32) + m_qStar(actions.unwrap<irl::Actions>())};
    }

    Actions Stationary::optimal() const
    {
        return act::greedy(m_qStar);
    }

    void Stationary::update() const {}
}