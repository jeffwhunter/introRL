#include <arrayfire.h>

#include "introRL/act.hpp"
#include "introRL/environments.hpp"
#include "introRL/banditTypes.hpp"
#include "introRL/basicTypes.hpp"

namespace irl::bandit::environments
{
    Stationary::Stationary(ActionCount nActions, RunCount nRuns)
        : m_qStar{
            af::randn(nRuns.unwrap<RunCount>(), nActions.unwrap<ActionCount>(), f32)}
    {}

    Rewards Stationary::reward(const LinearActions& actions) const
    {
        return Rewards{
            af::randn(m_qStar.dims(0), f32) + m_qStar(actions.unwrap<LinearActions>())};
    }

    LinearActions Stationary::optimal() const
    {
        return act::greedy(m_qStar);
    }

    void Stationary::update() const {}
}