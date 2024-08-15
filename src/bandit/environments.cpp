#include <arrayfire.h>

#include "introRL/act.hpp"
#include "introRL/bandit/environments.hpp"
#include "introRL/bandit/types.hpp"
#include "introRL/types.hpp"

namespace irl::bandit
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
        return greedy(m_qStar);
    }

    void Stationary::update() const {}
}