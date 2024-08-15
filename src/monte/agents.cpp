#include <random>

#include "introRL/cartesian.hpp"
#include "introRL/monte/agents.hpp"
#include "introRL/stats.hpp"

namespace irl::monte
{
    Explorer Explorer::make(int minAction, int maxAction, unsigned seed)
    {
        return Explorer{M{
            .actions{
                mdIota<2>(minAction, maxAction + 1)
                | std::views::transform([](auto&& action) { return Action{action}; })
                | std::ranges::to<std::set>()},
            .generator{std::mt19937{seed}}}};
    }

    bool Explorer::should_explore(float pExplore)
    {
        return std::bernoulli_distribution{pExplore}(m.generator);
    }

    Action Explorer::explore()
    {
        return irl::sample(m.actions, m.generator);
    }

    Explorer::Explorer(M m) : m{std::move(m)} {}

    ExpertAgent::ExpertAgent(
        size_t sprintSpeed,
        size_t sprintStop,
        size_t turnStart,
        Explorer& explorer)
    :
        m_sprintSpeed{sprintSpeed},
        m_sprintStop{sprintStop},
        m_turnStart{turnStart},
        m_explorer{explorer}
    {}

    Probable<Action> ExpertAgent::act(const State& state, float pExplore)
    {
        if (m_explorer.should_explore(pExplore))
        {
            return {m_explorer.explore(), pExplore};
        }

        return {example(state), 1.f - pExplore};
    }

    static int dampen(int velocity)
    {
        return -velocity / abs(velocity);
    }

    Action ExpertAgent::example(const State& state) const
    {
        auto action{Action::make()};

        if (state.position[0] > m_sprintStop)
        {
            if (state.velocity[0] > -static_cast<int>(m_sprintSpeed))
            {
                action[0] = -1;
            }
        }
        else if (state.velocity[0] != 0)
        {
            action[0] = dampen(state.velocity[0]);
        }

        if (state.position[0] < m_turnStart)
        {
            action[1] = 1;
        }
        else if (state.velocity[1] != 0)
        {
            action[1] = dampen(state.velocity[1]);
        }

        return action;
    }

    TableAgent::TableAgent(std::map<State, Action>&& pi, Explorer& explorer) :
        m_pi{std::move(pi)},
        m_explorer{explorer}
    {}

    Probable<Action> TableAgent::act(State state, float pExplore) const
    {
        if (m_explorer.should_explore(pExplore))
        {
            return {m_explorer.explore(), pExplore};
        }

        return {
            m_pi.contains(state) ? m_pi.at(state) : Action::make(),
            1.f - pExplore};
    }
}