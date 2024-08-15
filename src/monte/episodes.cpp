#include "introRL/monte/episodes.hpp"
#include "introRL/stats.hpp"

namespace irl::monte
{
    void Episode::append(State state, Probable<Action> action)
    {
        m_steps.emplace_back(std::move(state), std::move(action));
    }

    size_t Episode::bigT() const
    {
        return m_steps.size();
    }

    const Episode::Step& Episode::getStep(unsigned i) const
    {
        return m_steps[i];
    }

    void Episode::setFinalPosition(Position position)
    {
        m_finalPosition = std::move(position);
    }

    std::experimental::generator<Position> Episode::getAllPositions() const
    {
        for (const Step& step : m_steps)
        {
            co_yield step.state.position;
        }

        if (m_finalPosition.has_value())
        {
            co_yield m_finalPosition.value();
        }
    }
}