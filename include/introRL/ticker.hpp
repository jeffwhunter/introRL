#pragma once

#include <functional>

/// <summary>
/// Calls some call every RATE calls.
/// </summary>
/// <typeparam name="RATE">Ticks the progress bar once per RATE calls.</typeparam>
template <unsigned RATE>
class Ticker
{
    using TTick = std::function<void()>;

public:
    /// <summary>
    /// Creates a Ticker.
    /// </summary>
    /// <param name="tick">- The call to call every RATE calls.</param>
    Ticker(TTick tick) : m_tick{tick} {}

    /// <summary>
    /// Potentially calls the call.
    /// </summary>
    void operator()()
    {
        if (++m_i % RATE == 0)
        {
            m_tick();
        }
    }

private:
    unsigned m_i{};
    TTick m_tick{};
};