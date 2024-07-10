#pragma once

namespace af { class array; }

namespace irl::stats
{
    /// <summary>
    /// Returns the probability of some number of events happening during some time
    /// period given some expected amount of events.
    /// </summary>
    /// <param name="expectation">- The expected number of events per time.</param>
    /// <param name="samples">
    /// - An array of arbitrary shape counting the actual number of visits per time.
    /// </param>
    /// <returns>
    /// An array with the same shape as samples, holding the probability of that number
    /// of events happening given the expectation.
    /// </returns>
    af::array poisson(unsigned expectation, const af::array& samples);
}