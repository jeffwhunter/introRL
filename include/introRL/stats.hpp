#pragma once

#include <random>
#include <set>

namespace af { class array; }

namespace irl
{
    /// <summary>
    /// Some probable value.
    /// </summary>
    /// <typeparam name="T">The type of the value.</typeparam>
    template <class T>
    struct Probable
    {
        /// <summary>
        /// Some value that could happen.
        /// </summary>
        T value{};

        /// <summary>
        /// The probability of that value happening.
        /// </summary>
        float probability{};

        auto operator<=>(const Probable<T>&) const = default;
    };

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

    template <class TValue>
    const TValue& sample(
        const std::set<TValue>& set,
        std::uniform_random_bit_generator auto& generator)
    {
        std::uniform_int_distribution<size_t> distribution{0, std::size(set) - 1};

        auto it{std::begin(set)};
        std::advance(it, distribution(generator));

        return *it;
    }
}