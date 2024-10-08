#pragma once

#include <algorithm>
#include <concepts>
#include <functional>
#include <map>
#include <utility>

namespace irl::math
{
    /// <summary>
    /// Returns the key associated with the largest value in the map.
    /// </summary>
    /// <typeparam name="TKey">The type of the keys of the map.</typeparam>
    /// <typeparam name="TValue">The type of the values of the map.</typeparam>
    /// <param name="m">- The map from which the maximum key will be returned.</param>
    /// <returns>The key associated with the largest value in the map.</returns>
    template <std::regular TKey, std::three_way_comparable TValue>
    auto argmax(const std::map<TKey, TValue>& m)
    {
        return std::ranges::max_element(
            m,
            std::less<TValue>{},
            &std::pair<const TKey, TValue>::second
        )->first;
    }
}