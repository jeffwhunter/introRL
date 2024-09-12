#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <map>
#include <ranges>
#include <unordered_map>
#include <utility>

#include "introRL/concepts.hpp"

namespace irl::math
{
    /// <summary>
    /// The room mean squared error of some state value estimate.
    /// </summary>
    /// <typeparam name="MappedValuesT">The type of the estimate.</typeparam>
    /// <param name="values">- The estimates to judge.</param>
    /// <param name="answers">- The answers to judge with.</param>
    /// <returns>The square root of the average squared error.</returns>
    template <class MappedValuesT>
        requires CIsMap<MappedValuesT>
    [[nodiscard]] auto rmse(const MappedValuesT& values, const MappedValuesT& answers)
    {
        const auto squared{
            answers
            | std::views::transform(
                [&](const auto& keyValue)
                {
                    const auto& [key, answer] {keyValue};

                    const auto& value{
                        values.contains(key)
                            ? values.at(key)
                            : typename MappedValuesT::mapped_type{}};

                    return std::pow(static_cast<double>(value - answer), 2.);
                })};

        return std::sqrt(std::ranges::fold_left(squared, .0, std::plus<>{}) / answers.size());
    }
}