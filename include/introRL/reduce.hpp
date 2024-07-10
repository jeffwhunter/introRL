#pragma once

#include <functional>
#include <ranges>
#include <utility>

#include "arrayfire.h"

namespace irl::reduce
{
    /// <summary>
    /// Returns indices of the maximum elements along some dimension.
    /// </summary>
    /// <typeparam name="DIMENSION">
    /// The dimension along which to find the index of the maximum.
    /// </typeparam>
    /// <param name="m">- The matrix to find the maximum indices of.</param>
    /// <returns>The maximum indices of m along DIMENSION.</returns>
    template <unsigned DIMENSION>
    af::array argMax(const af::array& m)
    {
        static_assert(DIMENSION < 4, "af::array has only 4 dimensions.");

        constexpr auto discard{
            [] <class T, class D>(T&& t, D)
            {
                return std::forward<decltype(t)>(t);
            }};

        constexpr auto elementGetter{
            []<size_t ... PRE, size_t ... POST>
            (
                const af::array& m,
                unsigned i,
                std::index_sequence<PRE ...>,
                std::index_sequence<POST ...>)
            {
                return m(discard(af::span, PRE)..., i, discard(af::span, POST)...);
            }};

        auto resultShape{m.dims()};
        resultShape.dims[DIMENSION] = 1;

        auto max{af::constant(-af::Inf, resultShape, m.type())};
        auto maxI{af::constant(0, max.dims(), u32)};

        for (const auto i : std::views::iota(dim_t{0}, m.dims(DIMENSION)))
        {
            const auto element{
                elementGetter(
                    m,
                    i,
                    std::make_index_sequence<DIMENSION>(),
                    std::make_index_sequence<3 - DIMENSION>())};

            const auto larger{element > max};

            max = af::select(larger, element, max);
            maxI = af::select(larger, i, maxI);
        }

        return maxI;
    }
}