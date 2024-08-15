#pragma once

#include <array>
#include <concepts>
#include <ranges>
#include <utility>

#include <arrayfire.h>

#include "introRL/types.hpp"

namespace irl
{
    /// <summary>
    /// Returns a view over arrays where each holds one element of the cartesian product
    /// of the input ranges.
    /// </summary>
    /// <typeparam name="...RANGES">The types of the input ranges.</typeparam>
    /// <param name="...ranges">- The input ranges.</param>
    /// <returns>
    /// A range of arrays where each array is one element of the cartesian product of the
    /// inputs.
    /// </returns>
    template <std::ranges::range ... RANGES>
    auto cartesianArrays(RANGES ... ranges)
    {
        return
            std::views::cartesian_product(ranges ...)
            | std::views::transform(
                [](auto&& tuple)
                {
                    return std::apply(
                        [](auto&& ... e)
                        {
                            return std::array{std::forward<decltype(e)>(e) ...};
                        },
                        std::forward<decltype(tuple)>(tuple));
                });
    }

    /// <summary>
    /// A multidimensional iota.
    /// </summary>
    /// <typeparam name="DIMENSION">The dimension of the results.</typeparam>
    /// <typeparam name="T">The type of the elements of the results.</typeparam>
    /// <param name="start">- The starting value of the iota in each dimension.</param>
    /// <param name="bound">- The bound of the iota in each dimension.</param>
    /// <returns>
    /// A range of DIMENSION dimensional arrays whose elements range over [start, bound).
    /// </returns>
    template <unsigned DIMENSION, std::totally_ordered T>
    auto mdIota(T start, T bound)
    {
        constexpr auto discard{[](auto r, auto) { return r; }};

        constexpr auto helper{
            [] <size_t ... I>
            (const auto & range, std::index_sequence<I...>)
            {
                return cartesianArrays(discard(range, I) ...);
            }};

        return helper(
            std::views::iota(start, bound),
            std::make_index_sequence<DIMENSION>());
    }

    /// <summary>
    /// Indices suited for looping over a multidimensional array.
    /// </summary>
    /// <typeparam name="...TExtent">The types of the extents.</typeparam>
    /// <param name="...extents">- The size of the iteration in each dimension.</param>
    /// <returns>
    /// A range of arrays with one dimension for each input extent whose elements range
    /// over [0, e) for each extent e.
    /// </returns>
    template <class ... TExtent>
    auto mdIndices(TExtent ... extents)
    {
        return cartesianArrays(std::views::iota(TExtent{0}, extents) ...);
    }

    /// <summary>
    /// A type that produces arrayfire arrays which correspond to different elements of
    /// some square cartesian index.
    /// </summary>
    /// <typeparam name="EXTENT">The maximum iteration along any dimension.</typeparam>
    /// <typeparam name="RANK">The number of dimensions to iterate over.</typeparam>
    /// <typeparam name="AXIS">
    /// The arrayfire dimension that the final result should iterate over.
    /// </typeparam>
    /// <typeparam name="INDEX">
    /// The element of the cartesian indices that the result will embody.
    /// </typeparam>
    template <Extent EXTENT, Rank RANK, IndexAxis AXIS, Index INDEX>
    struct CartesianPower
    {
        static_assert(RANK.unwrap<Rank>() <= 4, "Rank too high for arrayfire.");
        static_assert(AXIS.unwrap<IndexAxis>() <= 4, "Axis too high for arrayfire.");
        static_assert(
            INDEX.unwrap<Index>() < RANK.unwrap<Rank>(),
            "Index must be lower than rank.");

        /// <summary>
        /// Returns an arrafire array whose elements are each some element from a set of
        /// cartesian indices.
        /// </summary>
        /// <returns>
        /// An arrayfire array whose elements are each some element from a set of
        /// cartesian indices.
        /// </returns>
        static [[nodiscard]] af::array elements()
        {
            return af::moddims(
                af::range(inputShape(), INDEX.unwrap<Index>(), s32), outputShape());
        }

    private:
        /// <summary>
        /// Returns the shape of the overall cartesian index space.
        /// </summary>
        /// <returns>The shape of the cartesian index space.</returns>
        static [[nodiscard]] af::dim4 inputShape()
        {
            af::dim4 result{1};
            std::fill_n(result.dims, RANK.unwrap<Rank>(), EXTENT.unwrap<Extent>());
            return result;
        }

        /// <summary>
        /// Returns the flattened shape of the cartesian index space.
        /// </summary>
        /// <returns>The shape of the flattened cartesian index space.</returns>
        static [[nodiscard]] af::dim4 outputShape()
        {
            af::dim4 result{1};
            result.dims[AXIS.unwrap<IndexAxis>()] =
                std::pow(EXTENT.unwrap<Extent>(), RANK.unwrap<Rank>());
            return result;
        }
    };
}