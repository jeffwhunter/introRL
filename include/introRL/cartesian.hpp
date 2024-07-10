#pragma once

#include <array>
#include <concepts>
#include <ranges>
#include <utility>

#include <arrayfire.h>

#include "introRL/basicTypes.hpp"

namespace irl::cartesian
{
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
    struct Power
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