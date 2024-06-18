#pragma once

#include <memory>
#include <ranges>
#include <vector>

#include <arrayfire.h>

namespace irl::afu
{
    /// <summary>
    /// Gets a unique pointer to a section of host memory holding a copy of the device
    /// memory in an arrayfire array.
    /// </summary>
    /// <typeparam name="T">
    /// The type of host memory to get (ex: float, unsigned, ...).
    /// </typeparam>
    /// <param name="m">- An arrayfire array to get a copy of the contents from.</param>
    /// <returns>
    /// A unique pointer to a section of host memory that will clean itself up.
    /// </returns>
    template <typename T>
    [[nodiscard]] auto hostPointer(const af::array& m) -> decltype(auto)
    {
        return std::unique_ptr<T, decltype(&af::freeHost)>{m.host<T>(), &af::freeHost};
    }

    /// <summary>
    /// Copies a one dimensional arrayfire array into a vector.
    /// </summary>
    /// <typeparam name="T">The element type of the vector to fill.</typeparam>
    /// <param name="m">The one dimensional arrayfire array to copy.</param>
    /// <returns>A vector full of the same elements as the arrayfire array.</returns>
    template <typename T>
    [[nodiscard]] std::vector<T> toHost(const af::array& m)
    {
        const auto pHost{hostPointer<T>(m)};
        return std::vector<T>{pHost.get(), pHost.get() + m.elements()};
    }

    /// <summary>
    /// Copies a range into an arrayfire array.
    /// </summary>
    /// <param name="v">The range to copy into the arrayfire array.</param>
    /// <returns>
    /// A new arrayfire array with the same contents at the input range.
    /// </returns>
    [[nodiscard]] af::array toArrayFire(std::ranges::contiguous_range auto r)
    {
        return af::array{static_cast<dim_t>(std::ranges::size(r)), std::ranges::data(r)};
    }
}