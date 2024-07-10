#pragma once

#include <mdspan>
#include <memory>
#include <ranges>
#include <span>

#include <arrayfire.h>

namespace irl
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
    [[nodiscard]] decltype(auto) toPointer(const af::array& m)
    {
        return std::unique_ptr<T, decltype(&af::freeHost)>{m.host<T>(), &af::freeHost};
    }

    /// <summary>
    /// Returns a non-owning mdspan of some shaped block of data.
    /// </summary>
    /// <typeparam name="T">The element type of the data block.</typeparam>
    /// <typeparam name="R">The rank of the data block.</typeparam>
    /// <param name="pData">- A non-owning pointer to the data block.</param>
    /// <param name="shape">- The shape of the data block.</param>
    /// <returns>An mdspan over some shaped block of data.</returns>
    template <class T, size_t R>
    [[nodiscard]] std::mdspan<T, std::dextents<size_t, R>, std::layout_left> toMDSpan(
        T* pData,
        af::dim4 shape)
    {
        static_assert(R < 5, "Arrayfire can only handle rank 4 tensors.");

        constexpr auto makeSpan{
            [] <typename T, size_t ... E>
            (T* pData, af::dim4 shape, std::index_sequence<E ...>)
            {
                return std::mdspan<T, std::dextents<size_t, R>, std::layout_left>{
                    pData,
                    shape[E]...};
            }};

        return makeSpan(pData, shape, std::make_index_sequence<R>());
    }

    /// <summary>
    /// Copies a vector from a rank 1 arrayfire array.
    /// </summary>
    /// <typeparam name="T">The element type of vector.</typeparam>
    /// <param name="m">- The arrayfire array to copy from.</param>
    /// <returns>A vector of Ts with elements copied from m.</returns>
    template <class T>
    [[nodiscard]] std::vector<T> toVector(const af::array& m)
    {
        const auto pHost{irl::toPointer<T>(m)};
        return
            std::span{pHost.get(), static_cast<size_t>(m.elements())}
            | std::ranges::to<std::vector>();
    }

    /// <summary>
    /// Copies a vector of vectors from a rank 1 arrayfire array.
    /// </summary>
    /// <typeparam name="T">The element type of the vector of vectors.</typeparam>
    /// <param name="m">- The arrayfire array to copy from.</param>
    /// <returns>A vector of vectors of Ts with elements copied from m.</returns>
    template <class T>
    [[nodiscard]] std::vector<std::vector<T>> toMatrix(const af::array& m)
    {
        const auto pHost{irl::toPointer<T>(m)};
        const auto span{toMDSpan<T, 2>(pHost.get(), m.dims())};

        const auto slowIndexToElementVector{
            [&](size_t slowIndex)
            {
                return
                    std::views::iota(size_t{0}, span.extent(1))
                    | std::views::transform(
                        [&](size_t fastIndex)
                        {
                            return span[std::to_array({slowIndex, fastIndex})];
                        })
                    | std::ranges::to<std::vector>();
            }};

        return
            std::views::iota(size_t{0}, span.extent(0))
            | std::views::transform(slowIndexToElementVector)
            | std::ranges::to<std::vector>();
    }

    /// <summary>
    /// Copies a range into an arrayfire array.
    /// </summary>
    /// <param name="r">- The range to copy into the arrayfire array.</param>
    /// <returns>
    /// A new arrayfire array with the same contents at the input range.
    /// </returns>
    [[nodiscard]] af::array toArrayFire(std::ranges::contiguous_range auto r)
    {
        return af::array{static_cast<dim_t>(std::ranges::size(r)), std::ranges::data(r)};
    }
}