#pragma once

#include <array>
#include <utility>

namespace irl {
    /// <summary>
    /// Arrays that protect their access by modding the index with their size.
    /// </summary>
    /// <typeparam name="T">The type to hold in the array.</typeparam>
    /// <typeparam name="N">The number of elements to hold in the array.</typeparam>
    template <class T, size_t N>
    class Ring : public std::array<T, N>
    {
    public:
        /// <summary>
        /// Returns a reference to the indexed element.
        /// </summary>
        /// <param name="i">- The index of the element to reference.</param>
        /// <returns>A reference to the indexed element.</returns>
        T& operator[](auto i) { return const_cast<T&>(std::as_const(*this)[i]); }

        /// <summary>
        /// Returns a const reference to the indexed element.
        /// </summary>
        /// <param name="i">- The index of the element to reference.</param>
        /// <returns>A reference to the indexed element.</returns>
        const T& operator[](auto i) const { return std::array<T, N>::operator[](i % N); }
    };
}