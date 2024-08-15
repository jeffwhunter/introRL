#pragma once

#include <map>
#include <utility>

namespace irl
{
    /// <summary>
    /// A sparse matrix or two dimensional map.
    /// </summary>
    /// <typeparam name="TOuterIndex">The type of the outer index of the map.</typeparam>
    /// <typeparam name="TInnerIndex">The type of the inner index of the map.</typeparam>
    /// <typeparam name="T">The type of the values stored in the map.</typeparam>
    template <class TOuterIndex, class TInnerIndex, class T>
    class SparseMatrix
    {
        using Inner = std::map<TInnerIndex, T>;
        using Outer = std::map<TOuterIndex, Inner>;

    public:
        /// <summary>
        /// Accesses the map.
        /// </summary>
        /// <param name="iOuter">- The outer index of the element to access.</param>
        /// <param name="iInner">- The inner index of the element to access.</param>
        /// <returns>A reference to the element at the indices.</returns>
        T& operator() (TOuterIndex iOuter, TInnerIndex iInner)
        {
            return m_values[std::move(iOuter)][std::move(iInner)];
        }

        /// <summary>
        /// Accesses the map.
        /// </summary>
        /// <param name="iOuter">- The out index of the map to access.</param>
        /// <returns>A reference to the map at the index.</returns>
        const Inner& operator() (const TOuterIndex& iOuter)
        {
            return m_values[iOuter];
        }

    private:
        Outer m_values{};
    };
}