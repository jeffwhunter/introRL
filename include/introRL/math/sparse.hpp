#pragma once

#include <concepts>
#include <map>
#include <utility>

namespace irl::math
{
    /// <summary>
    /// A sparse matrix or two dimensional map.
    /// </summary>
    /// <typeparam name="TOuterIndex">The type of the outer index of the map.</typeparam>
    /// <typeparam name="TInnerIndex">The type of the inner index of the map.</typeparam>
    /// <typeparam name="T">The type of the values stored in the map.</typeparam>
    template <
        std::totally_ordered TOuterIndex,
        std::totally_ordered TInnerIndex,
        std::semiregular T>
    class SparseMatrix
    {
        using Inner = std::map<TInnerIndex, T>;
        using Outer = std::map<TOuterIndex, Inner>;

    public:
        /// <summary>
        /// Checks if the matrix has any non-default entries with some outer index.
        /// </summary>
        /// <param name="iOuter">- The outer index to check.</param>
        /// <returns>
        /// True if the matrix has non-default entries under iOuter, false otherwise.
        /// </returns>
        bool contains(const TOuterIndex& iOuter) const
        {
            return m_values.contains(iOuter);
        }

        /// <summary>
        /// Accesses the map, inserting the element if it doesn't exist yet.
        /// </summary>
        /// <param name="iOuter">- The outer index of the element to access.</param>
        /// <param name="iInner">- The inner index of the element to access.</param>
        /// <returns>A reference to the element at the indices.</returns>
        T& operator() (const TOuterIndex& iOuter, const TInnerIndex& iInner)
        {
            return m_values[iOuter][iInner];
        }

        /// <summary>
        /// Accesses the map, returning default values if the sought indices don't exist.
        /// </summary>
        /// <param name="iOuter">- The outer index to check.</param>
        /// <param name="iInner">- The inner index to check.</param>
        /// <returns>
        /// The value in the map at the given indices, or a zero-initialized T.
        /// </returns>
        T peek(const TOuterIndex& iOuter, const TInnerIndex& iInner) const
        {
            auto itOuter{m_values.find(iOuter)};
            if (itOuter == m_values.cend())
            {
                return T{};
            }

            auto itInner{itOuter->second.find(iInner)};
            if (itInner == itOuter->second.cend())
            {
                return T{};
            }

            return itInner->second;
        }

        /// <summary>
        /// Accesses a submap for some outer index, throwing if it doesn't exist.
        /// </summary>
        /// <param name="iOuter">- The outer index of the submap to access.</param>
        /// <returns>A const reference to the submap at the index.</returns>
        const Inner& operator() (const TOuterIndex& iOuter) const
        {
            return m_values.at(iOuter);
        }

    private:
        Outer m_values{};
    };
}