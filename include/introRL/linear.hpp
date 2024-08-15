#pragma once

namespace af
{
    class array;
}

namespace irl
{
    /// <summary>
    /// When arrayfire gets an array index, it doesn't do anything convenient, and simply
    /// returns the elements indexed in memory order. In order to get it to act like
    /// numpy can, and return one indexed element per row, you must first 'linearize' the
    /// index's elements (convert them to memory order).
    /// </summary>
    /// <param name="i">
    /// - Row indices of the values to retreive from some matrix, the first indexing the
    /// first row, the second indexing the second row, the third indexing the third, and
    /// so on, one per row.
    /// </param>
    /// <returns>
    /// An array of indices in memory order, the first indexing  the first row, the
    /// second indexing the second row, the third indexing the third, and so on.
    /// </returns>
    af::array linearIndex(const af::array& i);
}