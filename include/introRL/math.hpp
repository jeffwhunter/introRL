#pragma once

namespace af { class array; }

namespace irl::math
{
    /// <summary>
    /// Raises some base by some power.
    /// </summary>
    /// <param name="b">- The base to raise.</param>
    /// <param name="p">- The power to raise with.</param>
    /// <returns>The base to the power of the power.</returns>
    constexpr unsigned power(unsigned int b, unsigned int p);

    /// <summary>
    /// Rounds an arrayfire array to some number of decimals.
    /// </summary>
    /// <param name="a">- The array to round.</param>
    /// <param name="decimals">- The number of decimals to round to.</param>
    /// <returns>The arrayfire array rounded to some number of decimals.</returns>
    af::array round(const af::array& a, unsigned decimals);
}