#include <arrayfire.h>

#include "introRL/math.hpp"

namespace irl::math
{
    constexpr unsigned power(unsigned int b, unsigned int p)
    {
        if (p == 0) return 1;
        if (p == 1) return b;

        const auto intSqrt{power(b, p / 2)};
        if ((p % 2) == 0) return intSqrt * intSqrt;
        return b * intSqrt * intSqrt;
    }

    af::array round(const af::array& a, unsigned decimals)
    {
        const unsigned scale{power(10, decimals)};

        return af::round(a * scale) / scale;
    }
}