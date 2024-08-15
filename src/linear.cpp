#include <arrayfire.h>

#include "introRL/linear.hpp"

namespace irl
{
    af::array linearIndex(const af::array & i)
    {
        const auto dZero{i.dims(0)};

        return af::iota(dZero, 1, u32) + (dZero * i).as(u32);
    }
}