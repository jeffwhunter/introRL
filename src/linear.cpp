#include <arrayfire.h>

#include "introRL/linear.hpp"

namespace introRL::linear
{
    const af::array index(const af::array & i)
    {
        auto dZero{i.dims(0)};

        return af::iota(dZero, 1, u32) + (dZero * i).as(u32);
    }
}