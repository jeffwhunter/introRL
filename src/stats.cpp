#include <cmath>

#include <arrayfire.h>

#include "introRL/stats.hpp"

namespace irl::stats
{
    af::array poisson(unsigned expectation, const af::array& samples)
    {
        return
            std::exp(-static_cast<int>(expectation)) *
            af::pow(expectation, samples) /
            af::factorial(samples);
    }
}