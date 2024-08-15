#include <cmath>

#include <arrayfire.h>

#include "introRL/stats.hpp"

namespace irl
{
    af::array poisson(unsigned expectation, const af::array& samples)
    {
        return
            std::exp(-static_cast<int>(expectation)) *
            af::pow(expectation, samples) /
            af::factorial(samples);
    }
}