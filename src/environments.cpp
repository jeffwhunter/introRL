#include <arrayfire.h>

#include "introRL/environments.hpp"

namespace introRL::environments
{
    const af::array bandit(const af::array& qStar, const af::array& linearActionIndices)
    {
        return af::randn(qStar.dims(0), f32) + qStar(linearActionIndices);
    }
}