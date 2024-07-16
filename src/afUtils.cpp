#include <arrayfire.h>

#include "introRL/afUtils.hpp"

namespace irl
{
    [[nodiscard]] af::array at(const af::array& m, const af::array& i)
    {
        return af::moddims(m(i), i.dims());
    }
}