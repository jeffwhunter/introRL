#include <arrayfire.h>

#include "introRL/actions.hpp"

namespace introRL::actions
{
    af::array explore(int runs, int actions)
    {
        return af::randu(runs, u32) % actions;
    }

    af::array eGreedy(const af::array& q, float epsilon)
    {
        return af::select(
            af::randu(q.dims(0), f32) > epsilon,
            greedy(q),
            explore(q.dims(0), q.dims(1)));
    }

    af::array greedy(const af::array& q)
    {
        af::array idx, vals;
        af::topk(vals, idx, af::transpose(q), 1);

        return af::transpose(idx);
    }
}