#include <ranges>

#include <arrayfire.h>

#include "introRL/act.hpp"
#include "introRL/basicTypes.hpp"

namespace irl::act
{
    Actions explore(RunCount nRuns, ActionCount nActions)
    {
        return Actions{
            af::randu(nRuns.unwrap<RunCount>(), u32) % nActions.unwrap<ActionCount>()};
    }

    Actions greedy(const af::array& q)
    {
        auto nActions{q.dims(1)};

        auto isMax{q == af::tile(af::max(q, 1), 1, nActions)};

        return choose(isMax.as(f32) / af::tile(af::sum(isMax, 1), 1, nActions));
    }

    Actions choose(const af::array& p)
    {
        auto roll{af::randu(p.dims(0), f32)};
        auto choice{af::constant(0u, p.dims(0), u32)};

        for (auto iCol : std::views::iota(dim_t{0}, p.dims(1) - dim_t{1}))
        {
            auto pC{p(af::span, iCol)};
            choice = af::select(roll > pC, iCol + 1, choice);
            roll -= pC;
        }

        return Actions{choice};
    }
}