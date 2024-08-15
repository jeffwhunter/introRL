#include "introRL/linear.hpp"
#include "introRL/types.hpp"

namespace irl::detail
{
    LinearActionsModel::LinearActionsModel(
        const af::array& columnIndices,
        bool linearize
    ) :
        af::array{linearize ? irl::linearIndex(columnIndices) : columnIndices}
    {}
}