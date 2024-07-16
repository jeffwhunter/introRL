#include "introRL/linear.hpp"
#include "introRL/basicTypes.hpp"

namespace irl::detail
{
    LinearActionsModel::LinearActionsModel(
        const af::array& columnIndices,
        bool linearize
    ) :
        af::array{linearize ? irl::linear::index(columnIndices) : columnIndices}
    {}
}