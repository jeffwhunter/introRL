#include "introRL/linear.hpp"
#include "introRL/basicTypes.hpp"

namespace irl::detail
{
    ActionsModel::ActionsModel(const af::array& columnIndices, bool linearize)
        : af::array{linearize ? irl::linear::index(columnIndices) : columnIndices}
    {}
}