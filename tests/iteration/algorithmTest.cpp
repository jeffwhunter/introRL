#include <limits>

#include <arrayfire.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/trompeloeil.hpp>

#include <introRL/types.hpp>
#include <introRL/iteration/algorithm.hpp>
#include <introRL/iteration/types.hpp>

namespace irl::iteration
{
    class MockPlotter
    {
    public:
        MAKE_MOCK1(plot, void(const Policy&));
        MAKE_MOCK1(plot, void(const StateValue&));
    };

    TEST_CASE("iteration.algorithm.PolicyIteration.plots at least once")
    {
        PolicyIteration testee{
            ActionCount{2},
            StateCount{3},
            [](const ActionIndices&, const StateValue& stateValue)
            {
                return stateValue.unwrap<StateValue>();
            }};

        MockPlotter plotter{};
        REQUIRE_CALL(plotter, plot(ANY(const Policy&)))
            .TIMES(1, std::numeric_limits<int>::max());
        REQUIRE_CALL(plotter, plot(ANY(const StateValue&)))
            .TIMES(1, std::numeric_limits<int>::max());

        testee.iterate(plotter);
    }

    TEST_CASE("iteration.algorithm.ValueIteration.plots at least once")
    {
        ValueIteration testee{ActionCount{2}, StateCount{3}};

        MockPlotter plotter{};
        REQUIRE_CALL(plotter, plot(ANY(const Policy&)))
            .TIMES(1, std::numeric_limits<int>::max());
        REQUIRE_CALL(plotter, plot(ANY(const StateValue&)))
            .TIMES(1, std::numeric_limits<int>::max());

        testee.iterate(
            [](const ActionIndices&, const StateValue& stateValue)
            {
                return stateValue.unwrap<StateValue>();
            },
            plotter);
    }
}