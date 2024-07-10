#include <limits>

#include <arrayfire.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/trompeloeil.hpp>

#include <introRL/basicTypes.hpp>
#include <introRL/policy.hpp>
#include <introRL/policyTypes.hpp>

namespace irl::policy
{
    class MockPlotter
    {
    public:
        MAKE_MOCK2(plot, void(const Policy&, const StateValue&));
    };

    TEST_CASE("policy.Iteration.plots at least once")
    {
        Iteration testee{
            ActionCount{2},
            StateCount{3},
            [](const Policy&, const StateValue& stateValue)
            {
                return stateValue;
            }};

        MockPlotter plotter{};
        REQUIRE_CALL(plotter, plot(ANY(const Policy&), ANY(const StateValue&)))
            .TIMES(1, std::numeric_limits<int>::max());

        testee.iterate(plotter);
    }
}