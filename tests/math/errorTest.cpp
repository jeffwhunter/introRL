#include <map>
#include <unordered_map>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <introRL/math/error.hpp>

namespace irl::math
{
    TEST_CASE("math.rmse.works with map")
    {
        std::map<int, float> values{{0, 1.f}, {1, -1.f}, {2, -1.f}, {3, 1.f}, {4, 1.f}, {5, -1.f}};
        std::map<int, float> answers{{0, .0f}, {1, .0f}, {2, .0f}, {3, .0f}, {4, .0f}, {5, .0f}};

        REQUIRE(rmse(values, answers) == 1.);
    }

    TEST_CASE("math.rmse.works with unordered map")
    {
        std::unordered_map<int, float> values{
            {0, 1.f}, {1, -1.f}, {2, -1.f}, {3, 1.f}, {4, 1.f}, {5, -1.f}};
        std::unordered_map<int, float> answers{
            {0, .0f}, {1, .0f}, {2, .0f}, {3, .0f}, {4, .0f}, {5, .0f}};

        REQUIRE(rmse(values, answers) == 1.);
    }

    TEST_CASE("math.rmse.handles missing values")
    {
        std::map<int, float> values{{1, 1.f}};
        std::map<int, float> answers{{0, 4.5f}, {1, -3.0f}, {2, 2.f}};

        REQUIRE(rmse(values, answers) - 3.662876 < 1e-5);
    }
}