#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <introRL/act/sparse.hpp>
#include <introRL/math/sparse.hpp>

namespace irl::act
{
    TEST_CASE("act.map.greedy.samples ties")
    {
        std::mt19937 generator{0};

        math::SparseMatrix<int, int, double> values{};
        values(0, 0) = 0.0;
        values(0, 1) = 3.330;
        values(0, 2) = 3.333;
        values(0, 3) = 2.7;
        values(0, 4) = 1.5;
        values(0, 5) = 2.9;
        values(0, 6) = -15.0;
        values(0, 7) = 3.333000000;
        values(0, 8) = 3.333000;
        values(0, 9) = -2.0;

        std::set<int> results{};

        for (auto i : std::views::iota(0, 100))
        {
            results.emplace(
                greedy(
                    values,
                    0,
                    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                    generator));
        }

        REQUIRE_THAT(results, Catch::Matchers::UnorderedRangeEquals(std::set{2, 7, 8}));
    }

    TEST_CASE("act.map.greedy.can select values that only implicitly exist")
    {
        std::mt19937 generator{0};

        math::SparseMatrix<int, int, double> values{};
        values(0, 0) = -0.1;
        values(0, 2) = -3.333;
        values(0, 3) = -2.7;
        values(0, 4) = -1.5;
        values(0, 6) = -15.0;
        values(0, 7) = -3.333000000;
        values(0, 8) = -3.333000;

        std::set<int> results{};

        for (auto i : std::views::iota(0, 100))
        {
            results.emplace(
                greedy(
                    values,
                    0,
                    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                    generator));
        }

        REQUIRE_THAT(results, Catch::Matchers::UnorderedRangeEquals(std::set{1, 5, 9}));
    }
}