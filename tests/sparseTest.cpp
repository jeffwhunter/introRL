#include <array>
#include <map>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <introRL/sparse.hpp>

namespace irl
{
    TEST_CASE("sparse.SparseMatrix.accessor.stores values")
    {
        SparseMatrix<size_t, size_t, size_t> testee{};

        testee(0, 0) = 5;
        testee(0, 1) = 7;
        testee(1, 0) = 9;
        testee(1, 1) = 6;

        REQUIRE(testee(0, 0) == 5);
        REQUIRE(testee(0, 1) == 7);
        REQUIRE(testee(1, 0) == 9);
        REQUIRE(testee(1, 1) == 6);
    }

    TEST_CASE("sparse.SparseMatrix.accessor.returns inner maps")
    {
        SparseMatrix<size_t, size_t, size_t> testee{};

        testee(0, 0) = 5;
        testee(0, 1) = 7;
        testee(1, 0) = 9;
        testee(1, 1) = 6;

        REQUIRE_THAT(
            testee(0),
            Catch::Matchers::RangeEquals(std::map<size_t, size_t>{{0, 5}, {1, 7}}));

        REQUIRE_THAT(
            testee(1),
            Catch::Matchers::RangeEquals(std::map<size_t, size_t>{{0, 9}, {1, 6}}));
    }

    TEST_CASE("sparse.SparseMatrix.accessor.handles complicated indices")
    {
        SparseMatrix<std::array<float, 3>, std::array<int, 2>, size_t> testee{};

        testee(std::array{0.f, 2.f, 1.f}, std::array{1, -2}) = 5;
        testee(std::array{2.f, 1.f, 0.f}, std::array{-1, 2}) = 10;

        REQUIRE(testee(std::array{0.f, 2.f, 1.f}, std::array{1, -2}) == 5);
        REQUIRE(testee(std::array{2.f, 1.f, 0.f}, std::array{-1, 2}) == 10);
    }
}