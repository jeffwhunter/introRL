#include <array>
#include <mdspan>
#include <ranges>
#include <type_traits>

#include <arrayfire.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <introRL/afUtils.hpp>

namespace irl
{
    TEST_CASE("afUtils.toPointer.creates pointers")
    {
        std::array a{0, 4, -8, 12};

        af::array m{toArrayFire(a)};

        const auto pData{toPointer<int>(m)};
        const std::span testee{pData.get(), a.size()};

        REQUIRE_THAT(testee, Catch::Matchers::RangeEquals(a));
    }

    TEST_CASE("afUtils.toMDSpan.creates rank 2 mdspans")
    {
        constexpr size_t columns{3};
        constexpr size_t rows{4};

        std::array a{
            0, 1, 2,
            4, 6, 8,
            11, 14, 17,
            21, 25, 29};

        af::array m{
            af::moddims(
                toArrayFire(a),
                af::dim4{columns, rows}).T()};

        const auto pData{toPointer<int>(m)};
        const auto testee{toMDSpan<int, 2>(pData.get(), m.dims())};

        for (auto rowIndex : std::views::iota(size_t{0}, rows))
        {
            const auto start{a.begin() + rowIndex * columns};

            REQUIRE_THAT(
                std::views::iota(size_t{0}, columns)
                | std::views::transform(
                    [&](size_t columnIndex)
                    {
                        return testee[std::to_array({rowIndex, columnIndex})];
                    }
                ),
                Catch::Matchers::RangeEquals(std::span{start, start + columns}));
        }
    }

    TEST_CASE("afUtils.toVector.creates rank 1 vectors")
    {
        std::array a{0, 4, -8, 12};

        af::array m{toArrayFire(a)};

        const auto testee{toVector<int>(m)};

        REQUIRE_THAT(testee, Catch::Matchers::RangeEquals(a));
    }

    TEST_CASE("afUtils.toMatrix.creates rank 2 vectors")
    {
        constexpr size_t columns{3};
        constexpr size_t rows{4};

        std::array a{
            0, 1, 2,
            4, 6, 8,
            11, 14, 17,
            21, 25, 29};

        af::array m{
            af::moddims(
                toArrayFire(a),
                af::dim4{columns, rows}).T()};

        const auto testee{toMatrix<int>(m)};

        for (auto rowIndex : std::views::iota(size_t{0}, rows))
        {
            const auto start{a.begin() + rowIndex * columns};

            REQUIRE_THAT(
                testee[rowIndex],
                Catch::Matchers::RangeEquals(std::span{start, start + columns}));
        }
    }
    
    TEST_CASE("afUtils.toMatrix.handles floats")
    {
        constexpr size_t columns{3};
        constexpr size_t rows{4};

        std::array a{
            0.f, 1.f, 2.f,
            4.f, 6.f, 8.f,
            11.f, 14.f, 17.f,
            21.f, 25.f, 29.f};

        af::array m{
            af::moddims(
                toArrayFire(a),
                af::dim4{columns, rows}).T()};

        const auto testee{toMatrix<float>(m)};

        for (auto rowIndex : std::views::iota(size_t{0}, rows))
        {
            const auto start{a.begin() + rowIndex * columns};

            REQUIRE_THAT(
                testee[rowIndex],
                Catch::Matchers::RangeEquals(std::span{start, start + columns}));
        }
    }
}