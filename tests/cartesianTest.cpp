#include <array>
#include <ranges>

#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <introRL/afUtils.hpp>
#include <introRL/types.hpp>
#include <introRL/cartesian.hpp>

namespace irl
{
    TEST_CASE("cartesian.cartesianArrays.includes all combinations")
    {
        REQUIRE_THAT(
            cartesianArrays(std::views::iota(1, 3), std::views::iota(-5, -2)),
            Catch::Matchers::RangeEquals(
                std::to_array({
                    std::to_array({1, -5}),
                    std::to_array({1, -4}),
                    std::to_array({1, -3}),
                    std::to_array({2, -5}),
                    std::to_array({2, -4}),
                    std::to_array({2, -3})})));
    }

    TEST_CASE("cartesian.mdIota.includes all combinations")
    {
        REQUIRE_THAT(
            mdIota<3>(2, 4),
            Catch::Matchers::RangeEquals(
                std::to_array({
                    std::to_array({2, 2, 2}),
                    std::to_array({2, 2, 3}),
                    std::to_array({2, 3, 2}),
                    std::to_array({2, 3, 3}),
                    std::to_array({3, 2, 2}),
                    std::to_array({3, 2, 3}),
                    std::to_array({3, 3, 2}),
                    std::to_array({3, 3, 3})})));
    }

    TEST_CASE("cartesian.mdIndices.includes all combinations")
    {
        REQUIRE_THAT(
            mdIndices(2U, 4U),
            Catch::Matchers::RangeEquals(
                std::to_array({
                    std::to_array({0U, 0U}),
                    std::to_array({0U, 1U}),
                    std::to_array({0U, 2U}),
                    std::to_array({0U, 3U}),
                    std::to_array({1U, 0U}),
                    std::to_array({1U, 1U}),
                    std::to_array({1U, 2U}),
                    std::to_array({1U, 3U})})));
    }

    TEST_CASE("cartesian.CartesianPower.elements.indexes two dimensions")
    {
        constexpr unsigned extent{5};
        constexpr unsigned rank{2};
        constexpr unsigned indexAxis{0};

        auto fastIndex{
            toVector<int>(
                CartesianPower<
                    Extent{extent},
                    Rank{rank},
                    IndexAxis{indexAxis},
                    Index{0}
                >::elements())};

        auto slowIndex{
            toVector<int>(
                CartesianPower<
                    Extent{extent},
                    Rank{rank},
                    IndexAxis{indexAxis},
                    Index{1}
                >::elements())};

        REQUIRE_THAT(
            fastIndex,
            Catch::Matchers::RangeEquals(
                std::views::iota(0, std::pow(extent, rank))
                | std::views::transform(
                    [](auto i) { return i % extent; })));

        REQUIRE_THAT(
            slowIndex,
            Catch::Matchers::RangeEquals(
                std::views::iota(0, std::pow(extent, rank))
                | std::views::transform(
                    [](auto i) { return i / extent; })));
    }

    TEST_CASE("cartesian.CartesianPower.elements.indexes four dimensions")
    {
        constexpr unsigned extent{3};
        constexpr unsigned rank{4};
        constexpr unsigned indexAxis{0};

        auto fastestIndex{
            toVector<int>(
                CartesianPower<
                    Extent{extent},
                    Rank{rank},
                    IndexAxis{indexAxis},
                    Index{0}
                >::elements())};

        auto fastIndex{
            toVector<int>(
                CartesianPower<
                    Extent{extent},
                    Rank{rank},
                    IndexAxis{indexAxis},
                    Index{1}
                >::elements())};

        auto slowIndex{
            toVector<int>(
                CartesianPower<
                    Extent{extent},
                    Rank{rank},
                    IndexAxis{indexAxis},
                    Index{2}
                >::elements())};

        auto slowestIndex{
            toVector<int>(
                CartesianPower<
                    Extent{extent},
                    Rank{rank},
                    IndexAxis{indexAxis},
                    Index{3}
                >::elements())};

        REQUIRE_THAT(
            fastestIndex,
            Catch::Matchers::RangeEquals(
                std::views::iota(0, std::pow(extent, rank))
                | std::views::transform(
                    [](auto i)
                    {
                        return (i / static_cast<unsigned>(std::pow(extent, 0))) % extent;
                    })));

        REQUIRE_THAT(
            fastIndex,
            Catch::Matchers::RangeEquals(
                std::views::iota(0, std::pow(extent, rank))
                | std::views::transform(
                    [](auto i)
                    {
                        return (i / static_cast<unsigned>(std::pow(extent, 1))) % extent;
                    })));

        REQUIRE_THAT(
            slowIndex,
            Catch::Matchers::RangeEquals(
                std::views::iota(0, std::pow(extent, rank))
                | std::views::transform(
                    [](auto i)
                    {
                        return (i / static_cast<unsigned>(std::pow(extent, 2))) % extent;
                    })));

        REQUIRE_THAT(
            slowestIndex,
            Catch::Matchers::RangeEquals(
                std::views::iota(0, std::pow(extent, rank))
                | std::views::transform(
                    [](auto i)
                    {
                        return (i / static_cast<unsigned>(std::pow(extent, 3))) % extent;
                    })));
    }
}
