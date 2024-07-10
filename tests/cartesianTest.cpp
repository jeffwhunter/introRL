#include <array>
#include <mdspan>
#include <ranges>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <introRL/afUtils.hpp>
#include <introRL/basicTypes.hpp>
#include <introRL/cartesian.hpp>

namespace irl::cartesian
{
    TEST_CASE("cartesian.Power.indexes two dimensions")
    {
        constexpr unsigned extent{5};
        constexpr unsigned rank{2};
        constexpr unsigned indexAxis{0};

        auto fastIndex{
            toVector<int>(
                Power<Extent{extent}, Rank{rank}, IndexAxis{indexAxis}, Index{0}>
                    ::elements())};

        auto slowIndex{
            toVector<int>(
                Power<Extent{extent}, Rank{rank}, IndexAxis{indexAxis}, Index{1}>
                    ::elements())};

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

    TEST_CASE("cartesian.Power.indexes four dimensions")
    {
        constexpr unsigned extent{3};
        constexpr unsigned rank{4};
        constexpr unsigned indexAxis{0};

        auto fastestIndex{
            toVector<int>(
                Power<Extent{extent}, Rank{rank}, IndexAxis{indexAxis}, Index{0}>
                    ::elements())};

        auto fastIndex{
            toVector<int>(
                Power<Extent{extent}, Rank{rank}, IndexAxis{indexAxis}, Index{1}>
                    ::elements())};

        auto slowIndex{
            toVector<int>(
                Power<Extent{extent}, Rank{rank}, IndexAxis{indexAxis}, Index{2}>
                    ::elements())};

        auto slowestIndex{
            toVector<int>(
                Power<Extent{extent}, Rank{rank}, IndexAxis{indexAxis}, Index{3}>
                    ::elements())};

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
