#include <array>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <introRL/math/array.hpp>

namespace irl::math
{
    TEST_CASE("math.array.operator+=.handles int values")
    {
        std::array testee{1, 2, 3};
        testee += std::array{3, 4, 5};

        REQUIRE_THAT(testee, Catch::Matchers::RangeEquals(std::array{4, 6, 8}));
    }
    
    TEST_CASE("math.array.operator+=.handles float values")
    {
        std::array testee{1.f, 2.f, 3.f};
        testee += std::array{3.f, 4.f, 5.f};

        REQUIRE_THAT(testee, Catch::Matchers::RangeEquals(std::array{4.f, 6.f, 8.f}));
    }

    TEST_CASE("math.array.operator+.handles int values")
    {
        auto testee{std::array{-5, 17} + std::array{9, -5}};

        REQUIRE_THAT(testee, Catch::Matchers::RangeEquals(std::array{4, 12}));
    }

    TEST_CASE("math.array.operator+.handles float values")
    {
        auto testee{std::array{-5.f, 17.f} + std::array{9.f, -5.f}};

        REQUIRE_THAT(testee, Catch::Matchers::RangeEquals(std::array{4.f, 12.f}));
    }

    TEST_CASE("math.array.operator/.handles int values")
    {
        auto testee{std::array{2, 3} / 2};

        REQUIRE_THAT(testee, Catch::Matchers::RangeEquals(std::array{1.f, 1.5f}));
    }

    TEST_CASE("math.array.operator/.handles float values")
    {
        auto testee{std::array{2.f, 3.f} / 2};

        REQUIRE_THAT(testee, Catch::Matchers::RangeEquals(std::array{1.f, 1.5f}));
    }

    TEST_CASE("math.array.cast.casts int to double")
    {
        auto testee{cast<double>(std::array{1, 2, 3})};

        REQUIRE_THAT(testee, Catch::Matchers::RangeEquals(std::array{1.f, 2.f, 3.f}));
        static_assert(std::same_as<decltype(testee), std::array<double, 3>>);
    }

    TEST_CASE("math.array.round.rounds double to int")
    {
        auto testee{round<int>(std::array{1.4f, 2.6f, 3.0f})};

        REQUIRE_THAT(testee, Catch::Matchers::RangeEquals(std::array{1, 3, 3}));
        static_assert(std::same_as<decltype(testee), std::array<int, 3>>);
    }

    TEST_CASE("math.array.interp.handles normal values")
    {
        REQUIRE_THAT(
            interp(std::array{1}, std::array{4}),
            Catch::Matchers::RangeEquals(
                std::array<std::array<int, 1>, 4>{{{2}, {3}, {4}, {5}}}));

        REQUIRE_THAT(
            interp(std::array{2}, std::array{-3}),
            Catch::Matchers::RangeEquals(
                std::array<std::array<int, 1>, 3>{{{1}, {0}, {-1}}}));

        REQUIRE_THAT(
            interp(std::array{1, 2}, std::array{-2, 3}),
            Catch::Matchers::RangeEquals(
                std::array<std::array<int, 2>, 3>{{{0, 3}, {0, 4}, {-1, 5}}}));

        REQUIRE_THAT(
            interp(std::array{1, 2}, std::array{-3, 2}),
            Catch::Matchers::RangeEquals(
                std::array<std::array<int, 2>, 3>{{{0, 3}, {-1, 3}, {-2, 4}}}));
    }
}