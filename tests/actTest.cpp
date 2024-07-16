#include <ranges>

#include <arrayfire.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <introRL/act.hpp>
#include <introRL/afUtils.hpp>
#include <introRL/basicTypes.hpp>

namespace irl::act
{

    TEST_CASE("act.explore.has the proper shape")
    {
        REQUIRE(
            explore(RunCount{5}, ActionCount{7}).unwrap<LinearActions>().dims() ==
            af::dim4{5});
    }

    TEST_CASE("act.explore.has the proper type")
    {
        REQUIRE(
            explore(RunCount{2}, ActionCount{3}).unwrap<LinearActions>().type() == u32);
    }

    TEST_CASE("act.eGreedy.has the proper shape")
    {
        REQUIRE(
            eGreedy(af::constant(0, 4, 8, f32), 0).unwrap<LinearActions>().dims() ==
            af::dim4{4});

        REQUIRE(
            eGreedy(
                af::constant(0, 4, 8, f32),
                af::array{0, 0, 0, 0}
            ).unwrap<LinearActions>().dims() == af::dim4{4});
    }

    TEST_CASE("act.eGreedy.has the proper type")
    {
        REQUIRE(
            eGreedy(af::constant(0, 4, 8, f32), 0).unwrap<LinearActions>().type() ==
            u32);

        REQUIRE(
            eGreedy(
                af::constant(0, 4, 8, f32),
                af::array{0, 0, 0, 0}
            ).unwrap<LinearActions>().type() == u32);
    }

    TEST_CASE("act.eGreedy.is greedy when epsilon is zero")
    {
        constexpr unsigned nRuns{3};
        constexpr unsigned nActions{5};

        const auto q{
            af::moddims(
                af::range(af::dim4{nRuns * nActions}, 0, f32),
                af::dim4{nRuns, nActions}
            ) % nActions};

        const auto allMax{
            Catch::Matchers::RangeEquals(
                std::views::repeat(nActions - 1)
                | std::views::take(nRuns))};

        REQUIRE_THAT(
            toVector<float>(q(eGreedy(q, 0).unwrap<LinearActions>())),
            allMax);

        REQUIRE_THAT(
            toVector<float>(
                q(eGreedy(q, af::constant(0, nRuns)).unwrap<LinearActions>())),
            allMax);
    }

    TEST_CASE("act.greedy.returns highest index")
    {
        constexpr unsigned nRuns{3};
        constexpr unsigned nActions{5};

        const auto q{
            af::moddims(
                af::range(af::dim4{nRuns * nActions}, 0, f32),
                af::dim4{nRuns, nActions}
            ) % nActions};

        REQUIRE_THAT(
            toVector<float>(q(greedy(q).unwrap<LinearActions>())),
            Catch::Matchers::RangeEquals(
                std::views::repeat(nActions - 1)
                | std::views::take(nRuns)));
    }

    TEST_CASE("act.greedy.probably breaks ties")
    {
        REQUIRE(
            af::setUnique(
                greedy(af::constant(0, 10, 10)).unwrap<LinearActions>()
            ).dims(0) >
            1);
    }

    TEST_CASE("act.choose.picks definite results")
    {
        constexpr unsigned nRuns{3};
        constexpr unsigned nActions{5};

        const auto p{
            ((af::moddims(
                af::range(af::dim4{nRuns * nActions}, 0, f32),
                af::dim4{nRuns, nActions}
            ) % nActions) == 0.).as(f32)};

        REQUIRE_THAT(
            toVector<float>(p(choose(p).unwrap<LinearActions>())),
            Catch::Matchers::RangeEquals(
                std::views::repeat(1.)
                | std::views::take(nRuns)));
    }

    TEST_CASE("act.choose.probably breaks ties")
    {
        REQUIRE(
            af::setUnique(
                choose(af::constant(0, 10, 10)).unwrap<LinearActions>()
            ).dims(0) >
            1);
    }

    TEST_CASE("act.choose.has the proper shape")
    {
        REQUIRE(
            choose(af::constant(0, 4, 8, f32)).unwrap<LinearActions>().dims() ==
            af::dim4{4});
    }

    TEST_CASE("act.choose.has the proper type")
    {
        REQUIRE(
            choose(af::constant(0, 4, 8, f32)).unwrap<LinearActions>().type() == u32);
    }
}