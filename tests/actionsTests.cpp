#include <vector>

#include <arrayfire.h>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <catch2/catch_test_macros.hpp>

#include <introRL/actions.hpp>

#include "testUtils.hpp"

namespace introRL::actions
{
	TEST_CASE("actions.explore.has the proper shape")
	{
		REQUIRE(explore(5, 7).dims() == af::dim4{5});
	}

	TEST_CASE("actions.explore.has the proper type")
	{
		REQUIRE(explore(2, 3).type() == u32);
	}

	TEST_CASE("actions.eGreedy.has the proper shape")
	{
		REQUIRE(eGreedy(af::constant(0, 4, 8, f32), 0).dims() == af::dim4{4});
	}

	TEST_CASE("actions.eGreedy.has the proper type")
	{
		REQUIRE(eGreedy(af::constant(0, 4, 8, f32), 0).type() == u32);
	}

	TEST_CASE("actions.eGreedy.is greedy when epsilon is zero")
	{
		REQUIRE_THAT(
			utils::toHost<unsigned>(
				eGreedy(
					af::moddims(
						af::range(af::dim4{15}, 0, f32),
						af::dim4{3, 5})
					% 5,
					0)),
			Catch::Matchers::RangeEquals(std::vector<unsigned>{3, 1, 4}));
	}

	TEST_CASE("actions.greedy.returns highest index")
	{
		REQUIRE_THAT(
			utils::toHost<unsigned>(
				greedy(
					af::moddims(
						af::range(af::dim4{15}, 0, f32),
						af::dim4{3, 5})
					% 5)),
			Catch::Matchers::RangeEquals(std::vector<unsigned>{3, 1, 4}));
	}
}