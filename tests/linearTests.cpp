#include <vector>

#include <arrayfire.h>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <catch2/catch_test_macros.hpp>

#include <introRL/linear.hpp>

#include "testUtils.hpp"

namespace introRL::linear
{
	TEST_CASE("linear.index.linearizes indices")
	{
		REQUIRE_THAT(
			utils::toHost<unsigned>(index(af::array{0u, 2u, 4u})),
			Catch::Matchers::RangeEquals(std::vector<unsigned>{0, 7, 14}));
	}
}
