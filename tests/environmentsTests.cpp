#include <arrayfire.h>
#include <catch2/catch_test_macros.hpp>

#include <introRL/environments.hpp>

namespace introRL::environments
{
	TEST_CASE("environments.bandit.has the same shape as actions")
	{
		auto lActions{linear::index(af::array{0u, 2u, 4u})};

		REQUIRE(bandit(af::constant(0, 3, 5, f32), lActions).dims() == lActions.dims());
	}

	TEST_CASE("environments.bandit.has the same type as qStar")
	{
		auto qStar{af::constant(0, 3, 5, f32)};

		REQUIRE(
			bandit(qStar, linear::index(af::array{0u, 2u, 4u})).type() == qStar.type());
	}
}
