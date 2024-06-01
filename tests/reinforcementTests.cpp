#include <memory>
#include <vector>

#include <arrayfire.h>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/trompeloeil.hpp>

#include <introRL/linear.hpp>
#include <introRL/reinforcement.hpp>

#include "testUtils.hpp"

namespace introRL::reinforcement
{
	TEST_CASE("algorthms.AveragingStep.has the proper type")
	{
		REQUIRE(AveragingStep(1, 1).stepSize(linear::index(af::array{0})).type() == f32);
	}

	TEST_CASE("reinforcement.AveragingStep.decreases elementwise")
	{
		auto testee{AveragingStep(3, 3)};

		REQUIRE_THAT(
			utils::toHost<float>(testee.stepSize(linear::index(af::array{0u, 1u, 0u}))),
			Catch::Matchers::RangeEquals(std::vector<float>{1, 1, 1}));

		REQUIRE_THAT(
			utils::toHost<float>(testee.stepSize(linear::index(af::array{1u, 0u, 0u}))),
			Catch::Matchers::RangeEquals(std::vector<float>{1, 1, .5}));

		REQUIRE_THAT(
			utils::toHost<float>(testee.stepSize(linear::index(af::array{2u, 1u, 0u}))),
			Catch::Matchers::RangeEquals(std::vector<float>{1, .5, 1 / 3.}));
	}

	TEST_CASE("reinforcement.ConstantStep.has the proper type")
	{
		REQUIRE(ConstantStep(0).stepSize(af::array{0}).type() == f32);
	}

	TEST_CASE("reinforcement.ConstantStep.is constant")
	{
		float alpha{0.5};

		auto testee{ConstantStep(alpha)};

		auto linearZero{linear::index(af::array{0})};
		auto equalsAlpha{Catch::Matchers::RangeEquals(std::vector<float>{alpha})};

		REQUIRE_THAT(utils::toHost<float>(testee.stepSize(linearZero)), equalsAlpha);
		REQUIRE_THAT(utils::toHost<float>(testee.stepSize(linearZero)), equalsAlpha);
	}

	class StepSizeMock : public IStepSize
	{
	public:
		MAKE_MOCK1(stepSize, const af::array(const af::array&));
	};

	TEST_CASE("reinforcement.simpleBandit.returns Evaluations of proper size")
	{
		const int steps{5};

		auto pStepSize{std::make_shared<StepSizeMock>()};
		REQUIRE_CALL(*pStepSize, stepSize(ANY(const af::array&)))
			.RETURN(af::array{.5f})
			.TIMES(steps);

		auto evaluations{simpleBandit(3, steps, 2, 0, false, pStepSize)};

		REQUIRE(evaluations.rewards.size() == steps);
		REQUIRE(evaluations.optimality.size() == steps);
	}
}