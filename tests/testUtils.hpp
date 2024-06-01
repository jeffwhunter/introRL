#pragma once

#include <algorithm>
#include <vector>

#include <arrayfire.h>

namespace utils
{
	template <typename T>
	std::vector<T> toHost(const af::array& m)
	{
		std::vector<T> result;
		auto pHost{m.host<T>()};
		std::copy_n(pHost, m.elements(), std::back_inserter(result));
		af::freeHost(pHost);
		return result;
	}
}