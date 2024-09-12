#pragma once

#include <concepts>
#include <map>
#include <unordered_map>

namespace irl
{
    /// <summary>
    /// Models std::map, std::unordered_map, and literally nothing else.
    /// </summary>
    template <class T>
    concept CIsMap =
        std::same_as<
            T,
            std::map<
                typename T::key_type,
                typename T::mapped_type,
                typename T::key_compare,
                typename T::allocator_type>>
        || std::same_as<
            T,
            std::unordered_map<
                typename T::key_type,
                typename T::mapped_type,
                typename T::hasher,
                typename T::key_equal,
                typename T::allocator_type>>;

    /// <summary>
    /// Models types that are ranges of specific element types.
    /// </summary>
    template <class T, class E>
    concept CRangeOf =
        std::ranges::range<T> &&
        std::same_as<std::ranges::range_value_t<T>, E>;
}