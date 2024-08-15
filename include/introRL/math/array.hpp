#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <experimental/generator>
#include <ranges>
#include <utility>

namespace irl::math
{
    /// <summary>
    /// Models types that can increment other types.
    /// </summary>
    template <class T, class O>
    concept AddsTo = requires (T t, O o)
    {
        { o + t } -> std::same_as<O>;
        { t + o } -> std::same_as<O>;
    };

    /// <summary>
    /// Increments one array with another of equal size.
    /// </summary>
    /// <typeparam name="L">The element type of the left array.</typeparam>
    /// <typeparam name="R">The element type of the right array.</typeparam>
    /// <typeparam name="S">The length of both arrays.</typeparam>
    /// <param name="lhs">- The left hand side of the += operation.</param>
    /// <param name="rhs">- The right hand size of the += operation.</param>
    /// <returns>- A references to the elementwise incremented array.</returns>
    template <class L, AddsTo<L> R, size_t S>
    std::array<L, S>& operator+=(std::array<L, S>& lhs, const std::array<R, S>& rhs)
    {
        std::ranges::copy(
            std::views::zip_transform(
                [](const auto& l, const auto& r) { return l + r; },
                lhs,
                rhs),
            std::begin(lhs));

        return lhs;
    }

    /// <summary>
    /// Calculates the elementwise sum of two arrays.
    /// </summary>
    /// <typeparam name="L">The element type of the left array.</typeparam>
    /// <typeparam name="R">The element type of the right array.</typeparam>
    /// <typeparam name="S">The length of both arrays.</typeparam>
    /// <param name="lhs">- The left hand size of the + operation.</param>
    /// <param name="rhs">- The right hand size of the + operation.</param>
    /// <returns>The elementwise sum of lhs and rhs.</returns>
    template <class L, AddsTo<L> R, size_t S>
    auto operator+(std::array<L, S> lhs, const std::array<R, S>& rhs)
    {
        auto l{std::move(lhs)};

        l += rhs;

        return l;
    }

    /// <summary>
    /// Models types that can divide into a specific type.
    /// </summary>
    template <class T, class N>
    concept DividesUnder = requires (T t, N n)
    {
        { n / t } -> std::same_as<N>;
    };

    /// <summary>
    /// Models types that can be static cast to another type.
    /// </summary>
    template <class T, class O>
    concept CastsTo = requires (T t)
    {
        { static_cast<O>(t) } -> std::same_as<O>;
    };

    /// <summary>
    /// Divides a discrete array into a continuous one.
    /// </summary>
    /// <typeparam name="D">The type of the denominator.</typeparam>
    /// <typeparam name="N">The element type of the numerator.</typeparam>
    /// <typeparam name="S">The size of the numerator and output arrays.</typeparam>
    /// <typeparam name="O">The element type of the output array.</typeparam>
    /// <param name="num">- The numerator array.</param>
    /// <param name="den">- The denominator scalar.</param>
    /// <returns>
    /// The elementwise division of the numerator over the denominator.
    /// </returns>
    template <CastsTo<float> N, DividesUnder<float> D, size_t S>
    std::array<float, S> operator/(const std::array<N, S>& num, D den)
    {
        std::array<float, S> result{};

        std::ranges::copy(
            std::views::transform(
                num,
                [=](const auto& n) { return static_cast<float>(n) / den; }),
            std::begin(result));

        return result;
    }

    /// <summary>
    /// Elementwise casts one array into another of the same size.
    /// </summary>
    /// <typeparam name="O">The element type of the output array.</typeparam>
    /// <typeparam name="S">The size of both arrays.</typeparam>
    /// <typeparam name="I">The element type of the input array.</typeparam>
    /// <param name="a">- The array to cast.</param>
    /// <returns>The elementwise cast of a into O.</returns>
    template <class O, CastsTo<O> I, size_t S>
    constexpr std::array<O, S> cast(const std::array<I, S>& a)
    {
        constexpr auto helper{
            [] <size_t ... Is>
            (const std::array<I, S>&a, std::index_sequence<Is ...>)
            {
                return std::array<O, S>{{static_cast<O>(a[Is]) ...}};
            }};

        return helper(a, std::make_index_sequence<S>());
    }

    /// <summary>
    /// Models types that can be rounded then converted to another type.
    /// </summary>
    template <class T, class O>
    concept RoundsTo = requires (T t)
    {
        { std::round(t) } -> std::convertible_to<O>;
    };

    /// <summary>
    /// Elementwise rounds one array into another of the same size.
    /// </summary>
    /// <typeparam name="O">The elment type of the output array.</typeparam>
    /// <typeparam name="I">The element type of the input array.</typeparam>
    /// <typeparam name="S">The size of both arrays.</typeparam>
    /// <param name="a">- The array to round.</param>
    /// <returns>The elementwise rounding of a into O.</returns>
    template <class O, RoundsTo<O> I, size_t S>
    constexpr std::array<O, S> round(const std::array<I, S>& a)
    {
        constexpr auto helper{
            [] <size_t ... Is>
            (const std::array<I, S>&a, std::index_sequence<Is ...>)
            {
                return std::array<O, S>{{static_cast<O>(std::round(a[Is])) ...}};
            }};

        return helper(a, std::make_index_sequence<S>());
    }

    /// <summary>
    /// Generates all of the grid world steps between start and start + diff.
    /// </summary>
    /// <typeparam name="U">The element type of diff.</typeparam>
    /// <typeparam name="S">The size of start and diff.</typeparam>
    /// <typeparam name="T">The element type of start.</typeparam>
    /// <param name="start">- The starting point of the interpolation.</param>
    /// <param name="diff">- The total difference over the interpolation.</param>
    /// <returns>
    /// A sequence of all grid world steps between start and start + diff.
    /// </returns>
    template <CastsTo<float> T, RoundsTo<int> U, size_t S>
    std::experimental::generator<std::array<T, S>> interp(
        const std::array<T, S>& start,
        const std::array<U, S>& diff)
    {
        size_t n{
            std::ranges::max(
                round<int>(diff)
                | std::views::transform(
                    [](int x)
                    {
                        return static_cast<size_t>(std::abs(x));
                    }))};

        if (n == 0) co_return;

        auto chunk{diff / n};
        auto result{cast<float>(start)};

        for (auto _ : std::views::iota(0U, n))
        {
            result += chunk;

            co_yield round<T>(result);
        }
    }
}