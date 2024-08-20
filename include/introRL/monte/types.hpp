# pragma once

#include <stronk/stronk.h>

#include <array>
#include <concepts>
#include <utility>

#include "introRL/types.hpp"

namespace irl::monte
{
    /// <summary>
    /// A change in velocity in a grid world.
    /// </summary>
    struct Action :
        twig::stronk<
            Action,
            std::array<int, 2>,
            twig::can_equate,
            twig::can_index,
            twig::can_iterate,
            twig::can_order,
            Makeable<int, 0, 2>::Skill,
            XY<1, 0>::Skill>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// A change in position in a grid world.
    /// </summary>
    struct Velocity :
        twig::stronk<
            Velocity,
            std::array<int, 2>,
            twig::can_equate,
            twig::can_index,
            twig::can_order,
            Antidelta<Action>::Skill,
            Makeable<int, 0, 2>::Skill,
            XY<1, 0>::Skill>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// A position in a grid world.
    /// </summary>
    struct Position :
        twig::stronk<
            Position,
            std::array<size_t, 2>,
            twig::can_equate,
            twig::can_index,
            twig::can_order,
            Antidelta<Velocity>::Skill,
            Makeable<size_t, 0, 2>::Skill,
            XY<1, 0>::Skill>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// The state of a race car in a grid world.
    /// </summary>
    struct State
    {
        Position position{Position::make()};
        Velocity velocity{Velocity::make()};

        auto operator<=>(const State&) const = default;
    };

}