#pragma once

#include <concepts>
#include <cstdint>
#include <set>
#include <vector>

#include <mp++/mp++.hpp>
#include <mp++/rational.hpp>
#include <mp++/real.hpp>
#include <stronk/prefabs.h>
#include <stronk/stronk.h>

#include "introRL/math/sparse.hpp"
#include "introRL/td/skills.hpp"
#include "introRL/types.hpp"

namespace irl::td
{
    using rat1_t = mppp::rational<1>;

    /// <summary>
    /// The step size of a reinforcement learning process.
    /// </summary>
    struct Alpha :
        twig::stronk<
        Alpha,
        rat1_t,
        twig::can_forward_constructor_args,
        MultipliesInto<mppp::real>::Skill>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// A change in position in a grid world.
    /// </summary>
    struct GridAction :
        twig::stronk<
        GridAction,
        std::array<int, 2>,
        twig::can_equate,
        twig::can_order,
        Makeable<int, 0, 2>::Skill,
        XY<0, 1>::Skill>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// A position in a grid world.
    /// </summary>
    struct GridState :
        twig::stronk<
        GridState,
        std::array<size_t, 2>,
        twig::can_equate,
        twig::can_order,
        Antidelta<GridAction>::Skill,
        Makeable<size_t, 0, 2>::Skill,
        XY<0, 1>::Skill>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// A positioni in a random walk world.
    /// </summary>
    struct WalkState :
        twig::stronk<
        WalkState,
        unsigned,
        twig::can_equate,
        twig::can_hash,
        twig::can_order,
        AddsInto<int>::Skill,
        SubtractsInto<int>::Skill>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// The number of rows in a grid world.
    /// </summary>
    struct Height
        : twig::stronk<
            Height,
            size_t,
            OrdersWith<int>::Skill,
            SubtractsInto<int>::Skill>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// The number of columns in a grid world.
    /// </summary>
    struct Width
        : twig::stronk<
            Width,
            size_t,
            twig::can_multiply,
            OrdersWith<int>::Skill>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// The probability that some agent will take exploratory actions.
    /// </summary>
    struct Epsilon : twig::stronk<Epsilon, mppp::real>
    {
        using stronk::stronk;
    };

    /// <summary>
    /// A set of actions.
    /// </summary>
    using GridActions = std::set<GridAction>;

    /// <summary>
    /// One traversal from the start to the goal in a grid world.
    /// </summary>
    using Episode = std::vector<GridState>;

    /// <summary>
    /// A collection of episodes to render.
    /// </summary>
    using Demo = std::vector<Episode>;

    /// <summary>
    /// An action value table.
    /// </summary>
    using Q = math::SparseMatrix<GridState, GridAction, mppp::real>;
}