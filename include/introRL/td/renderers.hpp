#pragma once

#include <ranges>
#include <vector>

#include <blend2d.h>

#include "introRL/td/concepts.hpp"
#include "introRL/td/environments.hpp"
#include "introRL/td/types.hpp"

namespace irl::td
{
    /// <summary>
    /// Defines the layout of an image that shows a number of SARSA taught agents acting
    /// in a windy gridworld.
    /// </summary>
    struct Layout
    {
        /// <summary>
        /// The length of the edge of a grid space.
        /// </summary>
        size_t grid{};

        /// <summary>
        /// The number of columns in the grid world.
        /// </summary>
        size_t columns{};

        /// <summary>
        /// The number of rows in the grid world.
        /// </summary>
        size_t rows{};

        /// <summary>
        /// The x offset of text that should be centered in a cell.
        /// </summary>
        double xTextOffset{};

        /// <summary>
        /// The y offset of text that should be centered in a cell.
        /// </summary>
        double yTextOffset{};
    };

    /// <summary>
    /// Renders a windy grid world.
    /// </summary>
    /// <param name="context">- The context to render into.</param>
    /// <param name="position">
    /// - The top left of where the environment will be rendered.
    /// </param>
    /// <param name="layout">- The layout of the rendered image.</param>
    /// <param name="environment">- The windy grid world to render.</param>
    /// <param name="font">- The font to use for text.</param>
    void renderEnvironment(
        BLContext& context,
        BLPoint position,
        Layout layout,
        const CEnvironment auto& environment,
        const BLFont& font)
    {
        const size_t totalWidth{layout.columns * layout.grid};
        const size_t totalHeight{layout.rows * layout.grid};

        context.setStrokeWidth(1);
        context.setStrokeStyle(BLRgba32{0xFF000000});

        for (auto c : std::views::iota(0U, layout.columns + 1))
        {
            BLPath path{};
            auto x{static_cast<double>(c * layout.grid)};

            path.moveTo(position + BLPoint{x, 0.});
            path.lineTo(position + BLPoint{x, static_cast<double>(totalHeight)});

            context.strokePath(path);
        }

        for (auto r : std::views::iota(0U, layout.rows + 1))
        {
            BLPath path{};
            auto y{static_cast<double>(r * layout.grid)};

            path.moveTo(position + BLPoint{0., y});
            path.lineTo(position + BLPoint{static_cast<double>(totalWidth), y});

            context.strokePath(path);
        }

        context.setFillStyle(BLRgba32{0xFF000000});

        for (auto c : std::views::iota(size_t{0}, layout.columns))
        {
            context.fillUtf8Text(
                position +
                BLPoint{
                    (static_cast<double>(c) + layout.xTextOffset) * layout.grid,
                    totalHeight + layout.grid * layout.yTextOffset},
                font,
                std::format("{}", std::abs(environment.wind(c))).data());
        }

        const auto& start{environment.start()};
        const auto& goal{environment.goal()};

        context.fillUtf8Text(
            position +
            BLPoint{
                (start.x() + layout.xTextOffset) * layout.grid,
                (start.y() + layout.yTextOffset) * layout.grid},
                font,
                "S");

        context.fillUtf8Text(
            position +
            BLPoint{
                (goal.x() + layout.xTextOffset) * layout.grid,
                (goal.y() + layout.yTextOffset) * layout.grid},
                font,
                "G");
    }

    /// <summary>
    /// Renders one episode.
    /// </summary>
    /// <param name="context">- The context to render into.</param>
    /// <param name="position">
    /// - The top left of where the environment will be rendered.
    /// </param>
    /// <param name="layout">- The layout of the rendered image.</param>
    /// <param name="episode">
    /// - A sequence of states that traces some agent's trajectory.
    /// </param>
    /// <param name="colour">- The colour to render the trajectory in.</param>
    static void renderEpisode(
        BLContext& context,
        BLPoint position,
        Layout layout,
        const Episode& episode,
        const BLRgba32 colour)
    {
        context.setStrokeWidth(1);
        context.setStrokeStyle(colour);

        for (const auto& w : episode | std::views::slide(2))
        {
            const State& from{w[0]};
            const State& to{w[1]};

            auto stateToPos{
                [&](const State& state)
                {
                    BLPoint doubleState{
                        static_cast<double>(state.x()),
                        static_cast<double>(state.y())};

                    return (doubleState + .5f) * layout.grid;
                }};

            BLPath path{};
            path.moveTo(position + stateToPos(from));
            path.lineTo(position + stateToPos(to));

            context.strokePath(path);
        }
    }

    /// <summary>
    /// Renders a number of episodes.
    /// </summary>
    /// <param name="context">- The context to render into.</param>
    /// <param name="position">
    /// - The top left of where the environment will be rendered.
    /// </param>
    /// <param name="layout">- The layout of the rendered image.</param>
    /// <param name="environment">- The windy grid world to render.</param>
    /// <param name="demo">- A sequence of episodes.</param>
    /// <param name="font">- The font to use for text.</param>
    /// <param name="colours">- The colours to render the episodes in.</param>
    void renderDemo(
        BLContext& context,
        BLPoint position,
        Layout layout,
        const CEnvironment auto& environment,
        CRangeOf<Episode> auto&& demo,
        CRangeOf<BLRgba32> auto&& colours,
        const BLFont& font)
    {
        renderEnvironment(context, position, layout, environment, font);
        for (auto&& [episode, colour] :
            std::views::zip(
                std::forward<decltype(demo)>(demo),
                std::forward<decltype(colours)>(colours)))
        {
            renderEpisode(context, position, layout, episode, colour);
        }
    }
}