#include <span>
#include <vector>

#include <blend2d.h>

#include "introRL/monte/episodes.hpp"
#include "introRL/monte/renderers.hpp"
#include "introRL/monte/types.hpp"

namespace irl::monte
{
    void renderBorder(
        BLContext& context,
        BLPoint position,
        Layout layout)
    {
        context.setFillStyle(BLRgba32{0xFFFFFFFF});
        context.fillBox(
            position.x + 0,
            position.y + 0,
            position.x + layout.single.x,
            position.y + layout.tileSize);
        context.fillBox(
            position.x + 0,
            position.y + 0,
            position.x + layout.tileSize,
            position.y + layout.single.y);
        context.fillBox(
            position.x + layout.single.x - layout.tileSize,
            position.y + 0,
            position.x + layout.single.x,
            position.y + layout.single.y);
        context.fillBox(
            position.x + 0,
            position.y + layout.single.y - layout.tileSize,
            position.x + layout.single.x,
            position.y + layout.single.y);
    }

    void renderEpisode(
        BLContext& context,
        BLPoint position,
        Layout layout,
        const monte::Episode& episode,
        BLPoint pathOffset,
        std::span<BLRgba32, 2> colourPair)
    {
        using namespace irl::monte;

        context.setStrokeWidth(layout.strokeWidth);
        context.setStrokeCaps(BL_STROKE_CAP_ROUND);

        for (auto [i, w] : episode.getAllPositions()
            | std::ranges::to<std::vector>()
            | std::views::slide(2)
            | std::views::enumerate)
        {
            const Position& from{w[0]};
            const Position& to{w[1]};

            context.setStrokeStyle(colourPair[i % 2]);

            auto offsetMidpoint{
                [&](Position tile)
                {
                    return
                        (
                            BLPoint{
                                static_cast<double>(tile.x()),
                                static_cast<double>(tile.y())} +
                            pathOffset +
                            .5f
                        ) *
                        layout.tileSize;
                }};

            BLPath path{};
            path.moveTo(position + offsetMidpoint(from));
            path.lineTo(position + offsetMidpoint(to));

            context.strokePath(path);
        }
    }
}