#pragma once

#include <array>
#include <algorithm>
#include <concepts>
#include <ranges>
#include <span>
#include <string>
#include <utility>

#include <blend2d.h>

#include "introRL/cartesian.hpp"
#include "introRL/monte/environments.hpp"
#include "introRL/monte/episodes.hpp"

namespace irl
{
    /// <summary>
    /// Models a race car track! Vrrooom!
    /// </summary>
    template <class T>
    concept Track = requires (T t, size_t s, std::array<size_t, 2> a)
    {
        { t.extent(s) } -> std::unsigned_integral;
        { t[a] } -> std::same_as<const monte::TrackTile&>;
    };

    /// <summary>
    /// Defines the layout of an composite image containing a number of monte carlo race
    /// car demos.
    /// </summary>
    struct Layout
    {
        /// <summary>
        /// The size of a single image in the composate.
        /// </summary>
        BLPoint single{};

        /// <summary>
        /// The side length in pixels of each track tile.
        /// </summary>
        size_t tileSize{};

        /// <summary>
        /// The relative location of the title in each image.
        /// </summary>
        BLPoint titleLocation{};

        /// <summary>
        /// The relative location of the average length description in each image.
        /// </summary>
        BLPoint lengthLocation{};

        /// <summary>
        /// The range over which race car paths are spread so they can be differentiated
        /// when they overlap.
        /// </summary>
        BLPoint pathOffset{};

        /// <summary>
        /// The width of each race car path.
        /// </summary>
        size_t strokeWidth{};
    };

    /// <summary>
    /// Renders a border around a single image in a composate.
    /// </summary>
    /// <param name="context">- The context to render into.</param>
    /// <param name="position">- The top left of the single image.</param>
    /// <param name="layout">- The layout of the composite.</param>
    void renderBorder(BLContext& context, BLPoint position, Layout layout);

    /// <summary>
    /// Renders a track inside a single image in the composite.
    /// </summary>
    /// <param name="context">- The context to render into.</param>
    /// <param name="position">- The top left of the single image.</param>
    /// <param name="layout">- The layout of the composite.</param>
    /// <param name="track">- The track to render.</param>
    void renderTrack(
        BLContext& context,
        BLPoint position,
        Layout layout,
        Track auto&& track)
    {
        for (auto&& i : mdIndices(track.extent(0), track.extent(1)))
        {
            auto&& [row, col] {i};

            switch (track[i])
            {
            case monte::X:
                context.setFillStyle(BLRgba32{0xFFFFFFFF});
                break;
            case monte::_:
                context.setFillStyle(BLRgba32{0x00000000});
                break;
            case monte::S:
                context.setFillStyle(BLRgba32{0xFF0000FF});
                break;
            case monte::F:
                context.setFillStyle(BLRgba32{0xFF00FF00});
                break;
            default:
                std::unreachable();
            }

            context.fillBox(
                position.x + (col) * layout.tileSize,
                position.y + (row) * layout.tileSize,
                position.x + (col + 1) * layout.tileSize,
                position.y + (row + 1) * layout.tileSize);
        }
    }

    /// <summary>
    /// Renders an episode inside a single image in the composite.
    /// </summary>
    /// <param name="context">- The context to render into.</param>
    /// <param name="position">- The top left of the single image.</param>
    /// <param name="layout">- The layout of the composite.</param>
    /// <param name="episode">- The episode to render.</param>
    /// <param name="pathOffset">- The individual offset of this episode's path.</param>
    /// <param name="colourPair">
    /// - The colours to alternate between when rendering the path.
    /// </param>
    void renderEpisode(
        BLContext& context,
        BLPoint position,
        Layout layout,
        const monte::Episode& episode,
        BLPoint pathOffset,
        std::span<BLRgba32, 2> colourPair);

    /// <summary>
    /// Renders one episode from each start inside a single image in the composite.
    /// </summary>
    /// <typeparam name="N">The number of colours to render paths with.</typeparam>
    /// <param name="context">- The context to render into.</param>
    /// <param name="position">- The top left of the single image.</param>
    /// <param name="layout">- The layout of the composite.</param>
    /// <param name="track">- The race car track to render.</param>
    /// <param name="episodes">- The paths from each start.</param>
    /// <param name="colourPairs">- The colours to render the paths in.</param>
    /// <param name="title">- The title of the demo.</param>
    /// <param name="font">- The font used for rendering text.</param>
    template <size_t N>
    void renderDemo(
        BLContext& context,
        BLPoint position,
        Layout layout,
        Track auto&& track,
        std::ranges::contiguous_range auto&& episodes,
        std::span<std::array<BLRgba32, 2>, N> colourPairs,
        std::string_view title,
        BLFont& font)
    {
        renderBorder(context, position, layout);

        auto borderOffset{
            position +
            BLPoint{
                static_cast<double>(layout.tileSize),
                static_cast<double>(layout.tileSize)}};

        renderTrack(context, borderOffset, layout, track);

        BLPoint d{layout.pathOffset};
        d /= (std::size(episodes) - 1);

        for (auto&& [i, episode] : episodes | std::views::enumerate)
            renderEpisode(
                context,
                borderOffset,
                layout,
                episode,
                i * d - layout.pathOffset / 2,
                colourPairs[i % N]);

        auto averageLength{
            static_cast<float>(
                std::ranges::fold_left(
                    episodes
                    | std::views::transform(
                        [](const auto& episode) { return episode.bigT(); }),
                    0U,
                    std::plus<size_t>{})) /
                std::size(episodes)};

        context.setFillStyle(BLRgba32{0xFF000000});
        context.fillUtf8Text(position + layout.titleLocation, font, title.data());
        context.fillUtf8Text(
            position + layout.lengthLocation,
            font,
            std::format("avg length: {:.1f}", averageLength).data());
    }

    /// <summary>
    /// Renders a number of demos (a collection of episodes) from a number of agents into
    /// a single image. Each agent will have it's demo rendered into separate sub images
    /// within a composite.
    /// </summary>
    /// <typeparam name="N">The number of colours to render paths with.</typeparam>
    /// <param name="fileName">- The file to render into.</param>
    /// <param name="layout">- The layout of the image.</param>
    /// <param name="track">- The track to render.</param>
    /// <param name="demoTitles">
    /// - A zipped range of demos and titles, one per agent.
    /// </param>
    /// <param name="colourPairs">- The colours to render agent paths with.</param>
    /// <param name="font">- The font to use for text.</param>
    template <size_t N>
    void renderDemos(
        std::string_view fileName,
        Layout layout,
        Track auto&& track,
        std::ranges::range auto&& demoTitles,
        std::span<std::array<BLRgba32, 2>, N> colourPairs,
        BLFont& font)
    {
        BLImage image{
            static_cast<int>(layout.single.x * std::size(demoTitles)),
            static_cast<int>(layout.single.y),
            BL_FORMAT_PRGB32};

        BLContext context{image};

        context.clearAll();
        context.setFillStyle(BLRgba32{0xFF000000});
        context.fillAll();

        for (const auto& [i, demoTitle] : demoTitles | std::views::enumerate)
        {
            const auto& [demo, title] {demoTitle};

            renderDemo(
                context,
                BLPoint{static_cast<double>(i * layout.single.x), 0},
                layout,
                track,
                demo,
                colourPairs,
                title,
                font);
        }

        context.end();

        image.writeToFile(fileName.data());
    }
}