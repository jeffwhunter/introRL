#pragma once

#include <print>
#include <string>
#include <utility>
#include <vector>

#include <blend2d.h>
#include <indicators/color.hpp>
#include <indicators/cursor_control.hpp>

#include "introRL/td/agents.hpp"
#include "introRL/td/sarsa.hpp"
#include "introRL/td/charts.hpp"
#include "introRL/td/concepts.hpp"
#include "introRL/td/renderers.hpp"
#include "introRL/td/types.hpp"
#include "introRL/utils.hpp"
#include "introRL/ticker.hpp"

namespace irl::td
{
    /// <summary>
    /// Run SARSA, putting up a nice progress bar, and return an action value estimate
    /// and learning history.
    /// </summary>
    /// <typeparam name="N_STEPS">The number of steps to run SARSA for.</typeparam>
    /// <typeparam name="TICK_RATE">
    /// The number of steps between progress bar movements.
    /// </typeparam>
    /// <typeparam name="PROGRESS_WIDTH">The width of the progress bar.</typeparam>
    /// <param name="alpha">
    /// - The step size SARSA uses to update the action value estimates.
    /// </param>
    /// <param name="environment">- The environment in which SARSA learns.</param>
    /// <param name="agent">- The agent controlling exploration.</param>
    /// <param name="actions">- The action set available to the agent.</param>
    /// <param name="title">- The title of this run.</param>
    /// <param name="colour">- The colour of the progress bar.</param>
    /// <returns>
    /// A struct holding an action value estimate and a vector showing how long each
    /// episode took.</returns>
    template <
        StepCount N_STEPS,
        ProgressWidth PROGRESS_WIDTH = ProgressWidth{50},
        unsigned TICK_RATE = 1'000>
    [[nodiscard]] SarsaResult learnSarsa(
        const Alpha& alpha,
        CSarsaEnvironment auto& environment,
        CSarsaAgent auto& agent,
        const GridActions& actions,
        std::string_view title,
        indicators::Color colour = indicators::Color::blue)
    {
        indicators::show_console_cursor(false);

        auto bar{
            makeBar(
                title,
                colour,
                PROGRESS_WIDTH,
                ProgressTicks{N_STEPS.unwrap<StepCount>() / TICK_RATE})};

        bar.set_progress(0);

        SarsaController<N_STEPS> controller{alpha, actions};

        auto result{
            controller.sarsa(
                environment,
                agent,
                Ticker<TICK_RATE>{[&] { bar.tick(); }})};

        if (!bar.is_completed())
        {
            bar.mark_as_completed();
        }

        indicators::show_console_cursor(true);

        return result;
    }

    /// <summary>
    /// Renders learning histories and episodes into a single file.
    /// </summary>
    /// <param name="fileName">- The file to render into.</param>
    /// <param name="environment">- The environment to render.</param>
    /// <param name="sarsaResults">- The learning histories to render.</param>
    /// <param name="names">- The names of the learning histories.</param>
    /// <param name="demo">- The episodes to render.</param>
    /// <param name="colours">- The colours to render those episodes in.</param>
    /// <param name="chartDimensions">- The dimensions of the learning chart.</param>
    /// <param name="layout">- The layout of the episode rendering.</param>
    /// <param name="xLim">- The max step to chart.</param>
    /// <param name="fontSize">- The font size used.</param>
    void makeImage(
        std::string_view fileName,
        CSarsaEnvironment auto& environment,
        const std::vector<SarsaResult>& sarsaResults,
        const std::vector<std::string>& names,
        CRangeOf<Episode> auto&& demo,
        CRangeOf<BLRgba32> auto&& colours,
        ChartDimensions chartDimensions,
        Layout layout,
        double xLim,
        float fontSize)
    {
        constexpr auto tempChart{"tmp.jpeg"};

        chartSarsaToFile(
            chartDimensions,
            std::move(sarsaResults),
            names,
            tempChart,
            xLim);

        BLImage chart{};
        if (chart.readFromFile(tempChart) != BL_SUCCESS)
        {
            throw std::runtime_error{"Can't read chart file"};
        }

        BLImage image{
            static_cast<int>(chartDimensions.width),
            static_cast<int>(chartDimensions.height),
            BL_FORMAT_PRGB32};
        BLContext context{image};

        context.clearAll();

        context.blitImage(BLPointI{0, 0}, chart);

        BLFontFace face{};
        if (face.createFromFile("font.ttf") != BL_SUCCESS)
        {
            throw std::runtime_error{"Can't find font.ttf"};
        }

        BLFont font{};
        font.createFromFace(face, fontSize);

        renderDemo(
            context,
            BLPoint{200, 75},
            layout,
            environment,
            std::forward<decltype(demo)>(demo),
            std::forward<decltype(colours)>(colours),
            font);

        context.end();

        image.writeToFile(fileName.data());
    }
}