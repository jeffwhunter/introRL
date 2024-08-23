#include <string>

#include <indicators/progress_bar.hpp>

#include "introRL/types.hpp"
#include "introRL/utils.hpp"

namespace irl
{
    indicators::ProgressBar makeBar(
        std::string_view title,
        indicators::Color colour,
        ProgressWidth progressWidth,
        ProgressTicks progressTicks)
    {
        return indicators::ProgressBar{
            indicators::option::MaxProgress{progressTicks.unwrap<ProgressTicks>()},
            indicators::option::ForegroundColor{colour},
            indicators::option::BarWidth{progressWidth.unwrap<ProgressWidth>()},
            indicators::option::Start{"["},
            indicators::option::Fill{"="},
            indicators::option::Lead{">"},
            indicators::option::Remainder{" "},
            indicators::option::End{"]"},
            indicators::option::PrefixText{title},
            indicators::option::ShowRemainingTime{true}};
    }
}