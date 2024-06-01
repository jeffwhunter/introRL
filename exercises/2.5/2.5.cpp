#include <array>
#include <cmath>
#include <format>
#include <memory>
#include <random>

#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>
#include <matplot/matplot.h>

#include <introRL/charts.hpp>
#include <introRL/reinforcement.hpp>

struct ExperimentSetup
{
    std::string title;
    bool walk;
    std::shared_ptr<introRL::reinforcement::IStepSize> pStepSize;
};

const unsigned FIGURE_WIDTH{1000};
const unsigned FIGURE_HEIGHT{500};
const unsigned FONT_SIZE{5};
const unsigned PROGRESS_WIDTH{50};

const unsigned RUNS{2000};
const unsigned STEPS{10000};
const unsigned ACTIONS{10};

const float ALPHA{0.1};
const auto EPSILONS{std::to_array<float>({0, .01, .1})};

const auto SETUPS{std::to_array<ExperimentSetup>({
    {
        "1/N step",
        false,
        std::make_shared<introRL::reinforcement::AveragingStep>(RUNS, ACTIONS)
    }, {
        std::format("{} step", ALPHA),
        false,
        std::make_shared<introRL::reinforcement::ConstantStep>(ALPHA)
    }, {
        "Walk, 1/N step",
        true,
        std::make_shared<introRL::reinforcement::AveragingStep>(RUNS, ACTIONS)
    }, {
        std::format("Walk, {} step", ALPHA),
        true,
        std::make_shared<introRL::reinforcement::ConstantStep>(ALPHA)
    }})};

int main()
{
    using namespace indicators;

    af::getDefaultRandomEngine().setSeed(std::random_device{}());

    auto hFigure{matplot::figure(true)};
    hFigure->size(FIGURE_WIDTH, FIGURE_HEIGHT);

    auto nSetups{SETUPS.size()};

    matplot::tiledlayout(2, nSetups);

    matplot::ylabel(matplot::subplot(2, nSetups, 0), "reward");
    matplot::ylabel(matplot::subplot(2, nSetups, nSetups), "optimal");

    size_t tCount{0};
    size_t bCount{nSetups};

    show_console_cursor(false);

    ProgressBar bar{
        option::MaxProgress{nSetups * EPSILONS.size()},
        option::BarWidth{PROGRESS_WIDTH},
        option::Start{"["},
        option::Fill{"="},
        option::Lead{">"},
        option::Remainder{" "},
        option::End{"]"},
        option::PrefixText{"Training simple bandits"},
        option::ForegroundColor{Color::green},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}};

    bar.set_progress(0);

    for (const auto& setup : SETUPS)
    {
        std::vector<introRL::reinforcement::Evaluation> evaluations;

        for (auto epsilon : EPSILONS)
        {
            evaluations.push_back(
                introRL::reinforcement::simpleBandit(
                    RUNS,
                    STEPS,
                    ACTIONS,
                    epsilon,
                    setup.walk,
                    setup.pStepSize));

            bar.tick();
        }

        introRL::charts::evaluations(
            setup.title,
            nSetups,
            tCount++,
            bCount++,
            FONT_SIZE,
            STEPS,
            evaluations);
    }

    show_console_cursor(true);

    auto hLegend = matplot::legend(matplot::subplot(2, nSetups, 0), {});
    hLegend->box(false);
    hLegend->font_size(FONT_SIZE);
    hLegend->location(matplot::legend::general_alignment::topleft);

    matplot::show();

    return 0;
}