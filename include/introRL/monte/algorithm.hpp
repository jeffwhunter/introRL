#pragma once

#include <concepts>
#include <functional>
#include <map>
#include <random>
#include <ranges>
#include <set>
#include <utility>
#include <vector>

#include "introRL/math/map.hpp"
#include "introRL/math/sparse.hpp"
#include "introRL/monte/agents.hpp"
#include "introRL/monte/episodes.hpp"
#include "introRL/monte/types.hpp"
#include "introRL/stats.hpp"
#include "introRL/types.hpp"

namespace irl::monte
{
    /// <summary>
    /// Models an reinforcement agent that acts in response to states.
    /// </summary>
    template <class T>
    concept CAgent = requires (T agent, State state, float pExplore)
    {
        { agent.act(state, pExplore) } -> std::same_as<Probable<Action>>;
    };

    /// <summary>
    /// Models a reinforcement learning environment.
    /// </summary>
    template <class T>
    concept CEnvironment = requires (T environment, State state, Action action)
    {
        { environment.reset() } -> std::same_as<State>;
        { environment.step(state, action) } -> std::same_as<State>;
        { environment.done(state) } -> std::same_as<bool>;
        { environment.starts() } -> std::same_as<const std::set<Position>&>;
    };

    /// <summary>
    /// Generates an episode from a specific starting state.
    /// </summary>
    /// <typeparam name="MAX_EPISODE_STEPS">
    /// The maximum number of steps in the episode.
    /// </typeparam>
    /// <param name="agent">- The agent to generate the episode with.</param>
    /// <param name="environment">- The environment to generate the episode in.</param>
    /// <param name="state">- The starting state of the episode.</param>
    /// <param name="pExplore">- The chance for the agent to explore each turn.</param>
    /// <returns>The generated episode.</returns>
    template <StepCount MAX_EPISODE_STEPS>
    [[nodiscard]] auto generateEpisode(
        CAgent auto& agent,
        CEnvironment auto& environment,
        State state,
        float pExplore)
    {
        Episode episode{};

        for (const unsigned i :
            std::views::iota(0U, MAX_EPISODE_STEPS.unwrap<StepCount>())
            | std::views::take_while(
                [&](const unsigned) { return !environment.done(state); }))
        {
            Probable<Action> probableAction{agent.act(state, pExplore)};

            episode.append(state, probableAction);

            state = environment.step(state, probableAction.value);
        }

        if (episode.bigT() < MAX_EPISODE_STEPS.unwrap<StepCount>())
        {
            episode.setFinalPosition(state.position);
        }

        return episode;
    }

    /// <summary>
    /// Generates one episode for each starting state in the environment.
    /// </summary>
    /// <typeparam name="MAX_EPISODE_STEPS">
    /// The maximum number of steps in the episode.
    /// </typeparam>
    /// <param name="agent">- The agent to generate the episode with.</param>
    /// <param name="environment">- The environment to generate the episodes in.</param>
    /// <returns>A vector of generated episodes.</returns>
    template <StepCount MAX_EPISODE_STEPS>
    [[nodiscard]] auto demoAllStarts(
        CAgent auto& agent,
        CEnvironment auto& environment)
    {
        return environment.starts()
            | std::views::transform(
                [&](const auto& start)
                {
                    return generateEpisode<MAX_EPISODE_STEPS>(
                        agent,
                        environment,
                        State{start, Velocity::make()},
                        0.f);
                })
            | std::ranges::to<std::vector>();
    }

    /// <summary>
    /// Rewinds an episode to implement off policy monte-carlo control. Updates a value
    /// estimates, weights, and policies for appropriate combinations of states and
    /// actions.
    /// </summary>
    /// <typeparam name="GAMMA">The discount factor in this learning process.</typeparam>
    /// <param name="c">
    /// - Importance weights that determine how much information each experience imparts.
    /// </param>
    /// <param name="q">
    /// - Action value estimates showing how much reward is expected for each action.
    /// </param>
    /// <param name="pi">- The estimated optimal policy.</param>
    /// <param name="episode">- The episode to learn from.</param>
    template <float GAMMA>
    static void rewind(
        math::SparseMatrix<State, Action, float>& c,
        math::SparseMatrix<State, Action, float>& q,
        std::map<State, Action>& pi,
        const Episode episode,
        std::uniform_random_bit_generator auto& generator)
    {
        const auto T{episode.bigT()};
        auto g{0.f};
        auto w{1.f};

        for (auto t : std::views::iota(0U, T) | std::views::reverse)
        {
            g = GAMMA * g + (t == T - 1 ? 1.f : 0.f);
            const auto& [state, probableAction] {episode.getStep(t)};

            auto& c_sa{c(state, probableAction.value)};
            auto& q_sa{q(state, probableAction.value)};
            auto& pi_s{pi[state]};

            c_sa += w;
            q_sa += (w / c_sa) * (g - q_sa);
            pi_s = math::argmax(q(state));

            if (probableAction.value != pi_s)
            {
                return;
            }

            w /= probableAction.probability;
        }
    }

    /// <summary>
    /// Off policy monte carlo control!
    /// </summary>
    /// <typeparam name="MAX_EPISODE_STEPS">The maximum episode length.</typeparam>
    /// <typeparam name="P_EXPLORE">The chance for agents to explore.</typeparam>
    /// <param name="nEpisodes">- The number of episodes to learn from.</param>
    /// <param name="teacher"></param>
    /// <param name="environment"></param>
    /// <param name="explorer"></param>
    /// <param name="progressCallback"></param>
    /// <returns></returns>
    template <EpisodeCount N_EPISODES, StepCount MAX_EPISODE_STEPS, float P_EXPLORE>
    [[nodiscard]] auto control(
        CAgent auto& teacher,
        CEnvironment auto& environment,
        Explorer& explorer,
        std::uniform_random_bit_generator auto& generator,
        std::function<void(void)> progressCallback = []{})
    {
        math::SparseMatrix<State, Action, float> c{};
        math::SparseMatrix<State, Action, float> q{};
        std::map<State, Action> pi{};

        for (size_t t : std::views::iota(0U, N_EPISODES.unwrap<EpisodeCount>()))
        {
            auto episode{
                generateEpisode<MAX_EPISODE_STEPS>(
                    teacher,
                    environment,
                    environment.reset(),
                    P_EXPLORE)};

            if (episode.bigT() < MAX_EPISODE_STEPS.unwrap<StepCount>())
            {
                rewind<.1f>(c, q, pi, std::move(episode), generator);
            }

            progressCallback();
        }

        return TableAgent{std::move(pi), explorer};
    }
}