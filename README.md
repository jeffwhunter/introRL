# RL : Programming Exercises
C++ solutions to the exercises labelled '(programming)' from the excellent [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html) by Sutton & Barto.

## Running
MSVC has only implemented mdspan::operator[] with a std::array index, so this project requires MSVC.
This project uses [vcpkg](https://github.com/microsoft/vcpkg) to manage dependencies, so you need that. You will also need to set the environment variable `VCPKG_ROOT` to the root folder of vcpkg (it contains `scripts/`).
I wasn't able to get [arrayfire](https://github.com/arrayfire/arrayfire) working through vcpkg, so that must be [installed manually](https://arrayfire.com/blog/learning-arrayfire-from-scratch-installation/).
Once the dependencies are dealt with, clone this repository, open cmd, cd into the repo directory, and run `config-cmake.bat`. This should generate an appropriate project for your IDE in `build/`.

## Exercises
* [2.5 - Multi-armed bandits](
    https://github.com/jeffwhunter/introrl/tree/main/exercises/2.5)
* [2.11 - Bandit strategies](
    https://github.com/jeffwhunter/introrl/tree/main/exercises/2.11)
* [4.7 - Policy iteration](
    https://github.com/jeffwhunter/introrl/tree/main/exercises/4.7)
* [4.9 - Value iteration](
    https://github.com/jeffwhunter/introrl/tree/main/exercises/4.9)
* [5.12 - Monte Carlo control](
    https://github.com/jeffwhunter/introrl/tree/main/exercises/5.12)
* [6.9 - SARSA control](
    https://github.com/jeffwhunter/introrl/tree/main/exercises/6.9)
* [6.10 - SARSA self-guidance](
    https://github.com/jeffwhunter/introrl/tree/main/exercises/6.10)
* [7.2 - N-step TD](
    https://github.com/jeffwhunter/introrl/tree/main/exercises/7.2)