# RL : Programming Exercises
C++ solutions to the exercises labelled '(programming)' from the excellent [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html) by Sutton & Barto.

## Running
Despite using cmake, I have only tested this repo on windows 11 with MSVC v143 - VS 2022.
This project uses [vcpkg](https://github.com/microsoft/vcpkg) to manage dependencies, so you need that. You will also need to set the environment variable `VCPKG_ROOT` to the root folder of vcpkg (it contains `scripts/`).
I wasn't able to get [arrayfire](https://github.com/arrayfire/arrayfire) working through vcpkg, so that must be [installed manually](https://arrayfire.com/blog/learning-arrayfire-from-scratch-installation/).
Once the dependencies are dealt with, clone this repository, open cmd, cd into the repo directory, and run `config-cmake.bat`. This should generate an appropriate project for your IDE in `build/`.

## Exercises
* [2.5](https://github.com/jeffwhunter/introrl/tree/main/exercises/2.5)