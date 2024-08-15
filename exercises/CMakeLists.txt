function(add_exercise file chapter exercise)
    add_executable(${exercise} ${file}/${file}.cpp ${file}/README.md)
    set_property(TARGET ${exercise} PROPERTY FOLDER "Exercises/Chapter ${chapter}")
    target_compile_definitions(${exercise} PRIVATE NOMINMAX=1)
    target_compile_features(${exercise} PRIVATE cxx_std_23)
    target_link_libraries(
        ${exercise}
        PRIVATE
        introRL
        ArrayFire::af
        indicators::indicators
        Matplot++::matplot)
endfunction()

add_exercise(2.5 2 05)
add_exercise(2.11 2 11)
add_exercise(4.7 4 7)
add_exercise(4.9 4 9)
add_exercise(5.12 5 12)