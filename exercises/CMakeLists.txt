function(add_exercise file chapter exercise)
    set(TARGET_NAME ${chapter}.${exercise})
    add_executable(${TARGET_NAME} ${file}/${file}.cpp ${file}/README.md)
    set_property(TARGET ${TARGET_NAME} PROPERTY FOLDER "Exercises/Chapter ${chapter}")
    target_compile_definitions(${TARGET_NAME} PRIVATE NOMINMAX=1)
    target_compile_features(${TARGET_NAME} PRIVATE cxx_std_23)
    target_link_libraries(
        ${TARGET_NAME}
        PRIVATE
        introRL
        ArrayFire::af
        indicators::indicators
        Matplot++::matplot
        mp++::mp++)
endfunction()

add_exercise(2.5 2 05)
add_exercise(2.11 2 11)
add_exercise(4.7 4 7)
add_exercise(4.9 4 9)
add_exercise(5.12 5 12)
add_exercise(6.9 6 09)
add_exercise(6.10 6 10)
add_exercise(7.2 7 2)