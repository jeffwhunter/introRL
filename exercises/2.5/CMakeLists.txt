add_executable(2.5 2.5.cpp README.md)
set_property(TARGET 2.5 PROPERTY FOLDER "Exercises")
target_compile_features(2.5 PRIVATE cxx_std_23)
target_link_libraries(
	2.5
	PRIVATE
	introRL
	ArrayFire::af
	indicators::indicators
	Matplot++::matplot)