add_executable(4.9 4.9.cpp README.md)
set_property(TARGET 4.9 PROPERTY FOLDER "Exercises")
target_compile_definitions(4.9 PRIVATE NOMINMAX=1)
target_compile_features(4.9 PRIVATE cxx_std_23)
target_link_libraries(
	4.9
	PRIVATE
	introRL
	ArrayFire::af
	indicators::indicators
	Matplot++::matplot)