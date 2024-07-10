add_executable(4.7 4.7.cpp README.md)
set_property(TARGET 4.7 PROPERTY FOLDER "Exercises")
target_compile_definitions(4.7 PRIVATE NOMINMAX=1)
target_compile_features(4.7 PRIVATE cxx_std_23)
target_link_libraries(
	4.7
	PRIVATE
	introRL
	ArrayFire::af
	indicators::indicators
	Matplot++::matplot)