add_executable(2.11 2.11.cpp README.md)
set_property(TARGET 2.11 PROPERTY FOLDER "Exercises")
target_compile_definitions(2.11 PRIVATE NOMINMAX=1)
target_compile_features(2.11 PRIVATE cxx_std_23)
target_link_libraries(
	2.11
	PRIVATE
	introRL
	ArrayFire::af
	indicators::indicators
	Matplot++::matplot)