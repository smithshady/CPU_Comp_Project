# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/smith3/CPU_Comp_Project/median_filter_cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/smith3/CPU_Comp_Project/median_filter_cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/vpi_sample_05_benchmark.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/vpi_sample_05_benchmark.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/vpi_sample_05_benchmark.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/vpi_sample_05_benchmark.dir/flags.make

CMakeFiles/vpi_sample_05_benchmark.dir/main.cpp.o: CMakeFiles/vpi_sample_05_benchmark.dir/flags.make
CMakeFiles/vpi_sample_05_benchmark.dir/main.cpp.o: /home/smith3/CPU_Comp_Project/median_filter_cpp/main.cpp
CMakeFiles/vpi_sample_05_benchmark.dir/main.cpp.o: CMakeFiles/vpi_sample_05_benchmark.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/smith3/CPU_Comp_Project/median_filter_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/vpi_sample_05_benchmark.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/vpi_sample_05_benchmark.dir/main.cpp.o -MF CMakeFiles/vpi_sample_05_benchmark.dir/main.cpp.o.d -o CMakeFiles/vpi_sample_05_benchmark.dir/main.cpp.o -c /home/smith3/CPU_Comp_Project/median_filter_cpp/main.cpp

CMakeFiles/vpi_sample_05_benchmark.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/vpi_sample_05_benchmark.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/smith3/CPU_Comp_Project/median_filter_cpp/main.cpp > CMakeFiles/vpi_sample_05_benchmark.dir/main.cpp.i

CMakeFiles/vpi_sample_05_benchmark.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/vpi_sample_05_benchmark.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/smith3/CPU_Comp_Project/median_filter_cpp/main.cpp -o CMakeFiles/vpi_sample_05_benchmark.dir/main.cpp.s

# Object files for target vpi_sample_05_benchmark
vpi_sample_05_benchmark_OBJECTS = \
"CMakeFiles/vpi_sample_05_benchmark.dir/main.cpp.o"

# External object files for target vpi_sample_05_benchmark
vpi_sample_05_benchmark_EXTERNAL_OBJECTS =

vpi_sample_05_benchmark: CMakeFiles/vpi_sample_05_benchmark.dir/main.cpp.o
vpi_sample_05_benchmark: CMakeFiles/vpi_sample_05_benchmark.dir/build.make
vpi_sample_05_benchmark: /opt/nvidia/vpi2/lib/aarch64-linux-gnu/libnvvpi.so.2.2.7
vpi_sample_05_benchmark: CMakeFiles/vpi_sample_05_benchmark.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/smith3/CPU_Comp_Project/median_filter_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable vpi_sample_05_benchmark"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vpi_sample_05_benchmark.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/vpi_sample_05_benchmark.dir/build: vpi_sample_05_benchmark
.PHONY : CMakeFiles/vpi_sample_05_benchmark.dir/build

CMakeFiles/vpi_sample_05_benchmark.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/vpi_sample_05_benchmark.dir/cmake_clean.cmake
.PHONY : CMakeFiles/vpi_sample_05_benchmark.dir/clean

CMakeFiles/vpi_sample_05_benchmark.dir/depend:
	cd /home/smith3/CPU_Comp_Project/median_filter_cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/smith3/CPU_Comp_Project/median_filter_cpp /home/smith3/CPU_Comp_Project/median_filter_cpp /home/smith3/CPU_Comp_Project/median_filter_cpp/build /home/smith3/CPU_Comp_Project/median_filter_cpp/build /home/smith3/CPU_Comp_Project/median_filter_cpp/build/CMakeFiles/vpi_sample_05_benchmark.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/vpi_sample_05_benchmark.dir/depend

