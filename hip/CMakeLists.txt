cmake_minimum_required(VERSION 3.10)
project(FindROCShmemExample LANGUAGES CXX HIP)

# Set paths for rocSHMEM manually
# set(ROCSHMEM_INCLUDE_DIR "/home1/muhaawad/rocm/rocshmem/include")
# set(ROCSHMEM_LIBRARY "/home1/muhaawad/rocm/rocshmem/lib/librocshmem.so")

# Define a custom target for rocSHMEM
# add_library(rocshmem INTERFACE)
# target_include_directories(rocshmem INTERFACE ${ROCSHMEM_INCLUDE_DIR})
# target_link_libraries(rocshmem INTERFACE ${ROCSHMEM_LIBRARY})
# add_library(rocshmem::rocshmem ALIAS rocshmem)

# set(CMAKE_FIND_DEBUG_MODE TRUE)
find_package(rocshmem REQUIRED)

# Check and print variables defined by the package
message(STATUS "rocshmem_FOUND: ${rocshmem_FOUND}")
message(STATUS "rocshmem_VERSION: ${rocshmem_VERSION}")
message(STATUS "rocshmem_INCLUDE_DIRS: ${rocshmem_INCLUDE_DIRS}")
message(STATUS "rocshmem_LIBRARIES: ${rocshmem_LIBRARIES}")


# Find HIP
find_package(hip REQUIRED)

# Find Open MPI
find_package(MPI REQUIRED)

# Message to confirm locations of dependencies
message(STATUS "Using manually configured rocSHMEM library at: ${ROCSHMEM_LIBRARY}")
message(STATUS "Using manually configured rocSHMEM include at: ${ROCSHMEM_INCLUDE_DIR}")
message(STATUS "Open MPI include path: ${MPI_INCLUDE_PATH}")
message(STATUS "Open MPI libraries: ${MPI_LIBRARIES}")

# Add the executable
set(EXECUTABLE_NAME example)
add_executable(${EXECUTABLE_NAME} main.hip)

# Include directories
target_include_directories(${EXECUTABLE_NAME} PRIVATE ${MPI_INCLUDE_PATH})

# Set the HIP architecture
set_target_properties(${EXECUTABLE_NAME} PROPERTIES HIP_ARCHITECTURES gfx90a)

# Link rocSHMEM, MPI, and required libraries
target_link_libraries(
    ${EXECUTABLE_NAME}
    PRIVATE
      roc::rocshmem
)

# Add compile and link flags
target_compile_options(
    ${EXECUTABLE_NAME}
    PRIVATE
      -fgpu-rdc
)

target_link_options(
    ${EXECUTABLE_NAME}
    PRIVATE
      -fgpu-rdc
      --hip-link
)
