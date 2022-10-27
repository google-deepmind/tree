# Version >= 3.12 required for new FindPython module
# https://cmake.org/cmake/help/v3.12/release/3.12.html
cmake_minimum_required (VERSION 3.12)
project (tree LANGUAGES CXX)

# Required for Python.h and python binding.
find_package(Python3 COMPONENTS Interpreter Development)
include_directories(SYSTEM ${Python3_INCLUDE_DIRS})
if(Python3_VERSION VERSION_LESS "3.6.0")
  message(FATAL_ERROR
    "Python found ${Python3_VERSION} < 3.6.0")
endif()

# Use C++14 standard.
set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ version selection")

# Position-independent code is needed for Python extension modules.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set default build type.
if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE RELEASE
       CACHE STRING "Choose the type of build: Debug Release."
       FORCE)
endif()
message("Current build type is: ${CMAKE_BUILD_TYPE}")
message("PROJECT_BINARY_DIR is: ${PROJECT_BINARY_DIR}")

if (NOT (WIN32 OR MSVC))
  if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    # Basic build for debugging (default).
    # -Og enables optimizations that do not interfere with debugging.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Og")
  endif()

  if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
    # Optimized release build: turn off debug runtime checks
    # and turn on highest speed optimizations.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -O3")
  endif()
endif()

if(APPLE)
  # On MacOS:
  #   -undefined dynamic_lookup is necessary for pybind11 linking
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-everything -w -undefined dynamic_lookup")

  # On MacOS, we need this so that CMake will use the right Python if the user
  # has a virtual environment active
  set (CMAKE_FIND_FRAMEWORK LAST)
endif()

# Fetch pybind to be able to use pybind11_add_module symbol.
set(PYBIND_VER v2.10.1)
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG        ${PYBIND_VER}
)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
    include_directories(${pybind11_INCLUDE_DIR})
endif()

# Needed to disable Abseil tests.
set (BUILD_TESTING OFF)

# Include abseil-cpp.
set(ABSEIL_VER 20210324.2)
include(ExternalProject)
set(ABSEIL_CMAKE_ARGS
    "-DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR}/abseil-cpp"
    "-DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    "-DCMAKE_POSITION_INDEPENDENT_CODE=${CMAKE_POSITION_INDEPENDENT_CODE}"
    "-DLIBRARY_OUTPUT_PATH=${CMAKE_SOURCE_DIR}/abseil-cpp/lib")
if(DEFINED CMAKE_OSX_ARCHITECTURES)
    set(ABSEIL_CMAKE_ARGS
        ${ABSEIL_CMAKE_ARGS}
        "-DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}")
endif()
ExternalProject_Add(abseil-cpp
  GIT_REPOSITORY    https://github.com/abseil/abseil-cpp.git
  GIT_TAG           ${ABSEIL_VER}
  PREFIX            ${CMAKE_SOURCE_DIR}/abseil-cpp
  CMAKE_ARGS        ${ABSEIL_CMAKE_ARGS}
)
ExternalProject_Get_Property(abseil-cpp install_dir)
set(abseil_install_dir ${install_dir})
include_directories (${abseil_install_dir}/include)


# Define pybind11 tree module.
pybind11_add_module(_tree tree.h tree.cc)
add_dependencies(_tree abseil-cpp)

if (WIN32 OR MSVC)
    set(ABSEIL_LIB_PREF "absl")
    set(LIB_SUFF "lib")
else()
    set(ABSEIL_LIB_PREF "libabsl")
    set(LIB_SUFF "a")
endif()

# Link abseil static libs.
# We don't use find_library here to force cmake to build abseil before linking.
set(ABSEIL_LIBS int128 raw_hash_set raw_logging_internal strings throw_delegate)
foreach(ABSEIL_LIB IN LISTS ABSEIL_LIBS)
  target_link_libraries(_tree PRIVATE
      "${abseil_install_dir}/lib/${ABSEIL_LIB_PREF}_${ABSEIL_LIB}.${LIB_SUFF}")
endforeach()

# Make the module private to tree package.
set_target_properties(_tree PROPERTIES OUTPUT_NAME tree/_tree)


