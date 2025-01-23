# Version >= 3.24 required for new `FindPython` module and `FIND_PACKAGE_ARGS`
# keyword of `FetchContent` module.
# https://cmake.org/cmake/help/v3.24/release/3.24.html
cmake_minimum_required(VERSION 3.24)

cmake_policy(SET CMP0135 NEW)

project (tree LANGUAGES CXX)

option(USE_SYSTEM_ABSEIL "Force use of system abseil-cpp" OFF)
option(USE_SYSTEM_PYBIND11 "Force use of system pybind11" OFF)

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

# Use `FetchContent` module to manage all external dependencies (i.e.
# abseil-cpp and pybind11).
include(FetchContent)

# Needed to disable Abseil tests.
set(BUILD_TESTING OFF)

# Try to find abseil-cpp package system-wide first.
if (USE_SYSTEM_ABSEIL)
    message(STATUS "Use system abseil-cpp: ${USE_SYSTEM_ABSEIL}")
    set(ABSEIL_FIND_PACKAGE_ARGS FIND_PACKAGE_ARGS)
endif()

# Include abseil-cpp.
set(ABSEIL_REPO https://github.com/abseil/abseil-cpp)
set(ABSEIL_CMAKE_ARGS
  "-DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR}/abseil-cpp"
  "-DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}"
  "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
  "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  "-DCMAKE_POSITION_INDEPENDENT_CODE=${CMAKE_POSITION_INDEPENDENT_CODE}"
  "-DLIBRARY_OUTPUT_PATH=${CMAKE_SOURCE_DIR}/abseil-cpp/lib"
  "-DABSL_PROPAGATE_CXX_STD=ON")
if(DEFINED CMAKE_OSX_ARCHITECTURES)
  set(ABSEIL_CMAKE_ARGS
      ${ABSEIL_CMAKE_ARGS}
      "-DCMAKE_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}")
endif()

FetchContent_Declare(
  absl
  URL         ${ABSEIL_REPO}/archive/refs/tags/20220623.2.tar.gz
  URL_HASH    SHA256=773652c0fc276bcd5c461668dc112d0e3b6cde499600bfe3499c5fdda4ed4a5b
  CMAKE_ARGS  ${ABSEIL_CMAKE_ARGS}
  EXCLUDE_FROM_ALL
  ${ABSEIL_FIND_PACKAGE_ARGS})

# Try to find pybind11 package system-wide first.
if (USE_SYSTEM_PYBIND11)
    message(STATUS "Use system pybind11: ${USE_SYSTEM_PYBIND11}")
    set(PYBIND11_FIND_PACKAGE_ARGS FIND_PACKAGE_ARGS)
endif()

FetchContent_Declare(
  pybind11
  URL       https://github.com/pybind/pybind11/archive/refs/tags/v2.10.1.tar.gz
  URL_HASH  SHA256=111014b516b625083bef701df7880f78c2243835abdb263065b6b59b960b6bad
  ${PYBIND11_FIND_PACKAGE_ARGS})

FetchContent_MakeAvailable(absl pybind11)

# Define pybind11 tree module.
pybind11_add_module(_tree tree.h tree.cc)

target_link_libraries(
  _tree
  PRIVATE
    absl::int128
    absl::raw_hash_set
    absl::raw_logging_internal
    absl::strings
    absl::throw_delegate)

# Make the module private to tree package.
set_target_properties(_tree PROPERTIES OUTPUT_NAME tree/_tree)


