# Build the `realistic composite` material model plugin for the ASPECT image
# gassmoeller/aspect:latest (ASPECT 2.5.0-pre, deal.II 9.4.0).
#
#   docker run --rm -v <this dir>:/work -w /work \
#       gassmoeller/aspect:latest bash -lc \
#       "mkdir -p build && cd build && cmake -DAspect_DIR=/home/dealii/aspect/build .. && make -j4"
#
# The resulting build/librealistic_composite.so is loaded by realistic.prm
# through `set Additional shared libraries`.

CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)

FIND_PACKAGE(Aspect 2.4.0 QUIET HINTS ${Aspect_DIR} ../ ../../ $ENV{ASPECT_DIR})

IF (NOT Aspect_FOUND)
  MESSAGE(FATAL_ERROR "\n"
    "Could not find a valid ASPECT build/installation directory. "
    "Please specify the directory where you are building ASPECT by passing\n"
    "   -D Aspect_DIR=<path to ASPECT>\n"
    "to cmake or by setting the environment variable ASPECT_DIR in your shell "
    "before calling cmake.")
ENDIF ()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

SET(TARGET "realistic_composite")
PROJECT(${TARGET})

ADD_LIBRARY(${TARGET} SHARED realistic_composite.cc)
ASPECT_SETUP_PLUGIN(${TARGET})
