# SPDX-License-Identifier: GPL-2.0-or-later
# This file is part of the QuaST ASPECT plugin; the licence text and the reason
# for the GPL are in LICENSE beside this file and in the top-level NOTICE.
# Build the `realistic composite` material model plugin for the pinned ASPECT
# image gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac
# (ASPECT 2.5.0-pre (main, 340c0d018), deal.II 9.4.0, image created 2023-04-19;
# the tag gassmoeller/aspect:latest currently resolves to this digest).
#
#   docker run --rm -v <this dir>:/work -w /work \
#       gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac \
#       bash -lc \
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
