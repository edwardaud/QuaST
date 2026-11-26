# Copyright (C) 2015 - 2021 by the authors of the ASPECT code.
#
# This file is part of ASPECT.
#
# ASPECT is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
#
# ASPECT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ASPECT; see the file LICENSE.  If not see
# <http://www.gnu.org/licenses/>.
#
# Provenance note added by the QuaST repository: this is the build file that
# ships with ASPECT 2.5.0-pre (main, 340c0d018) as
# benchmarks/davies_et_al/case-2.3-plugin/CMakeLists.txt in the pinned image
# gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac,
# tracked here unchanged so the `VoT` plugin of aspect_runs/annulus_demo can be
# rebuilt from the tracked source.  It is a work derived from ASPECT and is
# distributed under the GNU General Public License, version 2 or any later
# version; the licence text is in LICENSE beside this file.
# SPDX-License-Identifier: GPL-2.0-or-later
#
# Rebuild (inside the pinned image, from a hosts directory mounted at /work):
#
#   docker run --rm -v <this dir>:/work -w /work \
#       gassmoeller/aspect@sha256:e91937f39770efbc478699a62f41a1390644d40aec5d81b347cacbc4dae98cac \
#       bash -lc "mkdir -p build && cd build && \
#                 cmake -DAspect_DIR=/home/dealii/aspect/build .. && make -j4"
#
# The resulting build/libVoT.so is then named in demo.prm through
# `set Additional shared libraries` instead of the image's prebuilt
# libVoT.release.so (see README.md).

CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)

FIND_PACKAGE(Aspect 2.4.0 QUIET HINTS ${Aspect_DIR} ../ ../../ $ENV{ASPECT_DIR})

IF (NOT Aspect_FOUND)
  MESSAGE(FATAL_ERROR "\n"
    "Could not find a valid ASPECT build/installation directory. "
    "Please specify the directory where you are building ASPECT by passing\n"
    "   -D Aspect_DIR=<path to ASPECT>\n"
    "to cmake or by setting the environment variable ASPECT_DIR in your shell "
    "before calling cmake. See the section 'How to write a plugin' in the "
    "manual for more information.")
ENDIF ()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

SET(TARGET "VoT")
PROJECT(${TARGET})

ADD_LIBRARY(${TARGET} SHARED VoT.cc)
ASPECT_SETUP_PLUGIN(${TARGET})
