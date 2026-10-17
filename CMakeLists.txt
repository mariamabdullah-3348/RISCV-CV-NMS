cmake_minimum_required(VERSION 3.22)
project(RISCV_CV VERSION 1.0.0 LANGUAGES CXX)

# =============================================================================
# RISC-V-CV: RISC-V image processing for XS-GEM5 and qemu-riscv64 user-mode.
#
# This project cross-compiles to RISC-V and builds two image kinds per test/
# benchmark: bare-metal .bin images for the XiangShan gem5 fork (cycle-accurate)
# and static Linux ELFs for qemu-riscv64 user-mode (fast functional). Bare-
# metal images link against nexus-am's am/klib runtime. Configure with the
# toolchain file:
#
#   cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64_xs_toolchain.cmake
#
# Every install location and flag below is a cache variable, so anyone can
# point the build at their own nexus-am / gem5 / toolchain without editing
# the build files. Defaults come from the AM_HOME / XIANG_GEM5 environment
# variables used by the reference setup, falling back to $HOME/<dir>.
# =============================================================================

# ---- user-tunable cache variables ------------------------------------------

# nexus-am install location.
if(NOT DEFINED NEXUS_AM_HOME)
  if(DEFINED ENV{AM_HOME})
    set(NEXUS_AM_HOME "$ENV{AM_HOME}")
  else()
    set(NEXUS_AM_HOME "$ENV{HOME}/nexus-am")
  endif()
endif()
set(NEXUS_AM_HOME "${NEXUS_AM_HOME}" CACHE PATH "Path to the nexus-am checkout")

# gem5 (XiangShan fork) install location and entry points.
if(NOT DEFINED GEM5_HOME)
  if(DEFINED ENV{XIANG_GEM5})
    set(GEM5_HOME "$ENV{XIANG_GEM5}")
  else()
    set(GEM5_HOME "$ENV{HOME}/GEM5")
  endif()
endif()
set(GEM5_HOME "${GEM5_HOME}" CACHE PATH "Path to the XS-GEM5 checkout")
set(GEM5_BINARY "${GEM5_HOME}/build/RISCV/gem5.opt"
    CACHE FILEPATH "gem5 binary used to run images")
set(GEM5_CONFIG "${GEM5_HOME}/configs/example/kmhv3.py"
    CACHE FILEPATH "gem5 config script (KMHV3 O3 model)")
set(GEM5_EXTRA_ARGS "--disable-difftest"
    CACHE STRING "Extra arguments passed to the gem5 config script")

# qemu-riscv64 user-mode emulator.
if(NOT QEMU_BINARY)
  find_program(QEMU_BINARY NAMES qemu-riscv64 qemu-riscv64-static)
endif()
set(QEMU_BINARY "${QEMU_BINARY}" CACHE FILEPATH "qemu-riscv64 user-mode emulator")
# max is required so QEMU enables the XiangShan bitmanip/crypto extensions in RISCV_MARCH; rv64,v=true faults with illegal instruction.
set(QEMU_CPU "max"
    CACHE STRING "QEMU -cpu (max enables RVV + bitmanip/crypto used by RISCV_MARCH)")
set(QEMU_EXTRA_ARGS "" CACHE STRING "Extra qemu-riscv64 arguments")
option(BUILD_QEMU_TARGETS "Build optional QEMU user-mode targets (gem5 is the required evaluation target)" OFF)

# RISC-V cross toolchain.
set(RISCV_TOOLCHAIN_PREFIX "riscv64-linux-gnu-" CACHE STRING "GNU RISC-V toolchain triple prefix")
set(AM_ARCH "riscv64-xs" CACHE STRING "nexus-am target architecture")
set(RISCV_MABI "lp64d" CACHE STRING "RISC-V -mabi")
set(RISCV_MARCH
    "rv64gcv_zba_zbb_zbc_zbs_zbkb_zbkc_zbkx_zknd_zkne_zknh_zkr_zksed_zksh_zkt"
    CACHE STRING "RISC-V -march (kept in sync with the toolchain file)")

# ---- configure-time sanity checks ------------------------------------------
if(NOT EXISTS "${NEXUS_AM_HOME}")
  message(FATAL_ERROR
    "NEXUS_AM_HOME='${NEXUS_AM_HOME}' does not exist.\n"
    "Set it with -DNEXUS_AM_HOME=/path/to/nexus-am or export AM_HOME.")
endif()

if(NOT EXISTS "${NEXUS_AM_HOME}/am" OR NOT EXISTS "${NEXUS_AM_HOME}/libs/klib")
  message(FATAL_ERROR
    "NEXUS_AM_HOME='${NEXUS_AM_HOME}' does not look like nexus-am "
    "(missing am/ or libs/klib/).")
endif()

if(NOT EXISTS "${GEM5_BINARY}")
  message(FATAL_ERROR
    "GEM5_BINARY='${GEM5_BINARY}' does not exist.\n"
    "Set -DGEM5_HOME=/path/to/GEM5 or -DGEM5_BINARY=/path/to/gem5.opt.")
endif()

if(NOT EXISTS "${GEM5_CONFIG}")
  message(FATAL_ERROR
    "GEM5_CONFIG='${GEM5_CONFIG}' does not exist.\n"
    "Set -DGEM5_CONFIG=/path/to/kmhv3.py.")
endif()

if(BUILD_QEMU_TARGETS AND (NOT QEMU_BINARY OR NOT EXISTS "${QEMU_BINARY}"))
  message(FATAL_ERROR
    "BUILD_QEMU_TARGETS=ON but QEMU_BINARY='${QEMU_BINARY}' does not exist.\n"
    "Install qemu-user or configure with -DBUILD_QEMU_TARGETS=OFF for gem5-only work.")
endif()

# ---- shared build settings ---------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Linker script shared by the bare-metal helper module.
set(RISCV_LINKER_SCRIPT    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/riscv_init_array.ld")

set(RISCV_BAREMETAL_COMPILE_FLAGS
  -ffreestanding -fno-asynchronous-unwind-tables -fno-builtin -fno-stack-protector
  -Wno-main)

set(RISCV_AM_DEFINE_FLAGS
  "-D__ISA__=\"riscv64\""
  "-D__ISA_RISCV64__"
  "-D__PLATFORM__=\"xs\""
  "-D__PLATFORM_XS__"
  "-D__ARCH__=riscv64-xs"
  "-D__ARCH_RISCV64_XS"
  "-DARCH_H=\"arch/riscv64-xs.h\""
  "-DISA_H=\"riscv.h\""
  "-DRISCV_BAREMETAL")

include(cmake/riscv_add_baremetal.cmake)

# ---- targets -----------------------------------------------------------------

# The library itself.
add_subdirectory(lib)

enable_testing()
add_subdirectory(test)
add_subdirectory(benchmark)

message(STATUS "")
message(STATUS "RISC-V-CV configuration:")
message(STATUS "  NEXUS_AM_HOME   = ${NEXUS_AM_HOME}")
message(STATUS "  GEM5_BINARY     = ${GEM5_BINARY}")
message(STATUS "  GEM5_CONFIG     = ${GEM5_CONFIG}")
message(STATUS "  BUILD_QEMU_TARGETS = ${BUILD_QEMU_TARGETS}")
message(STATUS "  QEMU_BINARY     = ${QEMU_BINARY}")
message(STATUS "  QEMU_CPU        = ${QEMU_CPU}")
message(STATUS "  AM_ARCH         = ${AM_ARCH}")
message(STATUS "  RISCV_MARCH     = ${RISCV_MARCH}")
message(STATUS "")
