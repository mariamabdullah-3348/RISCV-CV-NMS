cmake_minimum_required(VERSION 3.22)

# Unit tests run on gem5 using the gtest-lite shim (real GoogleTest cannot
# link against klib's bare-metal runtime). Test sources keep standard gtest
# syntax; only the include line changes to gtest_lite.hpp.
#
# One aggregate image (unit_tests) bundles every test file for full-suite
# gem5/qemu runs and ctest. Each source/<Name>_test.cpp also builds its own
# image pair for focused runs — gtest-lite has no --gtest_filter, so isolating
# a single test file requires its own binary. Per-test gem5 run targets are
# registered without ctest entries; the aggregate unit_tests_gem5 ctest covers
# gem5 so a full ctest run does not simulate every test twice.

set(TEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR}/source/unit_tests.cpp)
file(GLOB TEST_CASE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/source/*_test.cpp)

function(unit_test_includes target)
  target_include_directories(${target} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../framework/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
endfunction()

# ---- aggregate: all test cases in one binary --------------------------------

riscv_add_baremetal_image(${TEST_MAIN})
target_sources(unit_tests_gem5 PRIVATE ${TEST_CASE_SOURCES})
target_link_libraries(unit_tests_gem5 PRIVATE ${CMAKE_PROJECT_NAME}_gem5 reference_gem5 riscv_baremetal_runtime)
unit_test_includes(unit_tests_gem5)
riscv_add_gem5_run(unit_tests_gem5)

if(BUILD_QEMU_TARGETS)
  riscv_add_linux_executable(${TEST_MAIN})
  target_sources(unit_tests_qemu PRIVATE ${TEST_CASE_SOURCES})
  target_link_libraries(unit_tests_qemu PRIVATE ${CMAKE_PROJECT_NAME}_qemu reference_qemu)
  unit_test_includes(unit_tests_qemu)
  riscv_add_qemu_run(unit_tests_qemu)
endif()

# ---- per-test-file images for focused runs ----------------------------------
foreach(TEST_CASE_SOURCE ${TEST_CASE_SOURCES})
  get_filename_component(TEST_NAME ${TEST_CASE_SOURCE} NAME_WE)

  # GEM5 image
  riscv_add_baremetal_image(${TEST_CASE_SOURCE})
  target_sources(${TEST_NAME}_gem5 PRIVATE ${TEST_MAIN})
  target_link_libraries(${TEST_NAME}_gem5 PRIVATE ${CMAKE_PROJECT_NAME}_gem5 reference_gem5 riscv_baremetal_runtime)
  unit_test_includes(${TEST_NAME}_gem5)
  set_target_properties(${TEST_NAME}_gem5 PROPERTIES RISCV_GEM5_NO_CTEST TRUE)
  riscv_add_gem5_run(${TEST_NAME}_gem5)

  # Optional QEMU image; gem5 is the required evaluation target.
  if(BUILD_QEMU_TARGETS)
    riscv_add_linux_executable(${TEST_CASE_SOURCE})
    target_sources(${TEST_NAME}_qemu PRIVATE ${TEST_MAIN})
    target_link_libraries(${TEST_NAME}_qemu PRIVATE ${CMAKE_PROJECT_NAME}_qemu reference_qemu)
    unit_test_includes(${TEST_NAME}_qemu)
    riscv_add_qemu_run(${TEST_NAME}_qemu)
  endif()
endforeach()
