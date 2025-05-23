cmake_minimum_required(VERSION 3.19)


###############################################################################
# Application related definitions
###############################################################################

if (CONFIG_MODEL_EMUL)
  set(TARGET_NAME         superpoint_emul)
else()
  set(TARGET_NAME         superpoint)
endif()
set(MAIN                superpoint.c)

###############################################################################
# CMake pre initialization
###############################################################################

include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)
set(MODEL_EXPRESSIONS   ${CMAKE_BINARY_DIR}/BUILD_MODEL/Expression_Kernels.c)

###############################################################################
# Autotiler Variables
###############################################################################

autotiler_define_gen_names(${TARGET_NAME})
autotiler_setup(${TARGET_NAME})

###############################################################################
# Application dependencies
###############################################################################

list(APPEND TARGET_SRCS ${MAIN}
                        ${CNN_LIB}
                        ${CONFIG_MODEL_BUILDDIR}/${KERNEL_C}
                        ${MODEL_EXPRESSIONS}
                        superpoint/postprocessing.c
                        utility/rigid_body_motion/rigid_body_motion.c 
                        utility/extended_kalman_filter/of_imu_ekf.c
)

list(APPEND TARGET_INCS -I.
                        -I${TILER_INC}
                        -I${TILER_EMU_INC}
                        -I$ENV{GAP_SDK_HOME}/libs/gap_lib/include
                          ${CNN_LIB_INCLUDE}
                        -I${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_MODEL_BUILDDIR}
                        -I${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${CMAKE_SOURCE_DIR}/../sensor_data/sample_data)
include_directories(${CMAKE_SOURCE_DIR}/utility)

if (CONFIG_MODEL_EMUL)
list(APPEND TARGET_CFLAGS   -g
                            -m32
                            -O0
                            -fcommon
                            -D__EMUL__)
else()
list(APPEND TARGET_CFLAGS   -gdwarf-2
                            -gstrict-dwarf
                            -O3
                            -mno-memcpy
                            -fcommon
                            -fno-tree-loop-distribute-patterns)
endif()

math(EXPR FREQ_CL ${CONFIG_FREQUENCY_CLUSTER}/1000000 OUTPUT_FORMAT DECIMAL)
math(EXPR FREQ_FC ${CONFIG_FREQUENCY_FC}/1000000 OUTPUT_FORMAT DECIMAL)
math(EXPR FREQ_PE ${CONFIG_FREQUENCY_PERIPH}/1000000 OUTPUT_FORMAT DECIMAL)

set(TEST_IMAGE ${CMAKE_CURRENT_SOURCE_DIR}/calibration_data/frame000114.pgm)
list(APPEND TARGET_PREPROCESSOR -DPERF
                                -DAT_MODEL_PREFIX=${MODEL_PREFIX}
                                -DSTACK_SIZE=${CONFIG_CL_MASTER_CORE_STACK_SIZE}
                                -DSLAVE_STACK_SIZE=${CONFIG_CL_SLAVE_CORE_STACK_SIZE}
                                -DFREQ_CL=${FREQ_CL}
                                -DFREQ_FC=${FREQ_FC}
                                -DFREQ_PE=${FREQ_PE}
                                -DAT_IMAGE=${TEST_IMAGE}
                                -DCI)

project(${TARGET_NAME} C ASM)

add_executable(${TARGET_NAME} ${TARGET_SRCS})

if (CONFIG_MODEL_EMUL)
find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    set(CMAKE_C_COMPILER_LAUNCHER   ccache)
    set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
endif()
target_link_libraries(${TARGET_NAME} PUBLIC m)
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "-m32" LINK_FLAGS "-m32")
if (NOT CONFIG_MODEL_HAS_NO_CONSTANTS)
  add_custom_command(TARGET ${TARGET_NAME}
    COMMENT "Copy tensors to build directory"
    POST_BUILD
    COMMAND cp ${MODEL_TENSORS} ${CMAKE_CURRENT_BINARY_DIR}
  )
endif()
add_custom_target(run ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME})
endif()

target_compile_options(${TARGET_NAME} PUBLIC   ${TARGET_CFLAGS}
                                                ${TARGET_INCS}
                                                ${TARGET_PREPROCESSOR})

###############################################################################
# Commands added for specific application
###############################################################################
add_custom_command(
  OUTPUT ${AT_MODEL_PATH} ${MODEL_EXPRESSIONS}
  COMMENT "Generate the autotiler model"
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/nntool_generate_model.py
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/nntool_generate_model.py 
                                                               --trained_model="${CMAKE_CURRENT_SOURCE_DIR}/superpoint.onnx"
                                                               --calibration="${CMAKE_SOURCE_DIR}/calibration_data/"
                                                               --at_model_path=${AT_MODEL_PATH}
                                                               --tensors_dir=${AT_TENSORS_DIR}
                                                               --flash_type=${MODEL_L3_FLASH}
                                                               --ram_type=${MODEL_L3_RAM}
)
add_custom_target(${NNTOOL_TARGET_PREFIX}clean_model
  COMMENT "clean_model: removing ${COMMON_MODEL_BUILDDIR}/${MODEL_GEN}"
  COMMAND rm -rf ${AT_MODEL_PATH}
)

###############################################################################
# CMake post initialization
###############################################################################
if (NOT CONFIG_MODEL_EMUL)
  setupos(${TARGET_NAME})
endif ()
