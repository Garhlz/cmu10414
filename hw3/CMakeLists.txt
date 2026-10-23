cmake_minimum_required(VERSION 3.10)
project(needle C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

# 使用运行构建命令的虚拟环境，避免链接到系统中另一版本的 Python。
set(Python_FIND_VIRTUALENV FIRST)
if(NOT Python_EXECUTABLE)
  if(DEFINED ENV{VIRTUAL_ENV} AND EXISTS "$ENV{VIRTUAL_ENV}/bin/python")
    set(Python_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python" CACHE FILEPATH "Python interpreter")
  elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
    set(Python_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python" CACHE FILEPATH "Python interpreter")
  endif()
endif()
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# find pybind
execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
  RESULT_VARIABLE __pybind_exit_code
  OUTPUT_VARIABLE __pybind_path
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT __pybind_exit_code EQUAL 0)
  message(FATAL_ERROR "pybind11 is missing; run uv sync --dev first")
endif()
find_package(pybind11 CONFIG REQUIRED PATHS ${__pybind_path})


if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
else()
  set(CMAKE_CXX_FLAGS "/std:c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
endif()

include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})
list(APPEND LINKER_LIBS ${pybind11_LIBRARIES})


###################
### CPU BACKEND ###
###################
add_library(ndarray_backend_cpu MODULE src/ndarray_backend_cpu.cc)
target_link_libraries(ndarray_backend_cpu PUBLIC ${LINKER_LIBS})
pybind11_extension(ndarray_backend_cpu)
pybind11_strip(ndarray_backend_cpu)


# directly output to ffi folder
set_target_properties(ndarray_backend_cpu
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
  CXX_VISIBILITY_PRESET "hidden"
)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set_property(TARGET ndarray_backend_cpu PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()



####################
### CUDA BACKEND ###
####################
find_package(CUDA QUIET)
if(CUDA_FOUND)
  message(STATUS "Found cuda, building cuda backend")

  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  list(APPEND LINKER_LIBS ${CUDA_CUDART_LIBRARY})

  # invoke nvidia smi to detect if we really have a GPU
  execute_process(COMMAND "nvidia-smi" OUTPUT_QUIET ERROR_QUIET RESULT_VARIABLE NV_RET)
  if(NV_RET EQUAL "0")
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
  else()
    # no GPU visible: target the T4 (sm_75) that Colab GPU runtimes use;
    # CUDA 12+ no longer supports the old K80 (sm_37) default
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 7.5)
  endif()

  # set arch flags properly
  CUDA_ADD_LIBRARY(ndarray_backend_cuda MODULE src/ndarray_backend_cuda.cu OPTIONS ${ARCH_FLAGS})

  target_link_libraries(ndarray_backend_cuda ${LINKER_LIBS})
  pybind11_extension(ndarray_backend_cuda)
  pybind11_strip(ndarray_backend_cuda)

  # directly output to ffi folder
  set_target_properties(ndarray_backend_cuda
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
)
else()
  message(STATUS "CUDA not found, building the CPU backend only")
endif()
