cmake_minimum_required(VERSION 3.20)
project(needle C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(NEEDLE_ENABLE_CUDA "Build the CUDA backend when a toolkit is available" ON)

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
  set(CMAKE_CXX_FLAGS "-O2 -march=native ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS "/O2 ${CMAKE_CXX_FLAGS}")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

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
if(NEEDLE_ENABLE_CUDA)
  # 项目本地安装优先；系统 Toolkit 也可通过 CUDAToolkit_ROOT 显式指定。
  set(LOCAL_CUDA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.tools/cuda-13.0.2")
  if(NOT CUDAToolkit_ROOT AND EXISTS "${LOCAL_CUDA_ROOT}/bin/nvcc")
    set(CUDAToolkit_ROOT "${LOCAL_CUDA_ROOT}" CACHE PATH "CUDA Toolkit location")
  endif()
  if(CUDAToolkit_ROOT AND NOT CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
  endif()
  # 覆盖课程常见的 T4、Ampere 和本机 RTX 4060；允许命令行指定其他架构。
  # 不依赖配置阶段访问 GPU，沙箱内也可以编译。
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
  endif()
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    # clangd 读取编译数据库时需要直接看到头文件路径，而非 NVCC 专用响应文件。
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
    pybind11_add_module(ndarray_backend_cuda src/ndarray_backend_cuda.cu)
    target_link_libraries(ndarray_backend_cuda PRIVATE CUDA::cudart)
    get_target_property(CUDA_RUNTIME_LOCATION CUDA::cudart IMPORTED_LOCATION)
    get_filename_component(CUDA_RUNTIME_DIR "${CUDA_RUNTIME_LOCATION}" DIRECTORY)
    set_target_properties(ndarray_backend_cuda PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
      CXX_VISIBILITY_PRESET "hidden"
      CUDA_VISIBILITY_PRESET "hidden"
      BUILD_RPATH "${CUDA_RUNTIME_DIR}"
    )
    message(STATUS "CUDA backend enabled; architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  else()
    message(STATUS "CUDA compiler not found; building the CPU backend only")
  endif()
endif()
