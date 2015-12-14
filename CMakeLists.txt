cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(ObjectRecognition)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

find_package(PCL 1.8 REQUIRED PATHS /usr/local NO_DEFAULT_PATH)
find_package(Boost 1.5 COMPONENTS program_options timer)
find_package(CUDA REQUIRED)

FILE(GLOB hdrs "*.h*")
FILE(GLOB srcs "*.cpp")
FILE(GLOB cuda "*.cu")

include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${CUDA_TOOLKIT_INCLUDE})
link_directories(${CUDA_CUDART_LIBRARY})

set(CUDA_NVCC_FLAGS "-Xcompiler -Wall;
			-gencode;arch=compute_30,code=sm_32;")

source_group("Source Files" FILES ${main} ${srcs})
source_group("Source Files\\cuda" FILES ${cuda})

#add_executable (main main.cpp ${srcs} ${hdrs})
cuda_add_executable(main main.cpp ${srcs} ${cuda} ${hdrs})
target_link_libraries (main ${PCL_LIBRARIES} ${Boost_LIBRARIES} ${CUDA_CUDART_LIBRARY})
