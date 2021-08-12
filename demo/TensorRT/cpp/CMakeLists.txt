#1. 草，cmakelist.txt里面的参数编写，千万别随便带空格，它的规则就是见到空格，自动换参数。。。。所以，直接用“”包含进路径就可以。
#2. set() 定义定义隐性变量要在 find_package（）之前。
#3. 首先，电脑依次源码编译 cuda->环境变量bin；通常cudnn->cuda的bin里面，环境变量->lib/x64；tensorrt按照官方指导就可以，/
#   环境变量bin；opencv编译选定次新版本，与cuda、cudnn对应，然后按照csdn收藏的博客一步步编译就好，将编译的bin环境变量。

cmake_minimum_required(VERSION 2.6)        #cmake版本要求，必须项

project(yolox_cao)                     #显性变量， 项目名字，没太大意义。

add_definitions(-std=c++11)               #必须项

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)       #CUDA是否静态推理，可选项
set(CMAKE_CXX_STANDARD 11)                     #设定cmake版本标准
set(CMAKE_BUILD_TYPE Debug)                     #debug

find_package(CUDA REQUIRED)                    #必须项
#include_directories("${CUDA_INCLUDE_DIRS}")

include_directories(${PROJECT_SOURCE_DIR}/include)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/include")  #用“cuda/include”
link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/lib/x64")    #cuda对应链接库
# cudnn
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/bin")       #。。。
link_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.1/bin/lib/x64")   #。。。
# tensorrt
include_directories("D:/TensorRT-7.2.3.4/include")                                        #。。。
link_directories("D:/TensorRT-7.2.3.4/lib")                                                #。。。

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

#set(OpenCV_DIR "D:/opencv/build/x64/vc15/lib")
set(OpenCV_DIR "D:/opencv/build/x64/MinGW/install")     #改为mingw-bulid的位置          注意这里，OpenCV_DIR "包含路径：编译后的文件夹里面的install文件夹下的"，不是未编译的opencv
set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)          #重要，找到opencv后，将其依赖库全部导入。

find_package(OpenCV)                                                                 #set（）部分是定义变量，这里是寻找
include_directories(${OpenCV_INCLUDE_DIRS})                                           #包含inculde



add_executable(yolox ${PROJECT_SOURCE_DIR}/yolox.cpp)                          #####必改项，第一个参数 yolox通常与x.cpp保持一致，第二个参数是 x.cpp
target_link_libraries(yolox nvinfer)                                            ###链接库，连接tensorrt下的nvifer库，cuda的cudart库，以及前面的opencv依赖库。
target_link_libraries(yolox cudart)
target_link_libraries(yolox ${OpenCV_LIBS})


add_definitions(-O2 -pthread)

