cmake_minimum_required(VERSION 3.3)
project(noteshrink-c LANGUAGES CXX)

if (APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
endif()
find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})
# linking
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(noteshrink-c src/main.cpp)

include_directories(include test/ext)

# linking
#target_link_libraries(/opt/homebrew/Cellar/opencv/4.5.5_2/lib)
#include_directories(/opt/homebrew/Cellar/opencv/4.5.5_2/include)
#
file(GLOB source_files include/*.h
                       src/*.cpp
                       test/*.cpp
                       CMakeLists.txt)
add_executable(noteshrink-c-test ${source_files})
target_link_libraries(noteshrink-c-test ${OpenCV_LIBS})