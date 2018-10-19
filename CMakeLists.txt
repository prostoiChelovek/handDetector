cmake_minimum_required(VERSION 3.8)
project(handDetector)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

set(SOURCE_FILES
        example.cpp
        Finger.cpp Finger.h utils.cpp utils.h Hand.cpp Hand.h HandDetector.cpp HandDetector.h)

add_executable(handDetector ${SOURCE_FILES})
target_link_libraries(handDetector ${OpenCV_LIBS})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")