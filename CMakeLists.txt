cmake_minimum_required(VERSION 3.8)
project(handDetector)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

set(SOURCE_FILES
        example.cpp
        Finger.cpp Finger.h utils.cpp utils.h Hand.cpp Hand.h HandDetector.cpp HandDetector.h Filter.cpp Filter.h)

set(headers Finger.h utils.h Hand.h HandDetector.h Filter.h)

set(LIBHANDD_SRC Finger.cpp utils.cpp Hand.cpp HandDetector.cpp Filter.cpp)

add_library(handDetector_dynamic SHARED ${LIBHANDD_SRC})
add_library(handDetector_static STATIC ${LIBHANDD_SRC})

set_target_properties(handDetector_dynamic PROPERTIES OUTPUT_NAME "handDetector")
set_target_properties(handDetector_dynamic PROPERTIES VERSION 1.2 SOVERSION 1)

set_target_properties(handDetector_static PROPERTIES OUTPUT_NAME "handDetector")

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

target_link_libraries(handDetector_dynamic ${OpenCV_LIBS})
target_link_libraries(handDetector_static ${OpenCV_LIBS})

install(TARGETS handDetector_dynamic handDetector_static
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(FILES ${headers} DESTINATION include/handDetector)

add_executable(handDetector ${SOURCE_FILES})
target_link_libraries(handDetector ${OpenCV_LIBS})
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-sign-compare -Wno-narrowing")