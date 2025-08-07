cmake_minimum_required(VERSION 3.10)
project(MTGCardMatcher)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(OpenCV REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(faiss REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})

# Add external JSON library (assumes nlohmann_json is installed)
find_package(nlohmann_json 3.2.0 REQUIRED)

# Source files
set(SOURCES
    matcher.cpp
)

# Library target (for Android NDK)
add_library(matcher SHARED ${SOURCES})
target_link_libraries(matcher
    ${OpenCV_LIBS}
    ${HDF5_CXX_LIBRARIES}
    faiss
    nlohmann_json::nlohmann_json
)

# Executable target
add_executable(matcher_test main.cpp)
target_link_libraries(matcher_test
    matcher
    ${OpenCV_LIBS}
    ${HDF5_CXX_LIBRARIES}
    faiss
    nlohmann_json::nlohmann_json
)

# Android NDK specific settings
if(ANDROID)
    set_target_properties(matcher PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/android/libs/${ANDROID_ABI}"
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()