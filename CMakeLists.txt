cmake_minimum_required(VERSION 3.16)
project(MTGCardMatcher)

# Policies for modern find_package behavior
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(OpenCV REQUIRED)

# FAISS: prefer config package, fall back to manual search via FAISS_ROOT/FAISS_INCLUDE_DIR
set(FAISS_PACKAGE_FOUND OFF)
find_package(faiss CONFIG QUIET)
if(TARGET faiss)
  set(FAISS_PACKAGE_FOUND ON)
else()
  message(STATUS "faiss CMake package not found. Falling back to manual lookup using FAISS_ROOT/FAISS_INCLUDE_DIR.")
  # Allow user/environment to provide FAISS_ROOT
  if(DEFINED ENV{FAISS_ROOT} AND NOT FAISS_ROOT)
    set(FAISS_ROOT "$ENV{FAISS_ROOT}")
  endif()
  # Try common lib locations under FAISS_ROOT
  find_library(FAISS_LIB
    NAMES faiss libfaiss
    HINTS ${FAISS_ROOT} ${FAISS_ROOT}/lib ${FAISS_ROOT}/lib64
  )
  # Try includes under FAISS_ROOT or an explicit FAISS_INCLUDE_DIR
  if(NOT FAISS_INCLUDE_DIR)
    find_path(FAISS_INCLUDE_DIR
      NAMES faiss/IndexBinaryIVF.h
      HINTS ${FAISS_ROOT}/include
    )
  endif()
  if(NOT FAISS_LIB OR NOT FAISS_INCLUDE_DIR)
    message(FATAL_ERROR "FAISS not found. Provide faiss via package config (faiss_DIR) or set FAISS_ROOT and optionally FAISS_INCLUDE_DIR.")
  endif()
endif()

# nlohmann_json: try package first, fall back to header-only
find_package(nlohmann_json 3.2.0 QUIET)
if(NOT nlohmann_json_FOUND)
  message(STATUS "nlohmann_json package not found. Looking for header-only installation.")
  # Check if NLOHMANN_JSON_INCLUDE_DIR was provided via command line
  if(NOT NLOHMANN_JSON_INCLUDE_DIR)
    find_path(NLOHMANN_JSON_INCLUDE_DIR
      NAMES nlohmann/json.hpp
      HINTS /usr/include /usr/local/include ${CMAKE_PREFIX_PATH}/include
      PATH_SUFFIXES nlohmann
    )
  endif()
  if(NLOHMANN_JSON_INCLUDE_DIR)
    message(STATUS "Found nlohmann/json.hpp at: ${NLOHMANN_JSON_INCLUDE_DIR}")
    add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
    set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${NLOHMANN_JSON_INCLUDE_DIR}"
    )
  else()
    message(FATAL_ERROR "nlohmann_json not found. Install nlohmann-json3-dev or provide via CMAKE_PREFIX_PATH.")
  endif()
endif()

# SQLite3: try package, on Android fall back to NDK's sqlite3
set(SQLITE_TARGET SQLite::SQLite3)
find_package(SQLite3 QUIET)
if(NOT SQLite3_FOUND)
  if(ANDROID)
    find_library(ANDROID_SQLITE3 sqlite3 REQUIRED)
    add_library(SQLite3::SQLite3 UNKNOWN IMPORTED)
    set_target_properties(SQLite3::SQLite3 PROPERTIES IMPORTED_LOCATION ${ANDROID_SQLITE3})
  else()
    message(FATAL_ERROR "SQLite3 not found. Install libsqlite3-dev or provide via CMAKE_PREFIX_PATH.")
  endif()
endif()

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})
include_directories(${NLOHMANN_JSON_INCLUDE_DIR})
include_directories(${FAISS_INCLUDE_DIR})

# Source files
set(SOURCES
    matcher.cpp
)

# Library target (for Android NDK)
add_library(matcher STATIC ${SOURCES})

# Executable target
add_executable(matcher_test main.cpp)

# Statically link all libraries for matcher_test
if(FAISS_PACKAGE_FOUND)
  target_link_libraries(matcher_test
      matcher
      ${OpenCV_LIBS}
      faiss
      nlohmann_json::nlohmann_json
      ${SQLITE_TARGET}
      -static
  )
else()
  target_link_libraries(matcher_test
      matcher
      ${OpenCV_LIBS}
      ${FAISS_LIB}
      nlohmann_json::nlohmann_json
      ${SQLITE_TARGET}
      -static
  )
endif()

# Android NDK specific settings
if(ANDROID)
    set_target_properties(matcher PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/android/libs/${ANDROID_ABI}"
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()
