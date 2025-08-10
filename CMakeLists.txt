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

# DISABLE OpenMP for now to avoid linking issues
# find_package(OpenMP REQUIRED) # Commented out

# FAISS: prefer config package, fall back to manual search via FAISS_ROOT/FAISS_INCLUDE_DIR
find_package(faiss CONFIG QUIET)
if(NOT TARGET faiss)
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
  # Create imported target for consistency with package mode
  add_library(faiss UNKNOWN IMPORTED)
  set_target_properties(faiss PROPERTIES
    IMPORTED_LOCATION "${FAISS_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${FAISS_INCLUDE_DIR}"
  )
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

# Include directories (only for non-target deps if needed; most are handled via targets)
include_directories(${CMAKE_SOURCE_DIR})

# Source files
set(SOURCES
    matcher.cpp
)

# Library target (for Android NDK)
add_library(matcher STATIC ${SOURCES})
target_link_libraries(matcher PUBLIC
  ${OpenCV_LIBS}
  faiss
  nlohmann_json::nlohmann_json
  ${SQLITE_TARGET}
  # OpenMP::OpenMP_CXX # Disabled for now
)

# Executable target
add_executable(matcher_test main.cpp)
target_link_libraries(matcher_test PRIVATE matcher)

# Simplified linking for Android
if(ANDROID)
  target_link_options(matcher_test PRIVATE -static)
else()
  target_link_options(matcher_test PRIVATE -static)
endif()

# Android NDK specific settings
if(ANDROID)
    set_target_properties(matcher PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/android/libs/${ANDROID_ABI}"
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()
