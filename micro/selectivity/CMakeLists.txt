cmake_minimum_required(VERSION 3.10)

project(run_selectivity-c++)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../duckdb/src/include)

add_executable(run_selectivity run_selectivity.cpp)

find_library(DUCKDB_LIB duckdb PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../../duckdb/build_release/src)

target_link_libraries(run_selectivity ${DUCKDB_LIB})