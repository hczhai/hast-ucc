CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

SET(CMAKE_CXX_STANDARD 20)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

PROJECT(hast VERSION 1.0 LANGUAGES CXX)

IF (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    SET(XPREP -Xpreprocessor)
    INCLUDE_DIRECTORIES(/usr/local/include)
ENDIF()

IF("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    SET(OPT_FLAG -O0 -g ${XPREP} -fstack-check -fno-omit-frame-pointer -fno-optimize-sibling-calls
        -Werror -Werror=return-type)
ELSE()
    SET(OPT_FLAG -O3 -funroll-loops ${XPREP} -Werror -Werror=return-type)
ENDIF()

ADD_LIBRARY(hast SHARED src/ctr.cpp)
SET(OPT_FLAG ${OPT_FLAG} -march=native -mtune=native -ffast-math)
SET_TARGET_PROPERTIES(hast PROPERTIES POSITION_INDEPENDENT_CODE 1)
SET_TARGET_PROPERTIES(hast PROPERTIES OUTPUT_NAME hast-ctr SUFFIX ".so")

TARGET_COMPILE_OPTIONS(hast BEFORE PRIVATE ${OPT_FLAG})
