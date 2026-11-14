cmake_minimum_required(VERSION 3.20)

project(mdk-crm VERSION 0.1.0 DESCRIPTION "mdk Canon Cinema RAW Light plugin" LANGUAGES C CXX)
if(APPLE)
  enable_language(OBJCXX)
endif()

# The MDK SDK with the C++ ABI headers in include/abi: scripts/fetch-mdk-sdk.sh.
# It must be the same MDK build the plugin is loaded into.
set(MDKSDK "${CMAKE_CURRENT_SOURCE_DIR}/mdk-sdk" CACHE PATH "libmdk SDK dir")
if(NOT EXISTS "${MDKSDK}/include/abi/mdk/FrameReader.h")
  message(FATAL_ERROR "No MDK ABI headers in ${MDKSDK}/include/abi: run scripts/fetch-mdk-sdk.sh or set MDKSDK")
endif()
list(APPEND CMAKE_MODULE_PATH ${MDKSDK}/lib/cmake)
find_package(MDK)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# The Rust core links the release C runtime.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

find_package(Threads REQUIRED)

# libmdk on Linux is built against libc++, and the plugin interface passes
# std::string, std::function and friends, so the plugin must use it too.
# The libc++.so.1 shipped with libmdk is from 2020, and current libc++ headers
# call functions it lacks (std::__hash_memory, bad_function_call's key
# function), so libc++ is linked in statically and kept private: the plugin
# imports nothing from that runtime, and the objects it shares with libmdk have
# libc++'s stable (v1) layout.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "On Linux build with clang and libc++ (libmdk's C++ library): -DCMAKE_CXX_COMPILER=clang++")
  endif()
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-stdlib=libc++>)
  add_link_options(-stdlib=libc++ -static-libstdc++ -Wl,--exclude-libs,ALL)
endif()

setup_mdk_plugin(${PROJECT_NAME})

target_sources(${PROJECT_NAME} PRIVATE
  src/CRMReader.cpp
  src/Gpu.cpp
)
if(APPLE)
  target_sources(${PROJECT_NAME} PRIVATE src/GpuMetal.mm)
  set_source_files_properties(src/GpuMetal.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
else()
  target_sources(${PROJECT_NAME} PRIVATE src/GpuCuda.cpp src/GpuOpenCL.cpp)
endif()
target_include_directories(${PROJECT_NAME} PRIVATE src)
# The plugin ships as built: only setup_mdk_plugin's $ORIGIN / @loader_path
# entries, not the build machine's libmdk directory that CMake would add.
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME ${PROJECT_NAME} BUILD_WITH_INSTALL_RPATH ON)
if(WIN32)
  target_compile_definitions(${PROJECT_NAME} PRIVATE _CRT_SECURE_NO_WARNINGS) # getenv
endif()

# ─── The Rust core ───────────────────────────────────────────────────
find_program(CARGO cargo HINTS "$ENV{CARGO_HOME}/bin" "$ENV{HOME}/.cargo/bin" REQUIRED)
set(CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/core")
set(CORE_TARGET_DIR "${CMAKE_BINARY_DIR}/cargo" CACHE PATH "cargo target dir of the Rust core")
file(GLOB CORE_SOURCES CONFIGURE_DEPENDS "${CORE_DIR}/src/*.rs")
list(APPEND CORE_SOURCES "${CORE_DIR}/Cargo.toml")
if(EXISTS "${CORE_DIR}/Cargo.lock")
  list(APPEND CORE_SOURCES "${CORE_DIR}/Cargo.lock")
endif()

if(WIN32)
  set(CORE_LIB_NAME mdk_crm_core.lib)
else()
  set(CORE_LIB_NAME libmdk_crm_core.a)
endif()

# Arguments to cargo build, e.g. --offline. --locked: the dependencies are the
# ones Cargo.lock names, or the build fails (drop it only for a local [patch]).
set(CORE_CARGO_ARGS "--locked" CACHE STRING "arguments to cargo build")
set(CARGO_BUILD ${CMAKE_COMMAND} -E env "CARGO_TARGET_DIR=${CORE_TARGET_DIR}"
  ${CARGO} build --release --manifest-path "${CORE_DIR}/Cargo.toml" ${CORE_CARGO_ARGS})

if(APPLE)
  # One static library per architecture, merged.
  set(CORE_ARCHS ${CMAKE_OSX_ARCHITECTURES})
  if(NOT CORE_ARCHS)
    set(CORE_ARCHS ${CMAKE_SYSTEM_PROCESSOR})
  endif()
  set(CORE_PARTS)
  set(CORE_COMMANDS)
  foreach(arch IN LISTS CORE_ARCHS)
    if(arch MATCHES "arm64|aarch64")
      set(triple aarch64-apple-darwin)
    else()
      set(triple x86_64-apple-darwin)
    endif()
    list(APPEND CORE_PARTS "${CORE_TARGET_DIR}/${triple}/release/${CORE_LIB_NAME}")
    list(APPEND CORE_COMMANDS COMMAND ${CMAKE_COMMAND} -E env
      "MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}" ${CARGO_BUILD} --target ${triple})
  endforeach()
  set(CORE_LIB "${CORE_TARGET_DIR}/universal/${CORE_LIB_NAME}")
  add_custom_command(OUTPUT ${CORE_LIB}
    ${CORE_COMMANDS}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CORE_TARGET_DIR}/universal"
    COMMAND lipo -create ${CORE_PARTS} -output ${CORE_LIB}
    DEPENDS ${CORE_SOURCES}
    WORKING_DIRECTORY "${CORE_DIR}" # cargo reads .cargo/config.toml from here up
    COMMENT "Building the Rust core (${CORE_ARCHS})"
    USES_TERMINAL VERBATIM)
else()
  set(CORE_LIB "${CORE_TARGET_DIR}/release/${CORE_LIB_NAME}")
  add_custom_command(OUTPUT ${CORE_LIB}
    COMMAND ${CARGO_BUILD}
    DEPENDS ${CORE_SOURCES}
    WORKING_DIRECTORY "${CORE_DIR}" # cargo reads .cargo/config.toml from here up
    COMMENT "Building the Rust core"
    USES_TERMINAL VERBATIM)
endif()
add_custom_target(mdk_crm_core DEPENDS ${CORE_LIB})
add_dependencies(${PROJECT_NAME} mdk_crm_core)

target_link_libraries(${PROJECT_NAME} PRIVATE ${CORE_LIB} Threads::Threads)
# What the Rust standard library and the core's crates link (`cargo rustc -- --print native-static-libs`).
if(WIN32)
  target_link_libraries(${PROJECT_NAME} PRIVATE ntdll userenv ws2_32 dbghelp)
elseif(APPLE)
  target_link_libraries(${PROJECT_NAME} PRIVATE "-framework Metal" "-framework Foundation" "-framework CoreFoundation" iconv)
else()
  target_link_libraries(${PROJECT_NAME} PRIVATE dl m)
endif()

if(TARGET cppcompat) # requires https://github.com/wang-bin/cppcompat
  target_link_libraries(${PROJECT_NAME} PRIVATE cppcompat)
endif()

# Test programs, run from a directory holding libmdk and the plugin.
option(MDK_CRM_TOOLS "Build the test programs in tools/" OFF)
if(MDK_CRM_TOOLS)
  add_executable(crm_read tools/crm_read.cpp)
  target_link_libraries(crm_read PRIVATE mdk::abi Threads::Threads)
  # The public Player API, rendering through D3D11 or Metal as an application does.
  if(WIN32)
    add_executable(crm_render tools/crm_render.cpp)
    target_link_libraries(crm_render PRIVATE mdk d3d11 dxgi)
  elseif(APPLE)
    add_executable(crm_render tools/crm_render.cpp)
    set_source_files_properties(tools/crm_render.cpp PROPERTIES LANGUAGE OBJCXX COMPILE_OPTIONS "-fobjc-arc")
    target_link_libraries(crm_render PRIVATE mdk "-framework Metal" "-framework Foundation")
  endif()
  add_executable(crm_process tools/crm_process.cpp)
  target_link_libraries(crm_process PRIVATE mdk Threads::Threads)
endif()
