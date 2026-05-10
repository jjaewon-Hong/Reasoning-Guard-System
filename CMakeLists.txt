cmake_minimum_required(VERSION 3.18)
project(ReasoningGuard LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── 경로 설정 (설치 후 아래 두 경로를 본인 환경에 맞게 수정) ──
# ONNX Runtime C++ SDK 경로
set(ONNXRUNTIME_DIR "C:/RG/sdk/onnxruntime" CACHE PATH "ONNX Runtime SDK 루트 경로")
# OpenCV 경로
set(OpenCV_DIR "C:/RG/sdk/opencv/build" CACHE PATH "OpenCV build 디렉토리")

# ── ONNX Runtime ────────────────────────────────────────────
set(ORT_INCLUDE "${ONNXRUNTIME_DIR}/include")
set(ORT_LIB    "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
set(ORT_DLL    "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll")

if(NOT EXISTS "${ORT_INCLUDE}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "ONNX Runtime 헤더를 찾을 수 없습니다.\n"
        "  다운로드: https://github.com/microsoft/onnxruntime/releases/tag/v1.26.0\n"
        "  (onnxruntime-win-x64-1.26.0.zip 압축 해제 → ${ONNXRUNTIME_DIR})")
endif()

# ── OpenCV ───────────────────────────────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR
        "OpenCV를 찾을 수 없습니다.\n"
        "  다운로드: https://github.com/opencv/opencv/releases\n"
        "  설치 후 OpenCV_DIR 경로를 수정하세요.")
endif()

# ── 실행 파일 ────────────────────────────────────────────────
add_executable(ReasoningGuard
    main.cpp
    tactical_engine.h
    preprocess.h
)

target_include_directories(ReasoningGuard PRIVATE
    ${ORT_INCLUDE}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ReasoningGuard PRIVATE
    ${ORT_LIB}
    ${OpenCV_LIBS}
)

# -- MSVC UTF-8 인코딩 강제 적용 ----------------------------
if(MSVC)
    target_compile_options(ReasoningGuard PRIVATE /utf-8)
endif()

# ── 빌드 후 DLL을 실행 디렉토리에 복사 ─────────────────────
add_custom_command(TARGET ReasoningGuard POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_DLL}"
        "$<TARGET_FILE_DIR:ReasoningGuard>"
    COMMENT "onnxruntime.dll 복사 중..."
)

# OpenCV DLL 복사 (릴리즈 빌드 기준)
foreach(lib ${OpenCV_LIBS})
    get_target_property(lib_location ${lib} LOCATION_Release)
    if(lib_location)
        add_custom_command(TARGET ReasoningGuard POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${lib_location}"
                "$<TARGET_FILE_DIR:ReasoningGuard>"
        )
    endif()
endforeach()
