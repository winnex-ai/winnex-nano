cmake_minimum_required(VERSION 3.20)
project(winnex_nano VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# Reuse the winnex-madhava engine WITHOUT duplicating code.
# winnex-nano compiles the Madhava sources directly (shared kernels: QK^T
# matmul, top-k, selective-top-K attention, quantization, AVX2/OpenMP/OpenCL).
# The Madhava sources are referenced (not copied) — same files, one build.
# ============================================================================
# OpenMP: obrigatório no Linux, opcional em macOS/Windows. O Apple Clang não
# traz libomp por padrão; exigir quebra o build nos runners macOS.
find_package(OpenMP)
if(NOT OpenMP_CXX_FOUND)
    message(STATUS "winnex-nano: OpenMP não encontrado — build single-thread")
endif()

# The spectral tokenizer's decode is the ANALYTIC inverse of the group
# isomorphism ℤ/256ℤ → μ₂₅₆ (O(1) per byte). No FFT, no correlation scan,
# no external dependency — the decode is pure arithmetic (atan2 + mod).

if(NOT DEFINED MADHAVA_ROOT)
    set(MADHAVA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../winnex-madhava")
endif()

# If the winnex-madhava sources are not present locally (e.g. building from an
# sdist), fetch them from the GitHub repo (the dependency is winnex-madhava on
# PyPI; its C++ sources live in winnex-ai/winnex-madhava).
if(NOT EXISTS "${MADHAVA_ROOT}/CMakeLists.txt")
    message(STATUS "winnex-nano: winnex-madhava sources not found locally — fetching from GitHub")
    include(FetchContent)
    # v1.9.3 e' de agosto e NAO tem as correcoes que destravam macOS e
    # Windows: flags de SIMD por compilador, OpenMP opcional, e carregamento
    # dinamico do OpenCL portavel. Fixar numa tag antiga fazia o CI baixar
    # codigo que nao compila nessas duas plataformas.
    FetchContent_Declare(
        winnex_madhava_src
        GIT_REPOSITORY https://github.com/winnex-ai/winnex-madhava.git
        GIT_TAG v1.9.17
    )
    FetchContent_Populate(winnex_madhava_src)
    set(MADHAVA_ROOT "${winnex_madhava_src_SOURCE_DIR}")
endif()
message(STATUS "winnex-nano: reusing winnex-madhava at ${MADHAVA_ROOT}")

# speed_opencl.cpp SO entra quando MADHAVA_HAS_OPENCL esta definido (Linux).
# Sem essa macro, speed_cpu.cpp tambem fornece as implementacoes GPU (pelo
# #if !defined(MADHAVA_HAS_OPENCL)), e incluir os dois da simbolos duplicados:
#   LNK2005: bound_stage1_gpu already defined in speed_opencl.obj
# Foi exatamente o erro de link no runner Windows.
set(MADHAVA_SRC
    ${MADHAVA_ROOT}/src/winnex_madhava.cpp
    ${MADHAVA_ROOT}/src/speed_cpu.cpp
)
add_library(madhava_reused STATIC ${MADHAVA_SRC})
target_include_directories(madhava_reused PUBLIC ${MADHAVA_ROOT}/include)
target_include_directories(madhava_reused PRIVATE ${MADHAVA_ROOT}/vendor)
# OpenCL só no Linux: o macOS removeu os headers no Xcode 12 e o Windows não
# os traz por padrão. Sem esta guarda, o build falha em duas das três
# plataformas que queremos publicar.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(madhava_reused PRIVATE MADHAVA_HAS_OPENCL)
    # o fonte correspondente entra junto — macro e fonte andam em par
    target_sources(madhava_reused PRIVATE ${MADHAVA_ROOT}/src/speed_opencl.cpp)
endif()
#
# ESTAS FLAGS TÊM DE CHEGAR AO ALVO `winnex_nano`, NÃO SÓ AO `madhava_reused`.
#
# Medido em 2026-10-24: estavam aplicadas apenas a `madhava_reused`. A biblioteca
# `winnex_nano` — que contém `dense_matmul`, o laço da CPU — era compilada com
# "-O3 -DNDEBUG" e nada mais (confirmado no build.ninja:
# LANGUAGE_COMPILE_FLAGS = -O3 -DNDEBUG). O bloco AVX2 de tensor_ops.cpp está sob
# `#if defined(__AVX2__)`: nunca compilava. Objdump do dense_matmul no binário
# publicado mostrava zero instruções vfmadd.
#
# Custo medido do defeito: 3,5x no laço isolado (64,8 -> 228,2 GFLOP/s) e a mesma
# razão no forward real, conforme forward.cpp:799 (30,5 s -> 9,1 s). No caminho
# CPU — o único que roda modelos maiores que a placa — é a maior perda do código
# para a menor correção possível.
#
# A macro WINNEX_COMPILED_WITH_SIMD é definida junto para que o runtime possa
# REGISTRAR se o build recebeu SIMD, em vez de o usuário descobrir por benchmark.
# Lista, não string: uma variável com "-mavx2 -mfma -funroll-loops" vira UM
# argumento e o compilador recusa com "unrecognized command-line option".
set(WINNEX_SIMD_FLAGS "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    if(MSVC)
        set(WINNEX_SIMD_FLAGS /arch:AVX2)
    else()
        set(WINNEX_SIMD_FLAGS -mavx2 -mfma -funroll-loops)
    endif()
    target_compile_options(madhava_reused PRIVATE ${WINNEX_SIMD_FLAGS})
    message(STATUS "winnex-nano: SIMD do alvo winnex_nano = ${WINNEX_SIMD_FLAGS}")
else()
    message(STATUS "winnex-nano: sem AVX2 nesta arquitetura (${CMAKE_SYSTEM_PROCESSOR})")
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(madhava_reused PUBLIC OpenMP::OpenMP_CXX)
endif()

# ============================================================================
# winnex-nano core (Winnex spectral layer + weight balancer + tensor engine)
# ============================================================================
add_library(winnex_nano
    src/spectral_tokenizer.cpp
    src/weight_balancer.cpp
    src/tensor_ops.cpp
    src/chunk_engine.cpp
    src/x_factor.cpp
    # A ponte entre a codificação sem vocabulário e o espaço de embeddings.
    # Ver o cabeçalho para o motivo medido (top-1 errado em 4/4 prompts).
    src/embed_bridge.cpp
    src/safetensors.cpp
    src/forward.cpp
    # OpenCL dense projections for the forward pass. The loader is resolved at
    # RUNTIME (dlopen/LoadLibrary), so there is no link-time OpenCL dependency
    # and the wheel stays portable — see the note at the top of forward_gpu.cpp.
    src/forward_gpu.cpp
)
target_include_directories(winnex_nano PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# CL/cl.h for forward_gpu.cpp. winnex-madhava vendors the headers and is
# fetched by the CMakeLists above, so the path is always available.
target_include_directories(winnex_nano PRIVATE ${MADHAVA_ROOT}/vendor)
target_link_libraries(winnex_nano PUBLIC madhava_reused)
target_link_libraries(winnex_nano PRIVATE ${CMAKE_DL_LIBS})

# O LAÇO DA CPU VIVE AQUI. Sem estas flags, dense_matmul compila o ramo escalar
# de tensor_ops.cpp e o motor perde 3,5x no único caminho que roda modelos
# maiores que a placa. Ver o bloco de comentário acima.
if(NOT "${WINNEX_SIMD_FLAGS}" STREQUAL "")
    target_compile_options(winnex_nano PRIVATE ${WINNEX_SIMD_FLAGS})
    target_compile_definitions(winnex_nano PUBLIC WINNEX_COMPILED_WITH_SIMD=1)
else()
    target_compile_definitions(winnex_nano PUBLIC WINNEX_COMPILED_WITH_SIMD=0)
endif()

# --- Python bindings (pybind11) --------------------------------------------
# scikit-build-core sets up Python3_ROOT_DIR / Python_EXECUTABLE in the build
# environment; pybind11's pybind11_add_module resolves the interpreter without
# us requiring Development/Development.Embed (which manylinux lacks).
option(WINNEX_NANO_BUILD_PYTHON "Build the pybind11 Python module" ON)
if(WINNEX_NANO_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_winnex_nano bindings/py_module.cpp)
    target_link_libraries(_winnex_nano PRIVATE winnex_nano)
    # Install the extension INSIDE the package dir (python/winnex_nano → the
    # wheel's winnex_nano/), so `from ._winnex_nano import ...` resolves.
    # SKBUILD_PLATLIB_DIR is the wheel root; the package lives in its
    # `winnex_nano` subdirectory.
    if(DEFINED SKBUILD_PLATLIB_DIR)
        install(TARGETS _winnex_nano DESTINATION ${SKBUILD_PLATLIB_DIR}/winnex_nano)
    endif()
endif()

# --- Tests -----------------------------------------------------------------
option(WINNEX_NANO_BUILD_TESTS "Build the C++ tests" ON)
if(WINNEX_NANO_BUILD_TESTS)
    enable_testing()
    add_executable(test_spectral_tokenizer tests/test_spectral_tokenizer.cpp)
    target_link_libraries(test_spectral_tokenizer PRIVATE winnex_nano)
    add_test(NAME spectral_tokenizer COMMAND test_spectral_tokenizer)

    # HRR: agregação com preservação de norma + NÃO-REGRESSÃO do modo atual
    add_executable(test_hrr tests/test_hrr.cpp)
    target_link_libraries(test_hrr PRIVATE winnex_nano)
    add_test(NAME hrr COMMAND test_hrr)

    add_executable(test_chunk_engine tests/test_chunk_engine.cpp)
    target_link_libraries(test_chunk_engine PRIVATE winnex_nano)
    add_test(NAME chunk_engine COMMAND test_chunk_engine)

    add_executable(test_x_factor tests/test_x_factor.cpp)
    target_link_libraries(test_x_factor PRIVATE winnex_nano)
    add_test(NAME x_factor COMMAND test_x_factor)

    # test_safetensors exige o caminho de um .safetensors. Ele estava
    # registrado sem argumento, então o CTest o invocava sem modelo, o
    # programa imprimia "usage: ..." e saía com código diferente de zero: um
    # teste permanentemente vermelho que não testava nada. Entra na mesma
    # condição do forward — só quando há checkpoint configurado.
    add_executable(test_safetensors tests/test_safetensors.cpp)
    target_link_libraries(test_safetensors PRIVATE winnex_nano)
    if(DEFINED WINNEX_TEST_MODEL_DIR AND EXISTS "${WINNEX_TEST_MODEL_DIR}/model.safetensors")
        add_test(NAME safetensors
                 COMMAND test_safetensors "${WINNEX_TEST_MODEL_DIR}/model.safetensors")
    else()
        message(STATUS "winnex-nano: test_safetensors nao registrado "
                       "(defina WINNEX_TEST_MODEL_DIR para ativar)")
    endif()

    # test_forward precisa de um checkpoint real. O caminho vem de
    # WINNEX_TEST_MODEL_DIR; sem ele o teste NÃO é registrado, em vez de ser
    # registrado e passar sem verificar nada — que era o estado anterior
    # (imprimia logits, não tinha asserção, e testava um caminho que a geração
    # não usa).
    #
    #   WINNEX_TEST_MODEL_DIR=/caminho/Qwen2.5-0.5B-Instruct \
    #     cmake -DWINNEX_TEST_MODEL_DIR=$WINNEX_TEST_MODEL_DIR ...
    #   ./test_forward $WINNEX_TEST_MODEL_DIR --record   # uma vez, commita o .bin
    #
    # A referência (forward_reference.bin) é gerada pela CPU em f32, então o
    # CI compara álgebra sem precisar de PyTorch.
    add_executable(test_forward tests/test_forward.cpp)
    target_link_libraries(test_forward PRIVATE winnex_nano)
    if(DEFINED WINNEX_TEST_MODEL_DIR AND EXISTS "${WINNEX_TEST_MODEL_DIR}/config.json")
        add_test(NAME forward_golden
                 COMMAND test_forward "${WINNEX_TEST_MODEL_DIR}")
        set_tests_properties(forward_golden PROPERTIES
            # rc=3 = referencia ausente. Falha, nao passa: um gate que nao
            # pode verificar tem que ser visivel.
            FAIL_REGULAR_EXPRESSION "FALHOU|SEM REFERENCIA|ERROR")
    else()
        message(STATUS "winnex-nano: test_forward nao registrado "
                       "(defina WINNEX_TEST_MODEL_DIR para ativar)")
    endif()
endif()
