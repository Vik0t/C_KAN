cmake_minimum_required(VERSION 4.1)
project(C_KAN C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # Улучшает индексацию и линтеры

# 🔹 Статическая библиотека (ядро ML)
file(GLOB_RECURSE LIB_SRC "src/*.c")
add_library(C_KAN STATIC ${LIB_SRC})
target_include_directories(C_KAN PUBLIC include)
target_link_libraries(C_KAN PRIVATE m)  # math.h (Linux/macOS). Windows игнорирует.

# 🔹 Примеры использования
file(GLOB EXAMPLES "examples/*.c")
foreach(EX ${EXAMPLES})
    get_filename_component(NAME ${EX} NAME_WE)
    add_executable(${NAME} ${EX})
    target_link_libraries(${NAME} PRIVATE C_KAN m)
endforeach()

# 🔹 Тесты (интегрируются с панелью CLion)
file(GLOB TESTS "tests/*.c")
enable_testing()
foreach(T ${TESTS})
    get_filename_component(NAME ${T} NAME_WE)
    add_executable(test_${NAME} ${T})
    target_link_libraries(test_${NAME} PRIVATE C_KAN m)
    add_test(NAME ${NAME} COMMAND test_${NAME})
endforeach()

# 🔹 Опционально: Address/UBSan для отладки (раскомментируй, если нужны)
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     target_compile_options(C_KAN PRIVATE -fsanitize=address,undefined)
#     target_link_options(C_KAN PRIVATE -fsanitize=address,undefined)
# endif()

