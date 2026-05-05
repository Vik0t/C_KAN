cmake_minimum_required(VERSION 3.16)
project(C_KAN C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)


# общие настройки
include_directories(include)

# все .c файлы из src
file(GLOB_RECURSE SRC_FILES
        src/*.c
)

# библиотека
add_library(C_KAN STATIC ${SRC_FILES})

target_include_directories(C_KAN PUBLIC include)


# демо
add_executable(fit_function examples/fit_function.c)
target_link_libraries(fit_function C_KAN)

add_executable(solve_ode examples/solve_ode.c)
target_link_libraries(solve_ode C_KAN)

add_executable(compare_models examples/compare_models.c)
target_link_libraries(compare_models C_KAN)


# тесты
enable_testing()

add_executable(test_mlp tests/test_mlp.c)
target_link_libraries(test_mlp C_KAN)
add_test(NAME test_mlp COMMAND test_mlp)

add_executable(test_kan tests/test_kan.c)
target_link_libraries(test_kan C_KAN)
add_test(NAME test_kan COMMAND test_kan)

add_executable(test_dataset tests/test_dataset.c)
target_link_libraries(test_dataset C_KAN)
add_test(NAME test_dataset COMMAND test_dataset)


source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SRC_FILES})