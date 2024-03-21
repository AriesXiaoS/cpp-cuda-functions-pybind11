

file(GLOB_RECURSE SRC_FILES source/*.cpp source/*.cu source/*/*.cpp source/*/*.cu)


add_library(cuda_functions SHARED
${SRC_FILES}
)
target_include_directories(cuda_functions PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

#target_include_directories(cuda_functions PUBLIC ${Python3_INCLUDE_DIRS})
target_include_directories(cuda_functions PUBLIC ${CMAKE_SOURCE_DIR}/pybind11/include)
target_link_libraries(cuda_functions PUBLIC
  #${PYTHON3_PYTHON}
  Python3::Python
  cudart)

# set what comes before the library name, here no "lib"
set_target_properties(cuda_functions PROPERTIES PREFIX "")

configure_file(test.py test.py COPYONLY)


# 

