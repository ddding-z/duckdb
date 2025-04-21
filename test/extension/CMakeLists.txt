set(PARAMETERS "-warnings")

build_loadable_extension_directory(
  loadable_extension_demo
  "CPP"
  test/extension
  "default-version"
  ""
  ${PARAMETERS}
  loadable_extension_demo.cpp)

# retree_inference_function
build_loadable_extension_directory(
  retree_inference_function_extension
  "CPP"
  test/extension
  "default-version"
  ""
  ${PARAMETERS}
  retree_inference_function_extension.cpp)

if(NOT WIN32 AND NOT SUN)
  add_definitions(-DDUCKDB_BUILD_DIRECTORY="${PROJECT_BINARY_DIR}")

  build_loadable_extension_directory(
    loadable_extension_optimizer_demo
    "CPP"
    test/extension
    "default-version"
    ""
    ${PARAMETERS}
    ../extension/loadable_extension_optimizer_demo.cpp)

  # retree_optimization_function
  build_loadable_extension_directory(
    retree_convert_rule_extension
    "CPP"
    test/extension
    "default-version"
    ""
    ${PARAMETERS}
    ../extension/retree_convert_rule_extension.cpp)
  
  build_loadable_extension_directory(
    retree_prune_rule_extension
    "CPP"
    test/extension
    "default-version"
    ""
    ${PARAMETERS}
    ../extension/retree_prune_rule_extension.cpp)
  
  build_loadable_extension_directory(
    retree_merge_rule_extension
    "CPP"
    test/extension
    "default-version"
    ""
    ${PARAMETERS}
    ../extension/retree_merge_rule_extension.cpp)

  build_loadable_extension_directory(
    retree_naive_merge_rule_extension
    "CPP"
    test/extension
    "default-version"
    ""
    ${PARAMETERS}
    ../extension/retree_naive_merge_rule_extension.cpp)

  build_loadable_extension_directory(
    retree_extension
    "CPP"
    test/extension
    "default-version"
    ""
    ${PARAMETERS}
    ../extension/retree_extension.cpp)
  
    build_loadable_extension_directory(
    retree_prune_extension
    "CPP"
    test/extension
    "default-version"
    ""
    ${PARAMETERS}
    ../extension/retree_prune_extension.cpp)

  set(TEST_EXT_OBJECTS test_remote_optimizer.cpp)

  add_library_unity(test_extensions OBJECT ${TEST_EXT_OBJECTS})
  set(ALL_OBJECT_FILES
      ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:test_extensions>
      PARENT_SCOPE)
endif()
