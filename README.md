# ONNX

Repo to store examples for generating ONNX Runtime Training Graphs

To build ONNX Runtime Training:
```
 .\build.bat --config RelWithDebInfo --cmake_generator "Visual Studio 17 2022" --build_shared_lib --parallel --enable_training --cmake_path "C:\Program Files\CMake\bin\cmake.exe" --ctest_path "C:\Program Files\CMake\bin\ctest.exe" --build_wheel
```

Example Graph
![training_model onnx](https://github.com/srijanie03/ONNX/assets/34174706/f28d458d-870f-47f6-b28b-38a8ce6273fa)
