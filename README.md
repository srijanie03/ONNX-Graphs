# ONNX

Repo to store examples for generating ONNX Runtime Training Graphs

To build ONNX Runtime Training:
```
 .\build.bat --config RelWithDebInfo --cmake_generator "Visual Studio 17 2022" --build_shared_lib --parallel --enable_training --cmake_path "C:\Program Files\CMake\bin\cmake.exe" --ctest_path "C:\Program Files\CMake\bin\ctest.exe" --build_wheel
```

Example Training Graph :
![training_model onnx](https://github.com/srijanie03/ONNX/assets/34174706/f28d458d-870f-47f6-b28b-38a8ce6273fa)

Converted Graph Rundown:
```
graph main_graph (
  %input[FLOAT, batch_sizex784]
  %target[FLOAT, batch_sizex10]
  %fc1.weight[FLOAT, 500x784]
  %fc1.bias[FLOAT, 500]
  %fc2.weight[FLOAT, 10x500]
  %fc2.bias[FLOAT, 10]
  %fc2.weight_grad.accumulation.buffer[FLOAT, 10x500]
  %fc2.bias_grad.accumulation.buffer[FLOAT, 10]
  %lazy_reset_grad[BOOL, 1]
) initializers (
  %onnx::pow_exponent::3[FLOAT, 1]
  %onnx::reducemean_output::6_grad[FLOAT, scalar]
  %/fc2/Gemm_Grad/ReduceAxes_for_/fc2/Gemm_Grad/dC_reduced[INT64, 1]
  %OneConstant_Type1[FLOAT, 1]
) {
  %/fc1/Gemm_output_0 = Gemm[alpha = 1, beta = 1, transA = 0, transB = 1](%input, %fc1.weight, %fc1.bias)
  %/relu/Relu_output_0 = Relu(%/fc1/Gemm_output_0)
  %output = Gemm[alpha = 1, beta = 1, transA = 0, transB = 1](%/relu/Relu_output_0, %fc2.weight, %fc2.bias)
  %onnx::sub_output::1 = Sub(%output, %target)
  %onnx::pow_output::4 = Pow(%onnx::sub_output::1, %onnx::pow_exponent::3)
  %onnx::ReduceMean::7_Grad/Sized_X = Size(%onnx::pow_output::4)
  %onnx::ReduceMean::7_Grad/Shaped_X = Shape(%onnx::pow_output::4)
  %onnx::ReduceMean::7_Grad/Sized_Grad = Size(%onnx::reducemean_output::6_grad)
  %onnx::ReduceMean::7_Grad/Scale = Div(%onnx::ReduceMean::7_Grad/Sized_X, %onnx::ReduceMean::7_Grad/Sized_Grad)
  %onnx::ReduceMean::7_Grad/Scaled_Grad = Scale[scale_down = 1](%onnx::reducemean_output::6_grad, %onnx::ReduceMean::7_Grad/Scale)
  %onnx::pow_output::4_grad = Expand(%onnx::ReduceMean::7_Grad/Scaled_Grad, %onnx::ReduceMean::7_Grad/Shaped_X)
  %onnx::Pow::5_Grad/Sub_I1 = Sub(%onnx::pow_exponent::3, %OneConstant_Type1)
  %onnx::Pow::5_Grad/Pow_I0 = Pow(%onnx::sub_output::1, %onnx::Pow::5_Grad/Sub_I1)
  %onnx::Pow::5_Grad/Mul_Pow_I0_I1 = Mul(%onnx::Pow::5_Grad/Pow_I0, %onnx::pow_exponent::3)
  %onnx::sub_output::1_grad = Mul(%onnx::Pow::5_Grad/Mul_Pow_I0_I1, %onnx::pow_output::4_grad)
  %output_grad = Identity(%onnx::sub_output::1_grad)
  %/fc2/Gemm_Grad/dC_reduced = ReduceSum[keepdims = 0, noop_with_empty_axes = 0](%output_grad, %/fc2/Gemm_Grad/ReduceAxes_for_/fc2/Gemm_Grad/dC_reduced)
  %fc2.bias_grad = Identity(%/fc2/Gemm_Grad/dC_reduced)
  %fc2.weight_grad = Gemm[alpha = 1, beta = 0, transA = 1, transB = 0](%output_grad, %/relu/Relu_output_0)
  %onnx::reducemean_output::6 = ReduceMean[keepdims = 0](%onnx::pow_output::4)
  %fc2.weight_grad.accumulation.out = InPlaceAccumulatorV2(%fc2.weight_grad.accumulation.buffer, %fc2.weight_grad, %lazy_reset_grad)
  %fc2.bias_grad.accumulation.out = InPlaceAccumulatorV2(%fc2.bias_grad.accumulation.buffer, %fc2.bias_grad, %lazy_reset_grad)
  return %onnx::reducemean_output::6, %output, %fc2.weight_grad.accumulation.out, %fc2.bias_grad.accumulation.out
}
```
