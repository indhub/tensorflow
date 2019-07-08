#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

namespace tensorflow {
namespace {

const char* getInt(const char *ptr, int *value) {
    
    // Get the first digit
    *value = *ptr - '0';
    ptr++;

    // Get subsequent digits until delimiter
    while(*ptr != '|') {
        *value = *value * 10 + (*ptr - '0');
    }
    ptr++; // Move pointer to the character after delimited

    return ptr;
}

// TODO(thangakr): Modify this function to handle types other than FP32
void do_custom_call(CUstream stream, void** buffers,
        const char* opaque, size_t opaque_len) {

/*    
    // Get ptr to input and output
    const float* input = reinterpret_cast<const float*>(buffers[0]);
    float* output = reinterpret_cast<float*>(buffers[1]);

    // Get buffer length
    int buffer_len = 0;
    getInt(opaque, &buffer_len);

    CUDACHECK(cudaMemcpy(output, input, buffer_len * sizeof(float), cudaMemcpyDeviceToDevice));
*/
}

//XLA_REGISTER_CUSTOM_CALL_TARGET(do_custom_call, "CUDA");
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(do_custom_call);

class XlaAllReduceOp : public XlaOpKernel {
 public:
  explicit XlaAllReduceOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {

    // Get input and input shape from context
    xla::XlaOp input = ctx->Input(0);
    const TensorShape input_shape = ctx->InputShape(0);

    // Create output shape. Note: Input/Output of XlaAllReduce is 1-D
    TensorShape output_shape;
    output_shape.AddDim(input_shape.dim_size(0));

    // Get output data type
    const DataType dtype = output_type(0);
    xla::PrimitiveType output_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(dtype, &output_type));

    // Grab the XLA Builder from context
    xla::XlaBuilder& b = *ctx->builder();

    // Create the custom call
    std::string opaque = std::to_string(input_shape.dim_size(0)) + "|";
    xla::XlaOp output = xla::CustomCall(&b, "do_custom_call", {input}, xla::ShapeUtil::MakeShape(output_type, output_shape.dim_sizes()), opaque);

    // Convert to the correct type
    // output = xla::ConvertElementType(output, output_type);
    
    // Set output
    ctx->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(XlaAllReduceOp);
};

REGISTER_XLA_OP(Name("XlaAllReduce"), XlaAllReduceOp);

}  // namespace
}  // namespace tensorflow
