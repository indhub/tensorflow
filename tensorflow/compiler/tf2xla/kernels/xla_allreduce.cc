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

#include <chrono>
#include <thread>

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

    const float* input = reinterpret_cast<const float*>(buffers[0]);
    float* output = reinterpret_cast<float*>(buffers[2]);

    const uint32* var_id_gpu = reinterpret_cast<const uint32*>(buffers[1]);
    const int64 flat_len = (int64) atoi(opaque);

    uint32 var_id_cpu;
    cudaMemcpy(&var_id_cpu, var_id_gpu, sizeof(uint32), cudaMemcpyDeviceToHost);

    cudaMemcpy(output, input, flat_len * sizeof(float), cudaMemcpyDeviceToDevice);
    /*
    // Get ptr to input and output
    const float* input = reinterpret_cast<const float*>(buffers[0]);
    float* output = reinterpret_cast<float*>(buffers[1]);

    // Get buffer length
    int buffer_len = 0;
    getInt(opaque, &buffer_len);

    CUDACHECK(cudaMemcpy(output, input, buffer_len * sizeof(float), cudaMemcpyDeviceToDevice));
    */
    unsigned long milliseconds_since_epoch = std::chrono::duration_cast<std::chrono::milliseconds> (std::chrono::system_clock::now().time_since_epoch()).count();
    std::cout << "var_id: " << var_id_cpu << " len: " << flat_len << " Time: " << milliseconds_since_epoch << std::endl;
}

XLA_REGISTER_CUSTOM_CALL_TARGET(do_custom_call, "CUDA");


/*
void do_custom_call(void* out, const void** in) {
}

XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(do_custom_call);
*/

class XlaAllReduceOp : public XlaOpKernel {
 public:
  explicit XlaAllReduceOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {

    // Get input and input shape from context
    xla::XlaOp input = ctx->Input(0);
    const TensorShape input_shape = ctx->InputShape(0);

    // Output is just one int32 which is an opaque data 
    // that must be handed back to the operator that completes the AllReduce
    TensorShape output_shape;
    //output_shape.AddDim(1);
    int64 flat_len = 1;
    for (int d = 0; d < input_shape.dims(); ++d) {
      int64 dim_size = input_shape.dim_size(d);
      output_shape.AddDim(dim_size);

      flat_len *= dim_size;
    }

    // Output datatype is int32
    const DataType dtype = output_type(0);
    //const DataType dtype = DataType::DT_INT32;
    xla::PrimitiveType output_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(dtype, &output_type));

    // Grab the XLA Builder from context
    xla::XlaBuilder& b = *ctx->builder();

    // Create args
    std::vector<xla::XlaOp> args;
    args.push_back(ctx->Input(0));
    args.push_back(ctx->Input(1));

    // Create the custom call
    std::string opaque = std::to_string(flat_len) + "\0";
    xla::XlaOp output = xla::CustomCall(&b, "do_custom_call", args, xla::ShapeUtil::MakeShape(output_type, output_shape.dim_sizes()), opaque);

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
