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

namespace tensorflow {
namespace {

void do_custom_call(void* out, const void** in) {
}

class XlaAllReduceOp : public XlaOpKernel {
 public:
  explicit XlaAllReduceOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {

    // Get input and input shape from context
    xla::XlaOp input = ctx->Input(0);
    const TensorShape input_shape = ctx->InputShape(0);

    // Create output shape
    TensorShape output_shape;
    for (int d = 0; d < input_shape.dims(); ++d) {
        output_shape.AddDim(input_shape.dim_size(d));
    }

    // Get output data type
    const DataType dtype = output_type(0);
    xla::PrimitiveType output_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(dtype, &output_type));

    // Grab the XLA Builder from context
    xla::XlaBuilder& b = *ctx->builder();

    // Create the custom call
    std::string opaque = "not_used";
    xla::XlaOp output = xla::CustomCall(&b, "do_custom_call", {input}, xla::ShapeUtil::MakeShape(output_type, output_shape.dim_sizes()), opaque);

    // Convert to the correct type
    // output = xla::ConvertElementType(output, output_type);
    
    // Set output
    ctx->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(XlaAllReduceOp);
};

REGISTER_XLA_OP(Name("XLAAllReduce")
                    .CompileTimeConstantInput("dimension"),
                XlaAllReduceOp);

}  // namespace
}  // namespace tensorflow
