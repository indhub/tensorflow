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
    const TensorShape input_shape = ctx->InputShape(0);

    const DataType dtype = output_type(0);
    xla::PrimitiveType output_type;
    OP_REQUIRES_OK(ctx, DataTypeToPrimitiveType(dtype, &output_type));

    xla::XlaOp output;
    TensorShape output_shape;
    for (int d = 0; d < input_shape.dims(); ++d) {
      output_shape.AddDim(input_shape.dim_size(d));
    }

    xla::XlaBuilder& b = *ctx->builder();

    std::string opaque = "whatever";
    xla::XlaOp param0 = xla::Parameter(&b, 0, xla::ShapeUtil::MakeShape(xla::F32, {128}), "p0");
    xla::CustomCall(&b, "do_custom_call", {param0}, xla::ShapeUtil::MakeShape(xla::F32, {2048}), opaque);

    output = xla::ConvertElementType(output, output_type);
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
