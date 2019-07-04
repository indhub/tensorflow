#ifndef TENSORFLOW_COMPILER_TF2XLA_KERNELS_XLA_ALLREDUCE_H_
#define TENSORFLOW_COMPILER_TF2XLA_KERNELS_XLA_ALLREDUCE_H_

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

class XlaAllReduceOp : public XlaOpKernel {
 public:
  explicit XlaAllReduceOp(OpKernelConstruction* ctx);
  void Compile(XlaOpKernelContext* ctx) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_KERNELS_XLA_ALLREDUCE_H_
