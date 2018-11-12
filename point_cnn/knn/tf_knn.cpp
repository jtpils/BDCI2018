#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>

using namespace tensorflow;
REGISTER_OP("KNN")
.Attr("k: int")
.Input("queries: float32")
.Input("points: float32")
.Output("dis: float32")
.Output("indices: int32")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * queries_num * channels
    c->WithRank(c->input(0), 3, &dims1);
    ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * points_num * channels
    c->WithRank(c->input(1), 3, &dims2);
    int k;
    TF_RETURN_IF_ERROR(c->GetAttr("k", &k));

    ::tensorflow::shape_inference::ShapeHandle output_dis = c->MakeShape({c->Dim(dims1, 0),
                                                                      c->Dim(dims1, 1),
                                                                      c->Dim(dims1, 2)});
    c->set_output(0, output_dis);
    ::tensorflow::shape_inference::ShapeHandle output_ids = c->MakeShape({c->Dim(dims1, 0),
                                                                      c->Dim(dims1, 1),
                                                                      c->Dim(dims1, 2)});
    c->set_output(1, output_ids);
    return Status::OK();
});


void knnLauncher(int batch_size, int qrs_num, int pts_num, int channels_num,
                 const float *queries, const float *points, int k,
                 float *out_dis, int *out_ids);
class KNNGpuOp: public OpKernel{
public:
    explicit KNNGpuOp(OpKernelConstruction* context):OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
        OP_REQUIRES(context, k_ > 0, errors::InvalidArgument("KNN expects positive k"));
    }
    void Compute(OpKernelContext * context)override{
        int k = k_;

        const Tensor& queries_tensor=context->input(0);
        OP_REQUIRES(context,queries_tensor.dims()==3 && queries_tensor.shape().dim_size(2)==3,
                    errors::InvalidArgument("KNN expects (batch_size,num_points,3) queries shape"));
        int batch_size = queries_tensor.shape().dim_size(0);
        int qrs_num = queries_tensor.shape().dim_size(1);
        int channels_num = queries_tensor.shape().dim_size(2);
        auto queries_flat=queries_tensor.flat<float>();
        const float * queries=&(queries_flat(0));

        const Tensor& points_tensor=context->input(1);
        OP_REQUIRES(context,points_tensor.dims()==3 && points_tensor.shape().dim_size(2)==3,
                    errors::InvalidArgument("KNN expects (batch_size,num_points,3) points shape"));
        int pts_num = points_tensor.shape().dim_size(1);
        auto points_flat=points_tensor.flat<float>();
        const float * points=&(points_flat(0));


        Tensor * out_tensor_dis;
        OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{batch_size, qrs_num, 3},&out_tensor_dis));
        auto out_flat_dis=out_tensor_dis->flat<float>();
        float * out_dis=&(out_flat_dis(0));

        Tensor * out_tensor_ids;
        OP_REQUIRES_OK(context,context->allocate_output(1,TensorShape{batch_size, qrs_num, 3},&out_tensor_ids));
        auto out_flat_ids=out_tensor_ids->flat<int>();
        int * out_ids=&(out_flat_ids(0));

        knnLauncher(batch_size, qrs_num, pts_num, channels_num, queries, points, k, out_dis, out_ids);
    }
private:
    int k_;
};
REGISTER_KERNEL_BUILDER(Name("KNN").Device(DEVICE_GPU),KNNGpuOp)
