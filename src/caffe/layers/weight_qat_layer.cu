#include <thrust/extrema.h>//lint !e7
#include <thrust/execution_policy.h>//lint !e7
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/weight_qat_layer.hpp"


template <typename Dtype>
__global__ void BlobTranspose(const int n, const Dtype* in, Dtype* out, const int cin, const int cout, const int hw) {
    CUDA_KERNEL_LOOP(index, n) {
        int cin_index = index / (cout * hw);
        int cout_index = index % (cout * hw) / hw;
        int transe_index = cout_index * (cin * hw) + cin_index * hw + index % hw;
        out[transe_index] = in[index];
    }
}

namespace caffe {

template <typename Dtype>
void WeightQATLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int count = bottom[0]->count();

    caffe_copy(count, bottom_data, top_data);

    Blob<Dtype>* weight_before_quant_blob;
    weight_before_quant_blob = this->blobs_[1].get();
    Dtype* weight_after_quant_data = this->blobs_[0]->mutable_gpu_data();
    Dtype* weight_before_quant_data = this->blobs_[1]->mutable_gpu_data();
    const unsigned int length = static_cast<unsigned int>(this->blobs_[0]->count());

    vector<int> weight_shape = weight_before_quant_blob->shape();
    if(!is_copy_) {
        caffe_copy(length, weight_after_quant_data, weight_before_quant_data);
        is_copy_ = true;
    }
    caffe_copy(length, weight_before_quant_data, weight_after_quant_data);
    Dtype* weight_after_quant_data_cpu = this->blobs_[0]->mutable_cpu_data();

    if (arq_param_.channelWise) {
        scale_.length = weight_shape[0];
        float scale[weight_shape[0]];
        scale_.data = scale;
        offset_.length = weight_shape[0];
        int offset[weight_shape[0]];
        offset_.data = offset;
        if (layer_type_ == "Deconvolution") {
            vector<int> weight_shape_transpose = weight_shape;
            weight_shape_transpose[0] = weight_shape[0];
            weight_shape_transpose[1] = weight_shape[1];
            shared_ptr<Blob<Dtype>> weight_transpose(new Blob<Dtype>(weight_shape_transpose));
            Dtype* weight_transepose_data = weight_transpose->mutable_gpu_data();
            BlobTranspose<Dtype><<<CAFFE_GET_BLOCKS(length), CAFFE_CUDA_NUM_THREADS>>>(
                length, weight_before_quant_data, weight_transepose_data, weight_shape[0], weight_shape[1],
                weight_shape[3] * weight_shape[2]);
            int ret = ArqQuantRetrainGPU(weight_transepose_data, length, arq_param_, scale_, offset_);
            CHECK_EQ(ret, 0) << "Arq quant gpu failed!";
            BlobTranspose<Dtype><<<CAFFE_GET_BLOCKS(length), CAFFE_CUDA_NUM_THREADS>>>(
                length, weight_transepose_data, weight_after_quant_data, weight_shape[0], weight_shape[1],
                weight_shape[3] * weight_shape[2]);
        } else {
            int ret = ArqQuantRetrainGPU(weight_after_quant_data, length, arq_param_, scale_, offset_);
            CHECK_EQ(ret, 0) << "Arq quant gpu failed!";
        }
    } else {
        scale_.length = 1;
        float scale;
        scale_.data = &scale;
        offset_.length = 1;
        int offset;
        offset_.data = &offset;
        int ret = ArqQuantRetrainGPU(weight_after_quant_data, length, arq_param_, scale_, offset_);
        CHECK_EQ(ret, 0) << "Arq quant gpu failed! ";
    }
}


template <typename Dtype>
void WeightQATLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    const int count = bottom[0]->count();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    caffe_copy(count, top_diff, bottom_diff);

    const int length = this->blobs_[1]->count();
    const Dtype* weight_after_quant_diff = this->blobs_[0]->gpu_diff();
    Dtype* weight_before_quant_diff = this->blobs_[1]->mutable_gpu_diff();

    caffe_copy(length, weight_after_quant_diff, weight_before_quant_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(WeightQATLayer);

}  // namespace caffe
