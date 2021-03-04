/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @brief weight quantization aware training layer source file
 *
 * @file weight_qat_layer.cpp
 *
 * @version 1.0
 */
#include <vector>

#include "caffe/layers/weight_qat_layer.hpp"
#include "caffe/util/math_functions.hpp"
 
namespace caffe {
template <typename Dtype>
void WeightQATLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>* >& bottom,
                                        const vector<Blob<Dtype>* >& top)
{
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int count = bottom[0]->count();
    caffe_copy(count, bottom_data, top_data);
    Dtype* weight_after_quant_data = this->blobs_[0]->mutable_cpu_data();
    Dtype* weight_before_quant_data = this->blobs_[1]->mutable_cpu_data();
    const unsigned int length =
        static_cast<unsigned int>(this->blobs_[0]->count());
    vector<int> weight_shape = this->blobs_[0]->shape();
    if (!is_copy_) {
        caffe_copy(length, weight_after_quant_data, weight_before_quant_data);
        is_copy_ = true;
    }
    caffe_copy(length, weight_before_quant_data, weight_after_quant_data);
    if (arq_param_.channelWise) {
        if (layer_type_ == "Deconvolution") {
            scale_.length = weight_shape[1];
            scale_data_.resize(weight_shape[1]);
            offset_.length = weight_shape[1];
            offset_data_.resize(weight_shape[1]);
        } else {
            scale_.length = weight_shape[0];
            scale_data_.resize(weight_shape[0]);
            offset_.length = weight_shape[0];
            offset_data_.resize(weight_shape[0]);
        }
    } else {
        scale_.length = 1;
        scale_data_.resize(1);
        offset_.length = 1;
        offset_data_.resize(1);
    }
    scale_.data = scale_data_.data();
    offset_.data = offset_data_.data();
    Status ret;
    if (layer_type_ == "Deconvolution") {
        TransposeAB(weight_after_quant_data, length, weight_shape);
        ret = ArqQuant(weight_after_quant_data, length, arq_param_, scale_,
            offset_);
        TransposeAB(weight_after_quant_data, length, weight_shape);
    } else {
        ret = ArqQuant(weight_after_quant_data, length, arq_param_, scale_,
            offset_);
    }
    CHECK_EQ(ret, 0) << "ARQ failed!";
}

template <typename Dtype>
void WeightQATLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>* >& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>* >& bottom)
{
    const int count = bottom[0]->count();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(count, top_diff, bottom_diff);
    const int length = this->blobs_[1]->count();
    const Dtype* weight_after_quant_diff = this->blobs_[0]->cpu_diff();
    Dtype* weight_before_quant_diff = this->blobs_[1]->mutable_cpu_diff();
    for (int i = 0; i < length; i++) {
        weight_before_quant_diff[i] = weight_after_quant_diff[i];
    }
}

#ifdef CPU_ONLY
STUB_GPU(WeightQATLayer);
#endif

INSTANTIATE_CLASS(WeightQATLayer);
REGISTER_LAYER_CLASS(WeightQAT);
}  // namespace caffe
