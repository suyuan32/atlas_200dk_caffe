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
 * @brief activation quantization aware training layer source file
 *
 * @file activation_qat_layer.cpp
 *
 * @version 1.0
 */
#include <vector>

#include "caffe/layers/activation_qat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void ActivationQATLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>* >& bottom,
                                            const vector<Blob<Dtype>* >& top)
{
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int count = bottom[0]->count();
    if (!record_flag_) {
        std::string delim = "-";
        auto start = 0U;
            object_layer_ = object_layer_ + "-";
        auto end = object_layer_.find(delim);

        while (end != std::string::npos) {
            auto layer = object_layer_.substr(start, end - start);
            Status ret = RecordRetrainShape(
                this->layer_param_.retrain_data_quant_param().record_file_path(),
                layer, bottom[0]->channels(),
                bottom[0]->height(), bottom[0]->width());
            CHECK_EQ(ret, 0) << "Record shape to file failed.";
            LOG(INFO) << "Do layer:\"" << layer << "\" activation retrain success!";
                start = end + delim.length();
                end = object_layer_.find(delim, start);
        }
        record_flag_ = true;
    }
    caffe_copy(count, bottom_data, top_data);
    Dtype* clip_max = this->blobs_[0]->mutable_cpu_data();
    Dtype* clip_min = this->blobs_[1]->mutable_cpu_data();
    bool flip = ClipCheck(clip_max, clip_min, clip_max_pre_, clip_min_pre_);
    if (flip) {
        LOG(INFO) << "Clip max value is less or equal to clip min value!";
    }
    if (ifmr_init_ && ifmrParam_.calibration == 0) {
        Status ret = IfmrQuant(top_data, count, ifmrParam_, scale_, offset_);
        *clip_max =
            scale_data_ * (pow(BINARY_BASE, NUM_BITS_QUANT) - 1) + *clip_min;
        *clip_min = scale_data_ *
            -(offset_data_ + pow(BINARY_BASE, NUM_BITS_QUANT - 1));
        CHECK_EQ(ret, 0) << "IFMR initialize failed!";
        ifmrParam_.calibration = 1;
    }
    if (fixed_min_) {
        *clip_min = 0;
    }
    scale_data_ =
        (*clip_max - *clip_min) / (pow(BINARY_BASE, NUM_BITS_QUANT) - 1);
    offset_data_ = round(*clip_min / scale_data_);
    Ulq(top_data, count, scale_, offset_);
}

template <typename Dtype>
void ActivationQATLayer<Dtype>::Backward_cpu(
        const vector<Blob<Dtype>* >& top, const vector<bool>& propagate_down,
        const vector<Blob<Dtype>* >& bottom)
{
    const int count = bottom[0]->count();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(count, top_diff, bottom_diff);
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype clip_max =
        scale_data_ * (offset_data_ + pow(BINARY_BASE, NUM_BITS_QUANT) - 1);
    Dtype clip_min = scale_data_ * offset_data_;
    for (int i = 0; i < count; i++) {
        if (bottom_data[i] < clip_min) {
            bottom_diff[i] = 0;
        } else if (bottom_data[i] > clip_max) {
            bottom_diff[i] = 0;
        }
    }
    Dtype* clip_max_diff = this->blobs_[0]->mutable_cpu_diff();
    Dtype* clip_min_diff = this->blobs_[1]->mutable_cpu_diff();
    Dtype clip_max_ori =
        static_cast<Dtype* >(this->blobs_[0]->mutable_cpu_data())[0];
    Dtype clip_min_ori =
        static_cast<Dtype* >(this->blobs_[1]->mutable_cpu_data())[0];
    vector<Dtype> clip {clip_max, clip_min, clip_max_ori, clip_min_ori};
    std::vector<Dtype> clip_diff = UlqDiff(bottom_data, top_diff, count, clip);
    *clip_max_diff = clip_diff[0];
    *clip_min_diff = clip_diff[1];
}

#ifdef CPU_ONLY
    STUB_GPU(ActivationQATLayer);
#endif

INSTANTIATE_CLASS(ActivationQATLayer);
REGISTER_LAYER_CLASS(ActivationQAT);
}  // namespace caffe
