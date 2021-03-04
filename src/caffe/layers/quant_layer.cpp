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
 * @brief quant layer src file
 *
 * @file quant_layer.cpp
 *
 * @version 1.0
 */
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <cfloat>
#include "caffe/layers/quant_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/amct_util.hpp"

namespace caffe {
template <typename Dtype>
void QuantLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>* >& bottom,
                                   const vector<Blob<Dtype>* >& top)
{
    scale_.length = 1;
    scale_.data = &scaleData_;
    offset_.length = 1;
    offset_.data = &offsetData_;
    ifmrParam_.calibration = 1; // skip calibration, do fake quant only
    ifmrParam_.numBits = NUM_BITS_QUANT;
    ifmrParam_.withOffset = this->layer_param_.quant_param().with_offset();
    ifmrParam_.startRatio = 0;
    ifmrParam_.endRatio = 0;
    ifmrParam_.step = 0;
    ifmrParam_.maxPercentile = 0;
    ifmrParam_.minPercentile = 0;
    if (!this->layer_param_.quant_param().has_scale()) {
        LOG(ERROR) << "Cannot find \"scale\" in layer " << this->layer_param_.name() << " quant_param";
    }
    CHECK_EQ(std::isinf(this->layer_param_.quant_param().scale()), false) <<
        "quant param scale should not be inf!";
    CHECK_GT(static_cast<double>(this->layer_param_.quant_param().scale()), DBL_EPSILON) <<
        "quant param scale should be greater than DBL_EPSILON!";

    scaleData_ = 1 / static_cast<double>(this->layer_param_.quant_param().scale());
    LOG(INFO) << "Find \"scale\"=" << scaleData_ << " in layer " << this->layer_param_.name() << " quant_param";
    if (!this->layer_param_.quant_param().has_offset()) {
        LOG(ERROR) << "Cannot find \"offset\" in layer" << this->layer_param_.name() << " quant_param";
    }
    offsetData_ = static_cast<int>(*(this->layer_param_.quant_param().offset().begin()));
    LOG(INFO) << "Find \"offset\"=" << offsetData_ << " in layer " << this->layer_param_.name() << " quant_param";
}

template <typename Dtype>
void QuantLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>* >& bottom,
                                    const vector<Blob<Dtype>* >& top)
{
    const int count = bottom[0]->count();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    caffe_copy(count, bottom_data, top_data);
    int ret = IfmrQuant(top_data, count, ifmrParam_, scale_, offset_);
    CHECK_EQ(ret, 0) << "do IFMR quantization failed.";

    int recordNum = bottom[0]->channels() * bottom[0]->height() * bottom[0]->width();
    if (recordData_) {
        std::string layerName = this->layer_param_.name();
        ConvertLayerName(layerName, "/", REPLACE_STR);
        std::string fileName = "./amct_log/" + layerName + ".log";
        INIT_LOG();
        DEBUG_LOG(fileName.c_str(), "[AMCT] bottom data of fake quant:\n");
        DATA_DEBUG_LOG(fileName.c_str(), bottom[0]->cpu_data(), recordNum);
        DEBUG_LOG(fileName.c_str(), "[AMCT] top data of fake quant:\n");
        DATA_DEBUG_LOG(fileName.c_str(), top[0]->cpu_data(), recordNum);
        recordData_ = false;
    }
}

template <typename Dtype>
void QuantLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>* >& top,
                                     const vector<bool>& propagate_down,
                                     const vector<Blob<Dtype>* >& bottom)
{
    const int count = bottom[0]->count();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(count, top_diff, bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(QuantLayer);
#endif

INSTANTIATE_CLASS(QuantLayer);
REGISTER_LAYER_CLASS(Quant);
}  // namespace caffe
