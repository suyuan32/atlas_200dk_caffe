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
 * @brief anti-quant layer src file
 *
 * @file anti_quant_layer.cpp
 *
 * @version 1.0
 */
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include "caffe/layers/anti_quant_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/amct_util.hpp"

namespace caffe {
template <typename Dtype>
void AntiQuantLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>* >& bottom,
                                        const vector<Blob<Dtype>* >& top)
{
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const int count = bottom[0]->count();
    Dtype* top_data = top[0]->mutable_cpu_data();
    caffe_copy(count, bottom_data, top_data);
    int batchLength = bottom[0]->channels() * bottom[0]->height() * bottom[0]->width();
    if (recordData_) {
        std::string recordLayerName = this->layer_param_.name();
        ConvertLayerName(recordLayerName, "/", REPLACE_STR);
        std::string fileName = "./amct_log/" + recordLayerName + ".log";
        INIT_LOG();
        DEBUG_LOG(fileName.c_str(), "[AMCT] bottom data of anti_quant:\n");
        DATA_DEBUG_LOG(fileName.c_str(), bottom[0]->cpu_data(), batchLength);
        DEBUG_LOG(fileName.c_str(), "[AMCT] top data of anti_quant:\n");
        DATA_DEBUG_LOG(fileName.c_str(), top[0]->cpu_data(), batchLength);
        recordData_ = false;
    }
}

template <typename Dtype>
void AntiQuantLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>* >& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>* >& bottom)
{
    const Dtype* top_diff = top[0]->cpu_diff();
    const int count = bottom[0]->count();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    caffe_copy(count, top_diff, bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(AntiQuantLayer);
#endif

INSTANTIATE_CLASS(AntiQuantLayer);
REGISTER_LAYER_CLASS(AntiQuant);
}  // namespace caffe
