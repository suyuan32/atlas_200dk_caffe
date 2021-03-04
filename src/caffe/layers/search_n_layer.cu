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
 * @brief search shift bit layer cuda source
 *
 * @version 1.0
 */
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include "caffe/layers/search_n_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/amct_util.hpp"

namespace caffe {
template <typename Dtype>
void SearchNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>* >& bottom,
                                      const vector<Blob<Dtype>* >& top)
{
    if (curBatchNum < targetBatchNum) {
        AccumulateData(bottom);
        curBatchNum++;
        LOG(INFO) << "Doing layer: " << showQuantLayerNames << " search shift bits calibration, already store "
                << curBatchNum << "/" << targetBatchNum << " data.";
    }

    if (curBatchNum == targetBatchNum) {
        GetDeqScale();
        vector<int> bestN(storedData.size());
        int ret = SearchN_gpu(storedData, deqScale, bestN);
        CHECK_EQ(ret, 0) << "search shift n value failed!";
        RecordN(bestN);
        LOG(INFO) << "Do layer: " << showQuantLayerNames << " search shift bits calibration success!";

        storedData.clear();
        storedData.shrink_to_fit();
        curBatchNum++;
    }

    if (top.size() == 1) {
        const int count = bottom[0]->count();
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        caffe_copy(count, bottom_data, top_data);
    }
}

template <typename Dtype>
void SearchNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>* >& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom)
{
    return;
}

INSTANTIATE_LAYER_GPU_FUNCS(SearchNLayer);
}  // namespace caffe
