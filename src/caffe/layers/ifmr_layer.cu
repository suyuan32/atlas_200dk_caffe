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
 * @brief ifmr layer cuda src file
 *
 * @file ifmr_layer.cu
 *
 * @version 1.0
 */
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include "caffe/layers/ifmr_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"
namespace caffe {

template <typename Dtype>
void IFMRLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>* >& bottom,
                                   const vector<Blob<Dtype>* >& top)
{
    Dtype* device_bottom_data = bottom[0]->mutable_gpu_data();
    int ret;
    // DO IFMR calibration
    if (ifmrParam_.calibration == 0) {
        // Phase 0, do calibration, search best scale and offset for feature map
        if (numBatchStored_ < targetBatchNum_) {
            //If numBatchStored_ less than batch_num, then store data in vector
            for (size_t bottomIndex = 0; bottomIndex < bottom.size(); ++bottomIndex) {
                const int count = bottom[bottomIndex]->count();
                const Dtype* bottom_data = bottom[bottomIndex]->cpu_data();
                for (int index = 0; index < count; ++index) {
                    storedDataForCalibration_.push_back(bottom_data[index]);
                }
            }
            numBatchStored_++;
            for (auto objectLayerName : objectLayerNames_) {
                LOG(INFO) <<  "Doing layer: \"" << objectLayerName << "\" calibration, already store "
                    << numBatchStored_ << "/" << targetBatchNum_ << " data.";
            }
        }
        if (numBatchStored_ == targetBatchNum_) {
            // Already have enough data, then do once IFMR calibration.

            LOG(INFO) << "Start to do ifmr quant.";
            ret = IfmrQuant_gpu(device_bottom_data,
                storedDataForCalibration_.data(), storedDataForCalibration_.size(), ifmrParam_, scale_, offset_);
            storedDataForCalibration_.clear();
            storedDataForCalibration_.shrink_to_fit();
            CHECK_EQ(ret, 0) << "Do IFMR calibration failed";
            // Record scale and offset
            for (int index = 0; index < objectLayerNames_.size(); ++index) {
               ret = RecordScaleOffset(this->layer_param_.ifmr_param().record_file_path(),
                                       objectLayerNames_.Get(index),
                                       scale_,
                                       offset_,
                                       bottom[index]->channels(),
                                       bottom[index]->height(),
                                       bottom[index]->width());
                CHECK_EQ(ret, 0) << "Record scale and offset to file failed.";
                LOG(INFO) << "Do layer:\"" << objectLayerNames_.Get(index) << "\" activation calibration success!";
            }
            ifmrParam_.calibration = 1;  // Phase 0 end, set calibration flag to test
        }
    }
}

template <typename Dtype>
void IFMRLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>* >& top,
                                    const vector<bool>& propagate_down,
                                    const vector<Blob<Dtype>* >& bottom)
{
    if (!propagate_down[0]) { return; }
}

INSTANTIATE_LAYER_GPU_FUNCS(IFMRLayer);
}  // namespace caffe
