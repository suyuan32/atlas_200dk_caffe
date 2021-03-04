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
 * @brief ifmr layer src file
 *
 * @file ifmr_layer.cpp
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
void RecordLayerScaleOffset(SingleLayerRecord* layer_record,
                            FloatData scale,
                            IntData offset,
                            unsigned int channels,
                            unsigned int height,
                            unsigned int width)
{
        // record scale_d
        layer_record->clear_scale_d();
        for (unsigned int index = 0; index < scale.length; ++index) {
            layer_record->set_scale_d(scale.data[index]);
        }
        // record offset_w
        layer_record->clear_offset_d();
        for (unsigned int index = 0; index < offset.length; ++index) {
             layer_record->set_offset_d(offset.data[index]);
        }
        layer_record->set_channels(channels);
        layer_record->set_height(height);
        layer_record->set_width(width);
}

template <typename Dtype>
int IFMRLayer<Dtype>::RecordScaleOffset(std::string file_name,
                                        std::string layer_name,
                                        FloatData scale,
                                        IntData offset,
                                        unsigned int channels,
                                        unsigned int height,
                                        unsigned int width)
{
    ScaleOffsetRecord records;
    if (!ReadProtoFromTextFile(file_name.c_str(), &records)) {
        LOG(ERROR) << "Read records from " << file_name << " failed.";
        return -1;
    }
    bool found_layer = false;
    for (int i = 0; i < records.record_size(); ++i) {
        ScaleOffsetRecord_MapFiledEntry* record = records.mutable_record(i);
        if (record->has_key() && record->key() == layer_name) {
            SingleLayerRecord* found_layer_record = record->mutable_value();
            RecordLayerScaleOffset(found_layer_record, scale,
                                   offset, channels, height, width);
            found_layer = true;
            break;
        }
    }
    if (!found_layer) {
        ScaleOffsetRecord_MapFiledEntry* record = records.add_record();
        record->set_key(layer_name);
        SingleLayerRecord* new_layer_record = record->mutable_value();
        RecordLayerScaleOffset(new_layer_record, scale, offset,
                               channels, height, width);
    }
    WriteProtoToTextFile(records, file_name.c_str());
    return 0;
}

template <typename Dtype>
void IFMRLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>* >& bottom,
                                  const vector<Blob<Dtype>* >& top)
{
    scale_.length = 1;
    scale_.data = &scaleData_;
    offset_.length = 1;
    offset_.data = &offsetData_;
    // Set IFMR quantize algorithm parameters
    ifmrParam_.calibration = 0; // first phase, store data
    ifmrParam_.numBits = NUM_BITS_QUANT;
    ifmrParam_.withOffset = this->layer_param_.ifmr_param().with_offset();
    ifmrParam_.startRatio = this->layer_param_.ifmr_param().search_range_start();
    ifmrParam_.endRatio = this->layer_param_.ifmr_param().search_range_end();
    ifmrParam_.step = this->layer_param_.ifmr_param().search_step();
    ifmrParam_.maxPercentile = this->layer_param_.ifmr_param().max_percentile();
    ifmrParam_.minPercentile = this->layer_param_.ifmr_param().min_percentile();
    objectLayerNames_ = this->layer_param_.ifmr_param().object_layer();
    targetBatchNum_ = this->layer_param_.ifmr_param().batch_num();
}

template <typename Dtype>
void IFMRLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>* >& bottom,
                                   const vector<Blob<Dtype>* >& top)
{
    int ret;
    // DO IFMR calibration
    if (ifmrParam_.calibration == 0) {
        // Phase 0, do calibration, search best scale and offset for feature map
        if (numBatchStored_ < targetBatchNum_) {
            // If numBatchStored_ < Seted num batch, then store data in vector
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
            LOG(INFO) << "Start to do ifmr quant.";
            // Already have enough data, then do once IFMR calibration.
            ret = IfmrQuant(storedDataForCalibration_.data(), storedDataForCalibration_.size(),
                            ifmrParam_, scale_, offset_);
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

#ifdef CPU_ONLY
STUB_GPU(IFMRLayer);
#endif

INSTANTIATE_CLASS(IFMRLayer);
REGISTER_LAYER_CLASS(IFMR);
}  // namespace caffe
