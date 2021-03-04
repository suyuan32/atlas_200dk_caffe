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
 * @brief dequant layer src file
 *
 * @file dequant_layer.cpp
 *
 * @version 1.0
 */
#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include "caffe/layers/dequant_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/amct_util.hpp"

namespace caffe {
const int NUM_AXES = 4;
const int LSTM_CHANNEL_NUM = 4;
const int LSTM_XH_CHANNEL_AXIS = 2;
const int LSTM_S_CHANNEL_AXIS = 1;
template <typename Dtype>
void DeQuantLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>* >& bottom,
                                     const vector<Blob<Dtype>* >& top)
{
    if (this->layer_param_.dequant_param().has_layer_type() &&
        (this->layer_param_.dequant_param().layer_type() == "LSTM_X" ||
         this->layer_param_.dequant_param().layer_type() == "LSTM_H" ||
         this->layer_param_.dequant_param().layer_type() == "LSTM_S")) {
        this->blobs_.resize(0);
    } else {
        this->blobs_.resize(BLOB_NUM);
        std::vector<int> shape;
        for (int i = 0; i < BLOB_NUM; i++) {
            if (this->layer_param_.dequant_param().channel_wise() && bottom[0]->num_axes() == NUM_AXES) {
                shape.clear();
                shape.push_back(bottom[0]->channels());
                this->blobs_[i].reset(new Blob<Dtype>(shape));
            } else {
                shape.clear();
                shape.push_back(1);
                this->blobs_[i].reset(new Blob<Dtype>(shape));
            }
        }
    }
}

template <typename Dtype>
bool DeQuantLayer<Dtype>::WhetherShiftN()
{
    vector<int> shape;
    Dtype* shiftN = NULL;
    vector<Dtype> tmpShiftN;
    if (this->blobs_.size() == BLOB_NUM) {
        shape = this->blobs_[shiftBitPosition_]->shape();
        shiftN = this->blobs_[shiftBitPosition_]->mutable_cpu_data();
    } else {
        shape.push_back(this->layer_param_.dequant_param().shift_bit_size());
        for (auto item : this->layer_param_.dequant_param().shift_bit()) {
            tmpShiftN.push_back(item);
        }
        shiftN = &tmpShiftN[0];
    }

    CHECK_EQ(shape.size(), 1);
    unsigned int channelNum = shape[0];
    if (channelNum == 0) {
        return false;
    }
    int zeroNum = 0;
    int nonzeroNUm = 0;
    for (int i = 0; i < channelNum; i++) {
        if (shiftN[i] == 0) {
            zeroNum++;
        } else {
            CHECK_GE(shiftN[i], MIN_SHIFT_BIT);
            CHECK_LE(shiftN[i], MAX_SHIFT_BIT);
            nonzeroNUm++;
        }
    }

    if (nonzeroNUm == channelNum) {
        return true;
    }

    CHECK_EQ(zeroNum, channelNum);
    return false;
}

void CheckDeqParamShape(const vector<int>& shiftNShape,
                        const vector<int>& deqScaleShape,
                        const vector<int>& offsetWShape)
{
    CHECK_EQ(shiftNShape.size(), 1);
    CHECK_EQ(deqScaleShape.size(), 1);
    CHECK_EQ(offsetWShape.size(), 1);

    CHECK_GT(shiftNShape[0], 0);
    CHECK_GT(deqScaleShape[0], 0);
    CHECK_GT(offsetWShape[0], 0);

    CHECK_EQ(shiftNShape[0], deqScaleShape[0]);
    CHECK_EQ(shiftNShape[0], offsetWShape[0]);
}

template <typename Dtype>
void ShiftNForwardCPU(const int count,
                      const DataInfo<Dtype>& dataInfo,
                      const int CHWSize,
                      const int HWSize,
                      const int channelNum)
{
    const Dtype* in = dataInfo.in;
    Dtype* out = dataInfo.out;
    const Dtype* deqScale = dataInfo.deqScale;
    const Dtype* shiftN = dataInfo.shiftN;

    for (int index = 0; index < count; index++) {
        int channelIndex = 0;
        if (channelNum != 1) {
            if (CHWSize == 0 || HWSize == 0) {
                CHECK_GT(CHWSize, 0) << "CHWSize must greater than zero.";
                CHECK_GT(HWSize, 0) << "HWSize must greater than zero.";
                return;
            }
            channelIndex = (index % (CHWSize)) / HWSize;
        }

        Dtype tmp = round((in[index]) / deqScale[channelIndex]);

        tmp = round(tmp / pow(BINARY_BASE, shiftN[channelIndex]));
        if (tmp < INT16_MIN) {
            tmp = INT16_MIN;
        } else if (tmp > INT16_MAX) {
            tmp = INT16_MAX;
        }
        tmp = tmp * deqScale[channelIndex] * pow(BINARY_BASE, shiftN[channelIndex]);

        out[index] = tmp;
    }
}

template <typename Dtype>
void DeQuantLayer<Dtype>::ExtractAndCheckDeqParam(Dtype* &shiftN,
                                                  Dtype* &deqScale,
                                                  vector<int>& deqScaleShape,
                                                  vector<Dtype>& tmpShiftN,
                                                  vector<Dtype>& tmpDeqScale)
{
    vector<int> shiftNShape;
    vector<int> offsetWShape;

    if (this->blobs_.size() == BLOB_NUM) {
        shiftNShape.push_back(this->blobs_[shiftBitPosition_]->shape()[0]);
        shiftN = this->blobs_[shiftBitPosition_]->mutable_cpu_data();

        deqScaleShape.push_back(this->blobs_[deqScalePosition_]->shape()[0]);
        deqScale = this->blobs_[deqScalePosition_]->mutable_cpu_data();

        offsetWShape.push_back(this->blobs_[offsetWPosition_]->shape()[0]);
    } else {
        shiftNShape.push_back(this->layer_param_.dequant_param().shift_bit_size());
        for (auto item : this->layer_param_.dequant_param().shift_bit()) {
            tmpShiftN.push_back(item);
        }
        shiftN = &tmpShiftN[0];

        deqScaleShape.push_back(this->layer_param_.dequant_param().deqscale_size());
        for (auto item : this->layer_param_.dequant_param().deqscale()) {
            tmpDeqScale.push_back(item);
        }
        deqScale = &tmpDeqScale[0];

        offsetWShape.push_back(this->layer_param_.dequant_param().offset_size());
    }
    CheckDeqParamShape(shiftNShape, deqScaleShape, offsetWShape);
}

template <typename Dtype>
void DeQuantLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>* >& bottom,
                                      const vector<Blob<Dtype>* >& top)
{
    const int count = bottom[0]->count();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    int recordNum = bottom[0]->channels() * bottom[0]->height() * bottom[0]->width();
    std::string layerName = this->layer_param_.name();
    ConvertLayerName(layerName, "/", REPLACE_STR);
    std::string fileName = "./amct_log/" + layerName + ".log";

    if (recordData_) {
        INIT_LOG();
        DEBUG_LOG(fileName.c_str(), "[AMCT] bottom data of dequant:\n");
        DATA_DEBUG_LOG(fileName.c_str(), bottom[0]->cpu_data(), recordNum);;
    }

    Dtype* shiftN = NULL;
    Dtype* deqScale = NULL;
    vector<int> deqScaleShape;
    vector<Dtype> tmpShiftN;
    vector<Dtype> tmpDeqScale;
    ExtractAndCheckDeqParam(shiftN, deqScale, deqScaleShape, tmpShiftN, tmpDeqScale);

    if (WhetherShiftN()) {
        struct DataInfo<Dtype> dataInfo;
        dataInfo.in = bottom_data;
        dataInfo.out = top_data;
        dataInfo.deqScale = deqScale;
        dataInfo.shiftN = shiftN;

        int CHWSize = bottom[0]->count(C_INDEX);
        int HWSize = bottom[0]->count(H_INDEX);
        if (this->layer_param_.dequant_param().has_layer_type()) {
            if (this->layer_param_.dequant_param().layer_type() == "LSTM_X" ||
                this->layer_param_.dequant_param().layer_type() == "LSTM_H") {
                CHWSize = bottom[0]->count(LSTM_XH_CHANNEL_AXIS);
                HWSize = bottom[0]->shape(LSTM_XH_CHANNEL_AXIS) / LSTM_CHANNEL_NUM;
            } else if (this->layer_param_.dequant_param().layer_type() == "LSTM_S") {
                CHWSize = bottom[0]->count(LSTM_S_CHANNEL_AXIS);
                HWSize = bottom[0]->shape(LSTM_S_CHANNEL_AXIS) / LSTM_CHANNEL_NUM;
            }
        }

        ShiftNForwardCPU(count, dataInfo, CHWSize, HWSize, deqScaleShape[0]);
    } else {
        caffe_copy(count, bottom_data, top_data);
    }

    if (recordData_) {
        INIT_LOG();
        DEBUG_LOG(fileName.c_str(), "[AMCT] top data of dequant:\n");
        DATA_DEBUG_LOG(fileName.c_str(), top[0]->cpu_data(), recordNum);
        recordData_ = false;
    }
}

template <typename Dtype>
void DeQuantLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>* >& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>* >& bottom)
{
    const int count = bottom[0]->count();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    caffe_copy(count, top_diff, bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(DeQuantLayer);
#endif

INSTANTIATE_CLASS(DeQuantLayer);
REGISTER_LAYER_CLASS(DeQuant);
}  // namespace caffe
