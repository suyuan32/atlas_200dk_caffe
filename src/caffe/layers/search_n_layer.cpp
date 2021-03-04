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
 * @brief search shift bit layer cpp source
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
#include "caffe/util/io.hpp"
#include "caffe/util/amct_util.hpp"

namespace caffe {
const int LSTM_CHANNEL_NUM = 4;
const int LSTM_XH_CHANNEL_AXIS = 2;
const int LSTM_S_CHANNEL_AXIS = 1;
const int CONV_HEIGHT_AXIS = 2;
const int CONV_WIDTH_AXIS = 3;
template <typename Dtype>
void SearchNLayer<Dtype>::Reshape(const vector<Blob<Dtype>* >& bottom,
                                  const vector<Blob<Dtype>* >& top)
{
    if (top.size() == 1) {
        top[0]->ReshapeLike(*bottom[0]);
    }
    return;
}

template <typename Dtype>
void SearchNLayer<Dtype>::ReadRecord(ScaleOffsetRecord* records)
{
    if (!ReadProtoFromTextFile(recordFileName.c_str(), records)) {
        LOG(ERROR) << "Read records from " << recordFileName << " failed.";
    }
    return;
}

template <typename Dtype>
void SearchNLayer<Dtype>::GetBottomChannels()
{
    ScaleOffsetRecord records;
    ReadRecord(&records);

    for (auto layerName : quantLayerNames) {
        bool found = false;
        for (int i = 0; i < records.record_size(); i++) {
            const ScaleOffsetRecord_MapFiledEntry& record = records.record(i);
            if (record.has_key() && record.key() == layerName) {
                found = true;
                const SingleLayerRecord& layerQuantInfo = record.value();
                int channel = layerQuantInfo.scale_w_size();
                CHECK_GE(channel, 1);
                bottomChannels.push_back(channel);
            }
        }
        CHECK(found) << "Layer " << layerName << " not found in " << recordFileName;
    }

    for (int i = 1; i < bottomChannels.size(); i++) {
        CHECK_EQ(bottomChannels[i], bottomChannels[0]) << "Channel number of layer " <<
            quantLayerNames.Get(i) << "(" << bottomChannels[i] << ")" <<
            " is different from layer " << quantLayerNames.Get(0) << "(" << bottomChannels[0] << ").";
    }

    for (int i = 0; i < bottomChannels.size(); i++) {
        if (quantLayerTypes.Get(i) == string("InnerProduct")) {
            CHECK_EQ(bottomChannels[i], 1) << "The channel number of InnerProduct layer(" << quantLayerNames.Get(i) <<
                ") should be 1, but actually is " << bottomChannels[i];
        }
    }
}

template <typename Dtype>
void SearchNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>* >& bottom,
                                     const vector<Blob<Dtype>* >& top)
{
    targetBatchNum = this->layer_param_.search_n_param().batch_num();
    quantLayerNames = this->layer_param_.search_n_param().layer_name();
    CHECK_GE(quantLayerNames.size(), 1);

    showQuantLayerNames = "[";
    for (auto layerName : quantLayerNames) {
        showQuantLayerNames += layerName;
        showQuantLayerNames += " ";
    }
    showQuantLayerNames[showQuantLayerNames.size() - 1] = ']';

    quantLayerTypes = this->layer_param_.search_n_param().layer_type();
    CHECK_EQ(quantLayerNames.size(), quantLayerTypes.size()) << "layer_name size(" << quantLayerNames.size() <<
        ") must be equal to layer_type size(" << quantLayerTypes.size() << ").";

    recordFileName = this->layer_param_.search_n_param().record_file_path();
    GetBottomChannels();
    channelNum = bottomChannels[0];

    for (int index = 0; index < quantLayerTypes.size(); ++index) {
        if (quantLayerTypes.Get(index) == "Convolution" && channelNum > 1) {
            if (bottom[index]->shape(CONV_HEIGHT_AXIS) == 1 && bottom[index]->shape(CONV_WIDTH_AXIS) == 1) {
                globalConvFlag_ = true;
                channelNum = 1;
                LOG(INFO) << "Find layer \"" << quantLayerNames.Get(index) << "\" is global conv";
            }
        }
    }
    storedData.resize(channelNum);
}

template <typename Dtype>
void SearchNLayer<Dtype>::AccumulateData(const vector<Blob<Dtype>* >& bottom)
{
    for (int i = 0; i < bottom.size(); i++) {
        CHECK_GE(bottom[i]->shape().size(), MIN_SHAPE_SIZE);
        if (quantLayerTypes.Get(0) != "LSTM_X" &&
            quantLayerTypes.Get(0) != "LSTM_H" &&
            quantLayerTypes.Get(0) != "LSTM_S") {
            if (channelNum != 1) {
                CHECK_EQ(bottom[i]->shape()[1], channelNum) << "Channel wise but scale size: " \
                    << channelNum << " != channel in: " << bottom[i]->shape()[1];
            }
        } else {
            CHECK_EQ(channelNum, LSTM_CHANNEL_NUM);
        }
    }

    for (int i = 0; i < bottom.size(); i++) {
        Blob<Dtype>* curBottom = bottom[i];
        int count = curBottom->count();
        const Dtype* bottomData = curBottom->cpu_data();
        unsigned int channelID = 0;
        unsigned int CHW = curBottom->count(C_INDEX);
        unsigned int HW = curBottom->count(H_INDEX);

        if (quantLayerTypes.Get(0) == "LSTM_X" || quantLayerTypes.Get(0) == "LSTM_H") {
            CHW = curBottom->count(LSTM_XH_CHANNEL_AXIS);
            HW = curBottom->shape(LSTM_XH_CHANNEL_AXIS) / LSTM_CHANNEL_NUM;
        } else if (quantLayerTypes.Get(0) == "LSTM_S") {
            CHW = curBottom->count(LSTM_S_CHANNEL_AXIS);
            HW = curBottom->shape(LSTM_S_CHANNEL_AXIS) / LSTM_CHANNEL_NUM;
        }
        if (channelNum > 1) {
            for (int offset = 0; offset < count; offset++) {
                channelID = (offset % CHW) / HW;
                storedData[channelID].push_back(bottomData[offset]);
            }
        } else {
            for (int offset = 0; offset < count; offset++) {
                storedData[channelID].push_back(bottomData[offset]);
            }
        }
    }

    return;
}

template <typename Dtype>
void SearchNLayer<Dtype>::GetDeqScale()
{
    ScaleOffsetRecord records;
    ReadRecord(&records);
    Dtype scaleD = 0;
    vector<vector<Dtype>> allDeqScales(quantLayerNames.size());
    for (int i = 0; i < quantLayerNames.size(); i++) {
        string layerName(quantLayerNames.Get(i));
        for (int j = 0; j < records.record_size(); j++) {
            const ScaleOffsetRecord_MapFiledEntry& record = records.record(j);
            if (record.has_key() && record.key() == layerName) {
                const SingleLayerRecord& layerQuantInfo = record.value();
                int scaleWSize = layerQuantInfo.scale_w_size();
                CHECK_EQ(layerQuantInfo.has_scale_d(), true);
                scaleD = layerQuantInfo.scale_d();
                for (int k = 0; k < scaleWSize; k++) {
                    allDeqScales[i].push_back(layerQuantInfo.scale_d() * layerQuantInfo.scale_w(k));
                }
            }
        }
    }

    for (int i = 1; i < allDeqScales.size(); i++) {
        CHECK_EQ(allDeqScales[i].size(), allDeqScales[0].size());
        for (int j = 0; j < allDeqScales[i].size(); j++) {
            CHECK_EQ(allDeqScales[i][j], allDeqScales[0][j]);
        }
    }

    for (int i = 0; i < allDeqScales[0].size(); i++) {
        deqScale.push_back(allDeqScales[0][i]);
    }

    if (globalConvFlag_) {
        Dtype maxDeqScale = 0;
        for (auto singleDeqScale : deqScale) {
            if (singleDeqScale > maxDeqScale) {
                maxDeqScale = singleDeqScale;
            }
        }
        for (int i = 0; i < deqScale.size(); i++) {
            deqScale[i] = maxDeqScale;
        }
        UpdateScaleW(deqScale, scaleD);
    }
    return;
}

template <typename Dtype>
void SearchNLayer<Dtype>::RecordN(vector<int>& bestN)
{
    ScaleOffsetRecord records;
    ReadRecord(&records);

    if (globalConvFlag_) {
        CHECK_EQ(bestN.size(), 1) << "Do global conv search_n, searched n should 1, actual " << bestN.size();
        for (int i = 1; i < bottomChannels[0]; ++i) {
            bestN.push_back(bestN[0]);
        }
    }

    for (int i = 0; i < quantLayerNames.size(); i++) {
        string layerName(quantLayerNames.Get(i));
        for (int j = 0; j < records.record_size(); j++) {
            ScaleOffsetRecord_MapFiledEntry* record = records.mutable_record(j);
            if (record->has_key() && record->key() == layerName) {
                SingleLayerRecord* layerQuantInfo = record->mutable_value();
                for (int k = 0; k < bestN.size(); k++) {
                    layerQuantInfo->add_shift_bit(bestN[k]);
                }
            }
        }
    }

    WriteProtoToTextFile(records, recordFileName.c_str());
    return;
}

template <typename Dtype>
void SearchNLayer<Dtype>::UpdateScaleW(vector<Dtype>& deqScale, Dtype scaleD)
{
    ScaleOffsetRecord records;
    ReadRecord(&records);

    for (int i = 0; i < quantLayerNames.size(); i++) {
        string layerName(quantLayerNames.Get(i));
        for (int j = 0; j < records.record_size(); j++) {
            ScaleOffsetRecord_MapFiledEntry* record = records.mutable_record(j);
            if (record->has_key() && record->key() == layerName) {
                SingleLayerRecord* layerQuantInfo = record->mutable_value();
                for (int k = 0; k < deqScale.size(); k++) {
                    layerQuantInfo->set_scale_w(k, deqScale[k] / scaleD);
                }
            }
        }
    }

    WriteProtoToTextFile(records, recordFileName.c_str());
    return;
}

template <typename Dtype>
void SearchNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>* >& bottom,
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

        vector<int> bestN;
        SearchShitBit(storedData, deqScale, bestN);
        RecordN(bestN);
        LOG(INFO) << "Do layer: " << showQuantLayerNames << " search shift bits calibration success!";

        storedData.clear();
        storedData.shrink_to_fit();
        curBatchNum++;
    }

    if (top.size() == 1) {
        const int count = bottom[0]->count();
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        caffe_copy(count, bottom_data, top_data);
    }
}

template <typename Dtype>
void SearchNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>* >& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom)
{
    return;
}

#ifdef CPU_ONLY
STUB_GPU(SearchNLayer);
#endif

INSTANTIATE_CLASS(SearchNLayer);
REGISTER_LAYER_CLASS(SearchN);
}  // namespace caffe
