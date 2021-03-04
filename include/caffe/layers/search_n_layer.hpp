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
 * @brief search n layer head file
 *
 * @version 1.0
 */
#ifndef CAFFE_SEARCH_N_LAYER_HPP_
#define CAFFE_SEARCH_N_LAYER_HPP_

#include <vector>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/neuron_layer.hpp"

#include "quant.h"

namespace caffe {
template <typename Dtype>
class SearchNLayer : public NeuronLayer<Dtype> {
public:
    explicit SearchNLayer(const LayerParameter& param)
        : NeuronLayer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>* >& bottom,
        const vector<Blob<Dtype>* >& top);
    virtual void Reshape(const vector<Blob<Dtype>* >& bottom,
        const vector<Blob<Dtype>* >& top);
    virtual inline const char* type() const { return "SearchN"; }

    virtual inline int MinBottomBlobs() const { return 1; }
    virtual inline int ExactNumBottomBlobs() const { return -1; }

    virtual inline int MinTopBlobs() const { return 0; }
    virtual inline int MaxTopBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return -1; }

    void GetBottomChannels();
    void AccumulateData(const vector<Blob<Dtype>* >& bottom);
    void GetDeqScale();
    void RecordN(vector<int>& bestN);
    void UpdateScaleW(vector<Dtype>& deqScale, Dtype scaleD);
    void ReadRecord(ScaleOffsetRecord* records);
    virtual ~SearchNLayer() {}

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>* >& bottom,
        const vector<Blob<Dtype>* >& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>* >& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom);
    virtual void Forward_gpu(const vector<Blob<Dtype>* >& bottom,
        const vector<Blob<Dtype>* >& top);
    virtual void Backward_gpu(const vector<Blob<Dtype>* >& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom);
private:
    vector<vector<Dtype>> storedData;
    bool globalConvFlag_ = false;
    unsigned int targetBatchNum = 0;
    unsigned int curBatchNum = 0;
    unsigned int channelNum = 0;
    google::protobuf::RepeatedPtrField<std::string> quantLayerNames;
    string showQuantLayerNames;
    google::protobuf::RepeatedPtrField<std::string> quantLayerTypes;
    vector<int> bottomChannels;
    string recordFileName;
    vector<Dtype> deqScale;
};

const int C_INDEX = 1;
const int H_INDEX = 2;
const int MIN_SHAPE_SIZE = 2;
}  // namespace caffe

#endif  // CAFFE_SEARCH_N_LAYER_HPP_
