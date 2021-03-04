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
 * @brief dequant layer head file
 *
 * @file dequant_layer.hpp
 *
 * @version 1.0
 */
#ifndef CAFFE_DEQUANT_LAYER_HPP_
#define CAFFE_DEQUANT_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/neuron_layer.hpp"
#include "quant.h"

namespace caffe {
template <typename Dtype>
class DeQuantLayer : public NeuronLayer<Dtype> {
public:
    explicit DeQuantLayer(const LayerParameter& param)
        : NeuronLayer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>* >& bottom,
        const vector<Blob<Dtype>* >& top);

    virtual inline const char* type() const { return "DEQUANT"; }
    virtual ~DeQuantLayer() {}
    bool WhetherShiftN();
    void ExtractAndCheckDeqParam(Dtype* &shiftN,
                                 Dtype* &deqScale,
                                 vector<int>& deqScaleShape,
                                 vector<Dtype>& tmpShiftN,
                                 vector<Dtype>& tmpDeqScale);
protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>* >& bottom,
        const vector<Blob<Dtype>* >& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>* >& bottom,
        const vector<Blob<Dtype>* >& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>* >& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>* >& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom);

    bool recordData_{true};
    const unsigned int offsetWPosition_ = 2;
    const unsigned int shiftBitPosition_ = 1;
    const unsigned int deqScalePosition_ = 0;
};

template <typename Dtype>
struct DataInfo {
    const Dtype* in;
    Dtype* out;
    const Dtype* deqScale;
    const Dtype* shiftN;
};

void CheckDeqParamShape(const vector<int>& shiftNShape,
                        const vector<int>& deqScaleShape,
                        const vector<int>& offsetWShape);

const int BLOB_NUM = 3; // the number of blobs used to store params

const int C_INDEX = 1;
const int H_INDEX = 2;
}  // namespace caffe

#endif  // CAFFE_DEQUANT_LAYER_HPP_
