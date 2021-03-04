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
 * @brief quant layer head file
 *
 * @file quant_layer.hpp
 *
 * @version 1.0
 */
#ifndef CAFFE_QUANT_LAYER_HPP_
#define CAFFE_QUANT_LAYER_HPP_

#include <vector>
#include <cstdlib>
#include <cstdio>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/neuron_layer.hpp"
#include "quant.h"

namespace caffe {
template <typename Dtype>
class QuantLayer : public NeuronLayer<Dtype> {
public:
    explicit QuantLayer(const LayerParameter& param)
        : NeuronLayer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>* >& bottom,
        const vector<Blob<Dtype>* >& top);
    virtual inline const char*  type() const { return "QUANT"; }
    virtual ~QuantLayer(){}

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>* >& bottom,
        const vector<Blob<Dtype>* >& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>* >& bottom,
        const vector<Blob<Dtype>* >& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>* >& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>* >& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom);

private:
    IfmrParam ifmrParam_;
    FloatData scale_;
    IntData offset_;
    float scaleData_{0};
    int offsetData_{0};
    unsigned int numBatchStored_{0};
    bool recordData_{true};
    std::vector<Dtype> storedDataForCalibration_;
};
}  // namespace caffe

#endif  // CAFFE_QUANT_LAYER_HPP_
