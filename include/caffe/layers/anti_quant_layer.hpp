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
 * @brief anti_quant layer head file
 *
 * @file anti_quant_layer.hpp
 *
 * @version 1.0
 */
#ifndef CAFFE_ANTI_QUANT_LAYER_HPP_
#define CAFFE_ANTI_QUANT_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/neuron_layer.hpp"
#include "quant.h"

namespace caffe {
template <typename Dtype>
class AntiQuantLayer : public NeuronLayer<Dtype> {
public:
    explicit AntiQuantLayer(const LayerParameter& param)
        : NeuronLayer<Dtype>(param) {}

    virtual inline const char* type() const { return "AnitQuant"; }
    virtual ~AntiQuantLayer(){}

protected:
    virtual void Forward_gpu(const vector<Blob<Dtype>* >& bottom,
        const vector<Blob<Dtype>* >& top);
    virtual void Forward_cpu(const vector<Blob<Dtype>* >& bottom,
        const vector<Blob<Dtype>* >& top);
    virtual void Backward_gpu(const vector<Blob<Dtype>* >& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom);
    virtual void Backward_cpu(const vector<Blob<Dtype>* >& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom);

    bool recordData_{true};
};
}  // namespace caffe

#endif  // CAFFE_ANTI_QUANT_LAYER_HPP_
