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
 * @brief ifmr layer head file
 *
 * @file ifmr_layer.hpp
 *
 * @version 1.0
 */
#ifndef CAFFE_IFMR_LAYER_HPP_
#define CAFFE_IFMR_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/neuron_layer.hpp"
#include "quant.h"

namespace caffe {
template <typename Dtype>
class IFMRLayer : public NeuronLayer<Dtype> {
public:

    explicit IFMRLayer(const LayerParameter& param)
        : NeuronLayer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>* >& bottom, const vector<Blob<Dtype>* >& top);
    virtual void Reshape(const vector<Blob<Dtype>* >& bottom,
        const vector<Blob<Dtype>* >& top) {}
    virtual inline const char* type() const { return "IFMR"; }
    virtual inline int ExactNumBottomBlobs() const { return -1; }
    virtual inline int MinBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 0; }
    int RecordScaleOffset(std::string file_name,
                          std::string layer_name,
                          FloatData scale,
                          IntData offset,
                          unsigned int channels,
                          unsigned int height,
                          unsigned int width);
    virtual ~IFMRLayer(){}

protected:

    virtual void Forward_cpu(const vector<Blob<Dtype>* >& bottom,
        const vector<Blob<Dtype>* >& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>* >& bottom,
        const vector<Blob<Dtype>* >& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>* >& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom)
    {
        for (int i = 0; i < propagate_down.size(); ++i) {
            if (propagate_down[i]) {
                NOT_IMPLEMENTED;
            }
        }
    }
    virtual void Backward_gpu(const vector<Blob<Dtype>* >& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom);

private:
    FloatData scale_;
    float scaleData_{0};
    IntData offset_;
    int offsetData_{0};
    IfmrParam ifmrParam_;
    unsigned int numBatchStored_{0};
    std::vector<Dtype> storedDataForCalibration_;
    google::protobuf::RepeatedPtrField<std::string> objectLayerNames_;
    unsigned int targetBatchNum_;
};
}  // namespace caffe

#endif  // CAFFE_IFMR_LAYER_HPP_
