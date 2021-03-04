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
 * @brief weight quantization aware training layer head file
 *
 * @file weight_qat_layer.hpp
 *
 * @version 1.0
 */
#ifndef CAFFE_WEIGHT_QAT_LAYER_HPP_
#define CAFFE_WEIGHT_QAT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "quant.h"

using Status = unsigned int;

namespace caffe {
template <typename Dtype>
class WeightQATLayer : public Layer<Dtype> {
public:
    explicit WeightQATLayer(const LayerParameter& param): Layer<Dtype>(param) {}
    ~WeightQATLayer() {}
    virtual void LayerSetUp(const vector<Blob<Dtype>* >& bottom,
                            const vector<Blob<Dtype>* >& top)
    {
        top[0]->ReshapeLike(*bottom[0]);
        RetrainWeightQuantParameter retrain_weight_quant_param =
            this->layer_param_.retrain_weight_quant_param();
        const int weight_shape_size = 2;
        vector<int> weight_shape(weight_shape_size);
        weight_shape[0] = retrain_weight_quant_param.cout();
        weight_shape[1] = retrain_weight_quant_param.cin();
        if (retrain_weight_quant_param.has_h() &&
            retrain_weight_quant_param.has_w()) {
            weight_shape.push_back(retrain_weight_quant_param.h());
            weight_shape.push_back(retrain_weight_quant_param.w());
        }
        this->blobs_.resize(1);
        this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
        shared_ptr<Blob<Dtype>> blob_pointer(new Blob<Dtype>());
        blob_pointer->CopyFrom(*this->blobs_[0], false, true);
        this->blobs_.push_back(blob_pointer);
        is_copy_ = false;
        layer_type_ =
            this->layer_param_.retrain_weight_quant_param().layer_type();
        arq_param_.numBits = NUM_BITS_QUANT;
        arq_param_.channelWise = retrain_weight_quant_param.channel_wise();
        arq_param_.withOffset = retrain_weight_quant_param.with_offset();
    }
    virtual void Reshape(const vector<Blob<Dtype>* >& bottom,
                         const vector<Blob<Dtype>* >& top) {}
    virtual inline const char* type() const { return "RetrainWeightQuant"; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>* >& bottom,
                             const vector<Blob<Dtype>* >& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>* >& bottom,
                             const vector<Blob<Dtype>* >& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>* >& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>* >& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>* >& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>* >& bottom);

    bool is_copy_;
    string layer_type_;
    ArqParam arq_param_;
    FloatData scale_;
    IntData offset_;
    vector<float> scale_data_;
    vector<int> offset_data_;
};
}  // namespace caffe

#endif  // CAFFE_WEIGHT_QAT_LAYER_HPP_
