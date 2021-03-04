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
 * @brief activation quantization aware training layer head file
 *
 * @file activation_qat_layer.hpp
 *
 * @version 1.0
 */
#ifndef CAFFE_RETRAIN_QUANT_LAYER_HPP_
#define CAFFE_RETRAIN_QUANT_LAYER_HPP_

#include <vector>
#include <mutex>
#include <fstream>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "quant.h"

using Status = unsigned int;

namespace caffe {
static std::mutex mtx_;
template <typename Dtype>
class ActivationQATLayer : public Layer<Dtype> {
public:
    explicit ActivationQATLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}
    ~ActivationQATLayer() {}
    virtual void LayerSetUp(const vector<Blob<Dtype>* >& bottom,
                            const vector<Blob<Dtype>* >& top)
    {
        top[0]->ReshapeLike(*bottom[0]);
        vector<int> shape(1);
        shape[0] = 1;
        const int blob_size = 2;
        this->blobs_.resize(blob_size);
        this->blobs_[0].reset(new Blob<Dtype>(shape));
        this->blobs_[1].reset(new Blob<Dtype>(shape));
        Dtype* clip_max = this->blobs_[0]->mutable_cpu_data();
        clip_max[0] = this->layer_param_.retrain_data_quant_param().clip_max();
        clip_max_pre_ = clip_max[0];
        Dtype* clip_min = this->blobs_[1]->mutable_cpu_data();
        clip_min[0] = this->layer_param_.retrain_data_quant_param().clip_min();
        clip_min_pre_ = clip_min[0];
        ifmr_init_ = this->layer_param_.retrain_data_quant_param().ifmr_init();
        fixed_min_ = this->layer_param_.retrain_data_quant_param().fixed_min();
        object_layer_ =
            this->layer_param_.retrain_data_quant_param().object_layer();
        record_flag_ = false;
        if (ifmr_init_) {
            ifmrParam_.calibration = 0;
            ifmrParam_.numBits = NUM_BITS_QUANT;
            ifmrParam_.withOffset = true;
            const float start_ratio = 0.7;
            ifmrParam_.startRatio = start_ratio;
            const float end_ratio = 1.3;
            ifmrParam_.endRatio = end_ratio;
            const float step = 0.01;
            ifmrParam_.step = step;
            const float max_percentile = 0.999999;
            ifmrParam_.maxPercentile = max_percentile;
            const float min_percentile = 0.999999;
            ifmrParam_.minPercentile = min_percentile;
        }
        scale_.length = 1;
        scale_.data = &scale_data_;
        offset_.length = 1;
        offset_.data = &offset_data_;
        mtx_.lock();
        string record_file = this->layer_param_.retrain_data_quant_param().record_file_path();
        std::ofstream outfile(record_file.c_str(), ios::out | ios::trunc);
        outfile.close();
        mtx_.unlock();
    };
    virtual void Reshape(const vector<Blob<Dtype>* >& bottom,
        const vector<Blob<Dtype>* >& top) {}
    virtual inline const char* type() const { return "RetrainDataQuant"; }
    Status RecordRetrainShape(string file_name, string layer_name,
        unsigned int channels, unsigned int height, unsigned int width)
    {
        mtx_.lock();
        RetrainShapeRecord records;
        if (!ReadProtoFromTextFile(file_name.c_str(), &records)) {
            LOG(ERROR) << "Read records from " << file_name << " failed.";
            return GENERIC_ERROR;
        }
        bool found_layer = false;
        for (int i = 0; i < records.record_size(); i++) {
            RetrainShapeRecord_MapFiledEntry* record =
                records.mutable_record(i);
            if (record->has_key() && record->key() == layer_name) {
                RetrainSingleLayerRecord* single_layer_record =
                    record->mutable_value();
                single_layer_record->set_channels(channels);
                single_layer_record->set_height(height);
                single_layer_record->set_width(width);
                found_layer = true;
                break;
            }
        }
        if (!found_layer) {
            RetrainShapeRecord_MapFiledEntry* record = records.add_record();
            record->set_key(layer_name);
            RetrainSingleLayerRecord* single_layer_record =
                record->mutable_value();
            single_layer_record->set_channels(channels);
            single_layer_record->set_height(height);
            single_layer_record->set_width(width);
        }
        WriteProtoToTextFile(records, file_name.c_str());
        mtx_.unlock();
        return SUCCESS;
    }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>* >& bottom,
                             const vector<Blob<Dtype>* >& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>* >& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>* >& bottom);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>* >& top);
    virtual void Backward_gpu(const vector<Blob<Dtype>* >& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>* >& bottom);

    bool ifmr_init_;
    bool fixed_min_;
    Dtype clip_max_pre_;
    Dtype clip_min_pre_;
    IfmrParam ifmrParam_;
    FloatData scale_;
    IntData offset_;
    float scale_data_;
    int offset_data_;
    vector<Dtype> stored_data_for_calibration_;
    bool record_flag_;
    string object_layer_;
};
}  // namespace caffe

#endif  // CAFFE_ACTIVATION_QAT_LAYER_HPP_
