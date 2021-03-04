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
 * @brief dequant layer cuda src file
 *
 * @file dequant_layer.cu
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

template <typename Dtype>
__global__ void ShiftNForwardGPU(const int count,
                                 const struct DataInfo<Dtype>* dataInfo,
                                 const int CHWSize,
                                 const int HWSize,
                                 const int channelNum)
{
    const Dtype* in = dataInfo->in;
    Dtype* out = dataInfo->out;
    const Dtype* deqScale = dataInfo->deqScale;
    const Dtype* shiftN = dataInfo->shiftN;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        int channelIndex = 0;
        if (channelNum != 1) {
            if (CHWSize == 0 || HWSize == 0) {
                LOG_ERROR("CHWSize or HWSize must greater than zero.\n");
                return;
            }
            channelIndex = (tid % (CHWSize)) / HWSize;
        }
        Dtype tmp = round((in[tid]) / deqScale[channelIndex]);
        tmp = round(tmp / pow(BINARY_BASE, shiftN[channelIndex]));
        if (tmp < INT16_MIN) {
            tmp = INT16_MIN;
        } else if (tmp > INT16_MAX) {
            tmp = INT16_MAX;
        }
        tmp = tmp * deqScale[channelIndex] * pow(BINARY_BASE, shiftN[channelIndex]);
        out[tid] = tmp;
    }
}

template <typename Dtype>
void DeQuantLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>* >& bottom,
                                      const vector<Blob<Dtype>* >& top)
{
    const int count = bottom[0]->count();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* mutable_top_data = top[0]->mutable_gpu_data();

    if (recordData_) {
        int recordNum = bottom[0]->channels() * bottom[0]->height() * bottom[0]->width();
        std::string layerName = this->layer_param_.name();
        ConvertLayerName(layerName, "/", REPLACE_STR);
        std::string fileName = "./amct_log/" + layerName + ".log";
        INIT_LOG();
        DEBUG_LOG(fileName.c_str(), "[AMCT] bottom data of dequant:\n");
        DATA_DEBUG_LOG(fileName.c_str(), bottom[0]->cpu_data(), recordNum);
    }

    vector<int> shiftNShape;
    Dtype* shiftN = NULL;
    vector<int> deqScaleShape;
    Dtype* deqScale = NULL;
    vector<int> offsetWShape;

    vector<Dtype> tmpShiftN;
    vector<Dtype> tmpDeqScale;

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
        cudaMalloc(&shiftN, sizeof(Dtype) * this->layer_param_.dequant_param().shift_bit_size());
        cudaMemcpy(shiftN, &tmpShiftN[0],
                   sizeof(Dtype) * this->layer_param_.dequant_param().shift_bit_size(), cudaMemcpyHostToDevice);

        deqScaleShape.push_back(this->layer_param_.dequant_param().deqscale_size());
        for (auto item : this->layer_param_.dequant_param().deqscale()) {
            tmpDeqScale.push_back(item);
        }
        cudaMalloc(&deqScale, sizeof(Dtype) * this->layer_param_.dequant_param().deqscale_size());
        cudaMemcpy(deqScale, &tmpDeqScale[0],
                   sizeof(Dtype) * this->layer_param_.dequant_param().deqscale_size(), cudaMemcpyHostToDevice);

        offsetWShape.push_back(this->layer_param_.dequant_param().offset_size());
    }
    CheckDeqParamShape(shiftNShape, deqScaleShape, offsetWShape);

    if (WhetherShiftN()) {
        struct DataInfo<Dtype> dataInfo;
        dataInfo.in = bottom_data;
        dataInfo.out = mutable_top_data;
        dataInfo.deqScale = deqScale;
        dataInfo.shiftN = shiftN;

        struct DataInfo<Dtype>* dataInfoDevice = NULL;
        cudaError_t cudaStatus = cudaMalloc(&dataInfoDevice, sizeof(struct DataInfo<Dtype>));
        if (cudaStatus != cudaSuccess) {
            std::cerr << "CUDA CALL FAILED:" << __func__ << "( " << __LINE__ << ")- " << cudaGetErrorString(cudaStatus) << std::endl;
            return;
        }
        cudaMemcpy(dataInfoDevice, &dataInfo, sizeof(struct DataInfo<Dtype>), cudaMemcpyHostToDevice);

        int CHWSize = 0;
        int HWSize = 0;
        if (this->layer_param_.dequant_param().has_layer_type() &&
            (this->layer_param_.dequant_param().layer_type() == "LSTM_X" ||
             this->layer_param_.dequant_param().layer_type() == "LSTM_H" ||
             this->layer_param_.dequant_param().layer_type() == "LSTM_S")) {
            if (this->layer_param_.dequant_param().layer_type() == "LSTM_X" ||
                this->layer_param_.dequant_param().layer_type() == "LSTM_H") {
                CHWSize = bottom[0]->count(2);
                HWSize = bottom[0]->shape(2) / 4;
            } else if (this->layer_param_.dequant_param().layer_type() == "LSTM_S") {
                CHWSize = bottom[0]->count(1);
                HWSize = bottom[0]->shape(1) / 4;
            }
        } else {
            CHWSize = bottom[0]->count(C_INDEX);
            HWSize = bottom[0]->count(H_INDEX);
        }
        ShiftNForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, dataInfoDevice, CHWSize, HWSize, deqScaleShape[0]);
        cudaFree(dataInfoDevice);
        CUDA_POST_KERNEL_CHECK;
    } else {
        caffe_copy(count, bottom_data, mutable_top_data);
    }

    if (recordData_) {
        int recordNum = bottom[0]->channels() * bottom[0]->height() * bottom[0]->width();
        std::string layerName = this->layer_param_.name();
        ConvertLayerName(layerName, "/", REPLACE_STR);
        std::string fileName = "./amct_log/" + layerName + ".log";
        INIT_LOG();
        DEBUG_LOG(fileName.c_str(), "[AMCT] top data of dequant:\n");
        DATA_DEBUG_LOG(fileName.c_str(), top[0]->cpu_data(), recordNum);
        recordData_ = false;
    }
}

template <typename Dtype>
void DeQuantLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>* >& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>* >& bottom)
{
    const int count = bottom[0]->count();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    caffe_copy(count, top_diff, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(DeQuantLayer);
}  // namespace caffe
