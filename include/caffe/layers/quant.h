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
 * @brief quant head file
 *
 * @file quant.h
 *
 * @version 1.0
 */

#ifndef QUANT_H
#define QUANT_H
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
/**
 * @ingroup quantize lib
 * @brief: error code.
 */
const unsigned int SUCCESS = 0x00000000;
const unsigned int GENERIC_ERROR = 0xFFFF0000;
const unsigned int BAD_FORMAT_ERROR = 0xFFFF0005;
const unsigned int BAD_PARAMETERS_ERROR = 0xFFFF0006;
const unsigned int OUT_OF_MEMORY_ERROR = 0xFFFF000C;
const unsigned int SHORT_BUFFER_ERROR = 0xFFFF0010;
const unsigned int NOT_SUPPORT_ERROR = 0xFFFF0011;
const unsigned int CUDA_ERROR = 0xFFFF0012;

const unsigned int NUM_BITS_QUANT = 8;
const unsigned int BINARY_BASE = 2;

const unsigned int MIN_SHIFT_BIT = 1;
const unsigned int MAX_SHIFT_BIT = 16;

using Status = unsigned int;

#define INIT_LOG() \
        std::map<std::string, int> mapLog; \
        mapLog.insert(std::pair<std::string, int>("ERROR", 1)); \
        mapLog.insert(std::pair<std::string, int>("WARNING", 2)); \
        mapLog.insert(std::pair<std::string, int>("INFO", 3)); \
        mapLog.insert(std::pair<std::string, int>("DEBUG", 4)); \
        time_t t = time(0); \
        char tmpDate[32] = {}; \
        strftime(tmpDate, sizeof(tmpDate), "%Y-%m-%d %H:%M:%S", localtime(&t)); \
        bool existFlag = false; \
        std::string pathvar; \
        char* amctenv = NULL; \
        amctenv = getenv("AMCT_LOG_FILE_LEVEL"); \
        if (amctenv == NULL) { \
            pathvar = "INFO"; \
        } else { \
            pathvar = amctenv; \
            transform(pathvar.begin(), pathvar.end(), pathvar.begin(), ::toupper); \
        } \
        for (std::map<std::string, int>::iterator itr = mapLog.begin(); itr !=mapLog.end(); itr++) { \
            if(pathvar == itr->first) { \
                existFlag = true; \
                    break; \
                } \
        } \
        if (!existFlag) { \
            pathvar = "INFO"; \
        } \

#define DATA_ERROR_LOG(logFilePath, inputArray, length) \
        if (mapLog[pathvar] >= mapLog["ERROR"]) { \
            std::ofstream pfLog; \
            pfLog.open(logFilePath, std::ios::app); \
            if (pfLog) { \
                pfLog << tmpDate << "[ERROR][" << __FILE__ << "][" << __func__ << "][" << __LINE__ << "]\n"; \
                for (int i = 0; i < length; i++) { \
                    pfLog << inputArray[i] << " "; \
                } \
                pfLog << "\n"; \
                pfLog.close(); \
            } \
        }

#define DATA_WARNING_LOG(logFilePath, inputArray, length) \
        if (mapLog[pathvar] >= mapLog["WARNING"]) { \
            std::ofstream pfLog; \
            pfLog.open(logFilePath, std::ios::app); \
            if (pfLog) { \
                pfLog << tmpDate<<"[WARNING][" << __FILE__ << "][" <<__func__ << "]["<<__LINE__<< "]\n"; \
                for (int i = 0; i < length; i++) { \
                    pfLog<<inputArray[i] << " "; \
                } \
                pfLog<<"\n"; \
                pfLog.close(); \
            } \
        }

#define DATA_INFO_LOG(logFilePath, inputArray, length) \
        if (mapLog[pathvar] >= mapLog["INFO"]) { \
            std::ofstream pfLog; \
            pfLog.open(logFilePath, std::ios::app); \
            if (pfLog) { \
                pfLog << tmpDate << "[INFO][" << __FILE__ << "][" << __func__ << "][" << __LINE__ << "]\n"; \
                for (int i = 0; i < length; i++) { \
                    pfLog << inputArray[i] << " "; \
                } \
                pfLog << "\n"; \
                pfLog.close(); \
            } \
        }

#define DATA_DEBUG_LOG(logFilePath, inputArray, length) \
        if (mapLog[pathvar] >= mapLog["DEBUG"]) { \
            std::ofstream pfLog; \
            pfLog.open(logFilePath, std::ios::app); \
            if (pfLog) { \
                pfLog << tmpDate << "[DEBUG][" << __FILE__ << "][" << __func__ << "][" << __LINE__ << "]\n"; \
                for (int i = 0; i < length; i++) { \
                    pfLog<<inputArray[i]<<" "; \
                } \
                pfLog<<"\n"; \
                pfLog.close(); \
            } \
        }

#define ERROR_LOG(logFilePath, inputArray) \
        if (mapLog[pathvar] >= mapLog["ERROR"]) { \
            std::ofstream pfLog; \
            pfLog.open(logFilePath, std::ios::app); \
            if (pfLog) { \
                pfLog << tmpDate << "[ERROR][" << __FILE__ << "][" << __func__ << "][" << __LINE__ << "]\n"; \
                pfLog << inputArray << "\n"; \
                pfLog.close(); \
            } \
        }

#define WARNING_LOG(logFilePath, inputArray) \
        if (mapLog[pathvar] >= mapLog["WARNING"]) { \
            std::ofstream pfLog; \
            pfLog.open(logFilePath, std::ios::app); \
            if (pfLog) { \
                pfLog << tmpDate << "[WARNING][" << __FILE__ << "][" << __func__ << "][" << __LINE__ << "]\n"; \
                pfLog << inputArray << "\n"; \
                pfLog.close(); \
            } \
        }

#define INFO_LOG(logFilePath, inputArray) \
        if (mapLog[pathvar] >= mapLog["INFO"]) { \
            std::ofstream pfLog; \
            pfLog.open(logFilePath, std::ios::app); \
            if (pfLog) { \
                pfLog<<tmpDate<<"[INFO]["<<__FILE__<<"]["<<__func__<<"]["<<__LINE__<<"]\n"; \
                pfLog<<inputArray<<"\n"; \
                pfLog.close(); \
            } \
        }

#define DEBUG_LOG(logFilePath, inputArray) \
        if (mapLog[pathvar] >= mapLog["DEBUG"]) { \
            std::ofstream pfLog; \
            pfLog.open(logFilePath, std::ios::app); \
            if (pfLog) { \
                pfLog<<tmpDate<<"[DEBUG]["<<__FILE__<<"]["<<__func__<<"]["<<__LINE__<<"]\n"; \
                pfLog<<inputArray<<"\n"; \
                pfLog.close(); \
            } \
        }

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
        i < (n); \
        i += blockDim.x * gridDim.x)

// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N)
{
    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

/**
 * @ingroup quantize lib
 * @brief: float data array and length.
 */
struct FloatData {
    unsigned int length;
    float* data;
};

/**
 * @ingroup quantize lib
 * @breif: clip index and quant bits.
 */
struct ClipInfo {
    unsigned int index;
    unsigned int QUANT_BITS;
};

/**
 * @ingroup quantize lib
 * @brief: int data array and length.
 */
struct IntData {
    unsigned int length;
    int* data;
};

/**
 * @ingroup quantize lib
 * @brief: Params for Arq Quantiation.
 */
struct ArqParam {
    unsigned int numBits;
    bool channelWise;
    bool withOffset;
};

/**
 * @ingroup quantize lib
 * @brief: Params for Ifmr Quantiation.
 */
struct IfmrParam {
    unsigned int calibration;
    unsigned int numBits;
    bool withOffset;
    float startRatio;
    float endRatio;
    float step;
    float maxPercentile;
    float minPercentile;
};

/**
 * @ingroup quantize lib
 * @brief: Params for Ifmr Quantiation
 */
template <class T>
struct MaxMinValue {
    T maxValue;
    T minValue;
};

/**
 * @ingroup quantize lib
 * @brief Param for NUQ
 */
struct NuqParam {
    unsigned int numSteps;
    bool withOffset;
    unsigned int numIter;
};

/**
  * @ingroup quantize lib
  * @brief: Ifmr Quantization Function.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] ifmrParam: ifmr quant param.
  * @param [in|out] scale: scale data.
  * @param [in|out] offset: offset data.
  * @return succ/fail
  */
int IfmrQuant(float* data, unsigned int length, IfmrParam ifmrParam, FloatData scale, IntData offset);

/**
  * @ingroup quantize lib
  * @brief: Ifmr Quantization Function.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] ifmrParam: ifmr quant param.
  * @param [in|out] scale: scale data.
  * @param [in|out] offset: offset data.
  * @return succ/fail
  */
int IfmrQuant(double* data, unsigned int length, IfmrParam ifmrParam, FloatData scale, IntData offset);

int IfmrQuant_gpu(float* deviceData, float* hostDatadata, unsigned int length, IfmrParam ifmrParam,
                  FloatData scale, IntData offset);

/**
  * @ingroup quantize lib
  * @brief: Ifmr Quantization Function.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] ifmrParam: ifmr quant param.
  * @param [in|out] scale: scale data.
  * @param [in|out] offset: offset data.
  * @return succ/fail
  */
int IfmrQuant_gpu(double* deviceData, double* hostDatadata, unsigned int length, IfmrParam ifmrParam,
                  FloatData scale, IntData offset);

/*lint -save -e1077*/
int SearchN_gpu(std::vector<std::vector<double>>& storedData,
    std::vector<double>& deqScale,
    std::vector<int>& bestN);

int SearchN_gpu(std::vector<std::vector<float>>& storedData,
    std::vector<float>& deqScale,
    std::vector<int>& bestN);

void SearchShitBit(std::vector<std::vector<float>>& storedData,
    std::vector<float>& deqScale,
    std::vector<int>& bestN);

void SearchShitBit(std::vector<std::vector<double>>& storedData,
    std::vector<double>& deqScale,
    std::vector<int>& bestN);
/*lint -restore*/

/**
 * @ingroup quantize lib
 * @brief: Switch the first and second axis
 * @param: [in|out] data: flatten input data
 * @param: [in] length: the length of data
 * @param: [in|out] shape: the shape of data
 */
template <class T>
void TransposeAB(T* data,
                 int length,
                 std::vector<int>& shape);//lint !e1077

/**
 * @ingroup quantize lib
 * @brief: Check whether the clip value is reasonable
 * @param: [in|out] clipMax: the new clip max value
 * @param: [in|out] clipMin: the new clip min value
 * @param: [in|out] clipMaxPre: last clip max value
 * @param: [in|out] clipMinPre: last clip min value
 */
template <class T>
bool ClipCheck(T* clipMax,
               T* clipMin,
               T& clipMaxPre,
               T& clipMinPre);

/**
 * @ingroup quantize lib
 * @brief: Universal Linear Quantization on Activations
 * @param [in|out] data: input data
 * @param [in] length: input data length
 * @param [in] scale: scale data
 * @param [in] offset: offset data
 */
template <class T>
void Ulq(T* data,
         int length,
         FloatData scale,
         IntData offset);

/**
 * @ingroup quantize lib
 * @brief: ULQ gradient calculation
 * @param: [in] bottomData: bottom data
 * @param: [in] topDiff: top diff
 * @param: [in] length: the length of data
 * @param: [in] clip: a vector of clip_max, clip_min, clip_max_ori, clip_min_ori
 * @return: vector<double> a vector of clip_max_diff and clip_min_diff
 */
/*lint -save -e1077*/
template <class T>
std::vector<T> UlqDiff(const T* bottomData,
                       const T* topDiff,
                       const int length,
                       std::vector<T> clip);
/*lint -restore*/

int ActQuantForwardGpu(const int count,
                       const double* bottomData,
                       double* topData,
                       double* clipMaxGpu,
                       double* clipMinGpu,
                       bool fixedMin,
                       int quantBitNum);

int ActQuantForwardGpu(const int count,
                       const float* bottomData,
                       float* topData,
                       float* clipMaxGpu,
                       float* clipMinGpu,
                       bool fixedMin,
                       int quantBitNum);

int UlqDiffGpu(const int count,
               const float* bottomData,
               float* bottomDiff,
               const float* topDiff,
               const float* clipMaxGpu,
               const float* clipMinGpu,
               float& diffMaxCpuRef,
               float& diffMinCpuRef,
               int quantBitNum);

int UlqDiffGpu(const int count,
               const double* bottomData,
               double* bottomDiff,
               const double* topDiff,
               const double* clipMaxGpu,
               const double* clipMinGpu,
               double& diffMaxCpuRef,
               double& diffMinCpuRef,
               int quantBitNum);

/**
  * @ingroup quantize lib
  * @brief: Arq Quantization Function.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] arqParam: arq quant param.
  * @param [in|out] scale: scale data.
  * @param [in|out] offset: offset data.
  * @return succ/fail
  */
template <class T>
int ArqQuant(T* data,
             unsigned int length,
             ArqParam arqParam,
             FloatData scale,
             IntData offset);

/**
  * @ingroup quantize lib
  * @brief: Arq Quantization Function.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] arqParam: arq quant param.
  * @param [in] scale: scale data.
  * @param [in] offset: offset data.
  * @param [out] int8Data: output int8 data.
  * @return succ/fail
  */
template <class T>
int ArqQuantReal(T* data,
                 unsigned int length,
                 ArqParam arqParam,
                 FloatData scale,
                 IntData offset,
                 char* int8Data);
/**
  * @ingroup quantize lib
  * @brief: Arq Quantization Function.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] arqParam: arq quant param.
  * @param [in|out] scale: scale data.
  * @param [in|out] offset: offset data.
  * @return succ/fail
  */
template <class T>
int ArqQuantGPU(T* data,
                unsigned int length,
                ArqParam arqParam,
                FloatData scale,
                IntData offset);

template <class Dtype>
int ArqQuantRetrainGPU(Dtype* devData,
                       unsigned int length,
                       ArqParam arqParam,
                       FloatData scale,
                       IntData offset);
/**
  * @ingroup quantize lib
  * @brief: Arq Quantization Function.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] arqParam: arq quant param.
  * @param [in] scale: scale data.
  * @param [in] offset: offset data.
  * @param [out] int8Data: output int8 data.
  * @return succ/fail
  */
template <class T>
int ArqQuantRealGPU(T* data,
                    unsigned int length,
                    ArqParam arqParam,
                    FloatData scale,
                    IntData offset,
                    char* int8Data);

/**
  * @ingroup quantize lib
  * @brief: Check Arq Quantization Inputs Params.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] arqParam: arq quant param.
  * @param [in] scale: scale data.
  * @param [in] offset: offset data.
  * @return succ/fail
  */
Status CheckArqQuantParams(float* data,
                           unsigned int length,
                           ArqParam arqParam,
                           FloatData scale,
                           IntData offset);

Status CheckArqQuantParams(double* data,
                           unsigned int length,
                           ArqParam arqParam,
                           FloatData scale,
                           IntData offset);

#ifdef __cplusplus
extern "C"
{
#endif

/**
  * @ingroup quantize lib
  * @brief: Arq Quantization Function.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] arqParam: arq quant param.
  * @param [in|out] scale: scale data.
  * @param [in|out] offset: offset data.
  * @return succ/fail
  */
int ArqQuantDoublePython(double* data, unsigned int length, ArqParam arqParam, FloatData scale, IntData offset);

/**
  * @ingroup quantize lib
  * @brief: Arq Quantization Function.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] scale: scale data.
  * @param [in] offset: offset data.
  * @param [in] numBits: bit number to quantize.
  * @param [out] int8Data: output int8 data.
  * @return succ/fail
  */
int QuantRealDoublePython(double* data,
                          unsigned int length,
                          FloatData scale,
                          IntData offset,
                          unsigned int numBits,
                          char* int8Data);
/**
  * @ingroup quantize lib
  * @brief: Arq Quantization Function.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] arqParam: arq quant param.
  * @param [in|out] scale: scale data.
  * @param [in|out] offset: offset data.
  * @return succ/fail
  */
int ArqQuantDoublePythonGPU(double* data,
                            unsigned int length,
                            ArqParam arqParam,
                            FloatData scale,
                            IntData offset);

/**
  * @ingroup quantize lib
  * @brief: Arq Quantization Function.
  * @param [in] data: input data.
  * @param [in] length: inputs data length.
  * @param [in] scale: scale data.
  * @param [in] offset: offset data.
  * @param [in] numBits: bit number to quantize.
  * @param [out] int8Data: output int8 data.
  * @return succ/fail
  */
int QuantRealDoublePythonGPU(double* data,
                             unsigned int length,
                             FloatData scale,
                             IntData offset,
                             unsigned int numBits,
                             char* int8Data);

/**
 * @ingroup quantize lib
 * @brief: Non-uniform Quantization Function
 * @param [in] data: input data
 * @param [in] length: input data length
 * @param [in] nuqParam: nuq parameters
 * @param [in] scale: scale data
 * @param [in] offset: offset data
 * @return [out] success/fail
 */
int NuqQuantPython(double* data,
                   unsigned int length,
                   NuqParam nuqParam,
                   FloatData scale,
                   IntData offset);

/**
 * @ingroup quantize lib
 * @brief: Non-uniform Quantization Function
 * @param [in] data: input data
 * @param [in] length: input data length
 * @param [in] scale: scale data
 * @param [in] offset: offset data
 * @param [out] int8Data: output int8 data
 * @return [out] success/fail
 */
int NuqQuantRealPython(double* data,
                       unsigned int length,
                       FloatData scale,
                       IntData offset,
                       char* int8Data);

/**
 * @ingroup quantize lib
 * @brief: Non-uniform Quantization Function
 * @param [in] data: input data
 * @param [in] length: input data length
 * @param [in] nuqParam: nuq parameters
 * @param [in] scale: scale data
 * @param [in] offset: offset data
 * @return [out] success/fail
 */
int NuqQuantPythonGPU(double* data,
                      unsigned int length,
                      NuqParam nuqParam,
                      FloatData scale,
                      IntData offset);

/**
 * @ingroup quantize lib
 * @brief: Non-uniform Quantization Function
 * @param [in] data: input data
 * @param [in] length: input data length
 * @param [in] scale: scale data
 * @param [in] offset: offset data
 * @param [out] int8Data: output int8 data
 * @return [out] success/fail
 */
int NuqQuantRealPythonGPU(double* data,
                          unsigned int length,
                          FloatData scale,
                          IntData offset,
                          char* int8Data);

#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif /* QUANT_H */
