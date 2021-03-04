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
 * @brief amct_util head file
 *
 * @file amct_util.h
 *
 * @version 1.0
 */
#ifndef AMCT_UTIL_H
#define AMCT_UTIL_H

#include <string>
namespace caffe {
const std::string REPLACE_STR = "%2F";
void ConvertLayerName(std::string& originalLayerName,
                      const std::string& subString,
                      const std::string& replaceString);
}
namespace util
{
    using Status = unsigned int;
    Status ProcessScale(float& currentScale);
    Status ProcessScale(double& currentScale);
}

#define RAW_PRINTF        printf
#define LOG_DEBUG(fmt, arg...) RAW_PRINTF("[DEBUG][%s][%d] " fmt, __FUNCTION__, __LINE__, ## arg)
#define LOG_INFO(fmt, arg...) RAW_PRINTF("[INFO][%s][%d] " fmt, __FUNCTION__, __LINE__, ## arg)
#define LOG_ERROR(fmt, arg...) RAW_PRINTF("[ERROR][%s][%d] " fmt, __FUNCTION__, __LINE__, ## arg)

#endif