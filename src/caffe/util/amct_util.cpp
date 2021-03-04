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
 * @brief amct_util src file
 *
 * @file amct_util.cpp
 *
 * @version 1.0
 */
#include "caffe/util/amct_util.hpp"

using std::string;

namespace caffe {
void ConvertLayerName(std::string& originalLayerName,
                      const std::string& subString,
                      const std::string& replaceString)
{
    string::size_type pos = 0;
    string::size_type subStrLength = subString.size();
    string::size_type replaceStringLen = replaceString.size();
    while ((pos = originalLayerName.find(subString, pos)) != string::npos) {
        originalLayerName.replace(pos, subStrLength, replaceString);
        pos += replaceStringLen;
    }
}
} // namespace caffe