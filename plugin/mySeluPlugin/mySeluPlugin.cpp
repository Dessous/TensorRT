/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mySeluPlugin.h"
#include <cstring>
#include <iostream>
#include <vector>

using namespace nvinfer1;

namespace
{
const char* MYSELU_PLUGIN_VERSION{"1"};
const char* MYSELU_PLUGIN_NAME{"mySelu"};
} // namespace

PluginFieldCollection mySeluPluginCreator::mFC{};
std::vector<PluginField> mySeluPluginCreator::mPluginAttributes;

mySeluPlugin::mySeluPlugin() {}

mySeluPlugin::mySeluPlugin(nvinfer1::DataType iType, int iC, int iH, int iW, int oC, int oH, int oW)
    : iType(iType)
    , iC(iC)
    , iH(iH)
    , iW(iW)
    , oC(oC)
    , oH(oH)
    , oW(oW)
    , alpha(1.6733)
    , lambda(1.0507)
{
}

mySeluPlugin::mySeluPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    iC = read<int>(d);
    iH = read<int>(d);
    iW = read<int>(d);
    oC = read<int>(d);
    oH = read<int>(d);
    oW = read<int>(d);
    alpha = 1.6733;
    lambda = 1.0507;
    ASSERT(d == a + length);
}

int mySeluPlugin::getNbOutputs() const
{
    return 1;
}

int mySeluPlugin::initialize()
{
    return STATUS_SUCCESS;
}

void mySeluPlugin::terminate() {}

Dims mySeluPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // CHW
    nvinfer1::Dims dimsOutput;
    dimsOutput.nbDims = inputs->nbDims;
    dimsOutput.d[0] = inputs->d[0];
    dimsOutput.d[1] = inputs->d[1];
    dimsOutput.d[2] = inputs->d[2];
    dimsOutput.d[3] = inputs->d[3];
    return dimsOutput;
}

size_t mySeluPlugin::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

size_t mySeluPlugin::getSerializationSize() const
{
    // iC, iH, iW, oC, oH, oW
    return sizeof(int) * 6;
}

void mySeluPlugin::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, iC);
    write(d, iH);
    write(d, iW);
    write(d, oC);
    write(d, oH);
    write(d, oW);
    ASSERT(d == a + getSerializationSize());
}

void mySeluPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, nvinfer1::PluginFormat format, int maxBatchSize)
{
    ASSERT(nbInputs == 1);
    ASSERT(nbOutputs == 1);

    iC = inputDims->d[0];
    iH = inputDims->d[1];
    iW = inputDims->d[2];

    oC = outputDims->d[0];
    oH = outputDims->d[1];
    oW = outputDims->d[2];

    iType = inputTypes[0];
}

bool mySeluPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    return ((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW);
}

const char* mySeluPlugin::getPluginType() const
{
    return MYSELU_PLUGIN_NAME;
}

const char* mySeluPlugin::getPluginVersion() const
{
    return MYSELU_PLUGIN_VERSION;
}

void mySeluPlugin::destroy()
{
    delete this;
}

IPluginV2Ext* mySeluPlugin::clone() const
{
    auto* plugin = new mySeluPlugin(iType, iC, iH, iW, oC, oH, oW);
    return plugin;
}

void mySeluPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* mySeluPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

nvinfer1::DataType mySeluPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return inputTypes[0];
}

bool mySeluPlugin::isOutputBroadcastAcrossBatch(
    int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool mySeluPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Plugin creator
mySeluPluginCreator::mySeluPluginCreator() {}

const char* mySeluPluginCreator::getPluginName() const
{
    return MYSELU_PLUGIN_NAME;
}

const char* mySeluPluginCreator::getPluginVersion() const
{
    return MYSELU_PLUGIN_VERSION;
}

const PluginFieldCollection* mySeluPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* mySeluPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    mySeluPlugin* plugin = new mySeluPlugin();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* mySeluPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    mySeluPlugin* plugin = new mySeluPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
