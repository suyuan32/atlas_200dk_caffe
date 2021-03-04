#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/activation_qat_layer.hpp"
#include "caffe/util/math_functions.hpp"

template <typename Dtype>
__global__ void QuantBackward(
    const int n,
    const Dtype* in_data,
    Dtype* in_diff,
    const Dtype* clip_max,
    const Dtype* clip_min) {
    CUDA_KERNEL_LOOP(index, n) {
        if (in_data[index] < clip_min[0]) {
            in_diff[index] = 0;
        } else if(in_data[index] > clip_max[0]) {
            in_diff[index] = 0;
        }
    }
}
namespace caffe {

template <typename Dtype>
void ActivationQATLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int count = bottom[0]->count();

    // record layer shape
    if (!record_flag_) {
        std::string delim = "-";
        auto start = 0U;
            object_layer_ = object_layer_ + "-";
        auto end = object_layer_.find(delim);

        while (end != std::string::npos) {
            auto layer = object_layer_.substr(start, end - start);
            Status ret = RecordRetrainShape(
                this->layer_param_.retrain_data_quant_param().record_file_path(),
                layer, bottom[0]->channels(),
                bottom[0]->height(), bottom[0]->width());
            CHECK_EQ(ret, 0) << "Record shape to file failed.";
            LOG(INFO) << "Do layer:\"" << layer << "\" activation retrain success!";
                start = end + delim.length();
                end = object_layer_.find(delim, start);
        }
        record_flag_ = true;
    }

    Dtype clip_max_ori_ = static_cast<Dtype*>(this->blobs_[0]->mutable_cpu_data())[0];
    Dtype clip_min_ori_ = static_cast<Dtype*>(this->blobs_[1]->mutable_cpu_data())[0];
    if (clip_max_ori_ <= clip_min_ori_) {
        LOG(INFO) << "# clipmax is less or equal to clip_min!";
        clip_max_ori_ = clip_max_pre_;
        clip_min_ori_ = clip_min_pre_;
        LOG(INFO) << "# correct clipmax clipmin success!";
    } else if (clip_max_ori_ < 0) {
        clip_max_ori_ = clip_max_pre_;
        clip_min_pre_ = clip_min_ori_;
    } else if (clip_min_ori_ > 0) {
        clip_min_ori_ = clip_min_pre_;
        clip_max_pre_ = clip_max_ori_;
    } else {
        clip_max_pre_ = clip_max_ori_;
        clip_min_pre_ = clip_min_ori_;
    }

    Dtype* clip_max_gpu_ = this->blobs_[0]->mutable_gpu_data();
    Dtype* clip_min_gpu_ = this->blobs_[1]->mutable_gpu_data();
    Dtype* clip_max_cpu_ = this->blobs_[0]->mutable_cpu_data();
    Dtype* clip_min_cpu_ = this->blobs_[1]->mutable_cpu_data();

    if (ifmr_init_ && ifmrParam_.calibration == 0) {
        const Dtype* host_bottom_data = bottom[0]->cpu_data();
        for (int index = 0; index < count; ++index) {
            stored_data_for_calibration_.push_back(host_bottom_data[index]);
        }
        IfmrQuant_gpu(top_data, stored_data_for_calibration_.data(), count, ifmrParam_, scale_, offset_);

        Dtype min = 0;
        Dtype max = 0;
        min = -(offset_.data[0] + pow(2, NUM_BITS_QUANT - 1)) * scale_.data[0];
        max = (pow(2, NUM_BITS_QUANT) - 1) * scale_.data[0] + min;

        stored_data_for_calibration_.clear();
        stored_data_for_calibration_.shrink_to_fit();
        ifmrParam_.calibration = 1;

        clip_max_cpu_[0] = max;
        clip_min_cpu_[0] = min;

        caffe_copy(count, bottom_data, top_data);
    } else {
        if (fixed_min_) {
            clip_min_cpu_[0] = 0;
        }
        int ret = ActQuantForwardGpu(count, bottom_data, top_data, clip_max_gpu_, clip_min_gpu_, fixed_min_, NUM_BITS_QUANT);
        CHECK_EQ(ret, 0) << "Do retrain forward act quant failed!";
    }
}

template <typename Dtype>
void ActivationQATLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                             const vector<bool>& propagate_down,
                                             const vector<Blob<Dtype>*>& bottom) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();

    Dtype* clip_max_gpu_ = this->blobs_[0]->mutable_gpu_data();
    Dtype* clip_min_gpu_ = this->blobs_[1]->mutable_gpu_data();

    //max min diff
    Dtype diff_max_cpu_, diff_min_cpu_;
    Dtype& diffMaxCpuRef = diff_max_cpu_;
    Dtype& diffMinCpuRef = diff_min_cpu_;
    int ret = UlqDiffGpu(count, bottom_data, bottom_diff, top_diff, clip_max_gpu_, clip_min_gpu_,
                         diffMaxCpuRef, diffMinCpuRef, NUM_BITS_QUANT);
    CHECK_EQ(ret, 0) << "Do retrain backward act diff failed!";

    Dtype* clip_max_diff = this->blobs_[0]->mutable_gpu_diff();
    Dtype* clip_min_diff = this->blobs_[1]->mutable_gpu_diff();
    cudaMemcpy(clip_max_diff, &diff_max_cpu_, sizeof(Dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(clip_min_diff, &diff_min_cpu_, sizeof(Dtype), cudaMemcpyHostToDevice);

    caffe_copy(count, top_diff, bottom_diff);
    QuantBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom_diff, clip_max_gpu_, clip_min_gpu_);
    CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ActivationQATLayer);

}  // namespace caffe
