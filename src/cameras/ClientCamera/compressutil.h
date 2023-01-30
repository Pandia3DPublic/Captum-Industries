#pragma once
#include <list>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "configvars.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include "protos/framedata.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/util/delimited_message_util.h>

using namespace std;

//Fast Lossless Depth Image Compression, Wilson 2017
int ff_librvldepth_compress_rvl(const short* input, char* output, int numPixels);
void ff_librvldepth_decompress_rvl(const char* input, short* output, int numPixels);

// Decompress ffmpeg
void decompressFFmpeg(list <shared_ptr<proto::FrameData>>* protobuffer, list <shared_ptr<cv::Mat>>* colorbuffer, list <shared_ptr<cv::Mat>>* depthbuffer, std::atomic<bool>* stop);
void decompressFFmpegRVL(list <shared_ptr<proto::FrameData>>* protobuffer, list <shared_ptr<cv::Mat>>* colorbuffer, list <shared_ptr<vector<short>>>* depthbuffer, std::atomic<bool>* stop);