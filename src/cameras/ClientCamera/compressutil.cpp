#include "compressutil.h"

using namespace std;
using namespace cv;

//####################################### Fast Lossless Depth Image Compression ###########################################

int ff_librvldepth_compress_rvl(const short* input, char* output, int numPixels) {
	int* buffer, * pBuffer, word, nibblesWritten, value;
	int zeros, nonzeros, nibble;
	int i;
	const short* p;
	const short* end;
	short previous;
	short current;
	int delta;
	int positive;
	word = 0;
	buffer = pBuffer = (int*)output;
	nibblesWritten = 0;
	end = input + numPixels;
	previous = 0;
	while (input != end) {
		zeros = 0, nonzeros = 0;
		for (; (input != end) && !*input; input++, zeros++);
		// EncodeVLE(zeros);  // number of zeros
		value = zeros;
		do {
			nibble = value & 0x7;  // lower 3 bits
			if (value >>= 3) nibble |= 0x8;  // more to come
			word <<= 4;
			word |= nibble;
			if (++nibblesWritten == 8) {
				// output word
				*pBuffer++ = word;
				nibblesWritten = 0;
				word = 0;
			}
		} while (value);
		for (p = input; (p != end) && *p++; nonzeros++);
		// EncodeVLE(nonzeros);
		value = nonzeros;
		do {
			nibble = value & 0x7;  // lower 3 bits
			if (value >>= 3) nibble |= 0x8;  // more to come
			word <<= 4;
			word |= nibble;
			if (++nibblesWritten == 8) {
				// output word
				*pBuffer++ = word;
				nibblesWritten = 0;
				word = 0;
			}
		} while (value);
		for (i = 0; i < nonzeros; i++) {
			current = *input++;
			delta = current - previous;
			positive = (delta << 1) ^ (delta >> 31);
			// EncodeVLE(positive);  // nonzero value
			value = positive;
			do {
				nibble = value & 0x7;  // lower 3 bits
				if (value >>= 3) nibble |= 0x8;  // more to come
				word <<= 4;
				word |= nibble;
				if (++nibblesWritten == 8) {
					// output word
					*pBuffer++ = word;
					nibblesWritten = 0;
					word = 0;
				}
			} while (value);
			previous = current;
		}
	}
	if (nibblesWritten)  // last few values
		*pBuffer++ = word << 4 * (8 - nibblesWritten);
	return (int)((char*)pBuffer - (char*)buffer);  // num bytes
}

void ff_librvldepth_decompress_rvl(const char* input, short* output, int numPixels) {
	const int* buffer, * pBuffer;
	int word, nibblesWritten, value, bits;
	unsigned int nibble;
	short current, previous;
	int numPixelsToDecode;
	int positive, zeros, nonzeros, delta;
	numPixelsToDecode = numPixels;
	buffer = pBuffer = (const int*)input;
	nibblesWritten = 0;
	previous = 0;
	while (numPixelsToDecode) {
		// int zeros = DecodeVLE();  // number of zeros
		value = 0;
		bits = 29;
		do {
			if (!nibblesWritten) {
				word = *pBuffer++;
				nibblesWritten = 8;
			}
			nibble = word & 0xf0000000;
			value |= (nibble << 1) >> bits;
			word <<= 4;
			nibblesWritten--;
			bits -= 3;
		} while (nibble & 0x80000000);
		zeros = value;
		numPixelsToDecode -= zeros;
		for (; zeros; zeros--)
			*output++ = 0;
		// int nonzeros = DecodeVLE();  // number of nonzeros
		value = 0;
		bits = 29;
		do {
			if (!nibblesWritten) {
				word = *pBuffer++;
				nibblesWritten = 8;
			}
			nibble = word & 0xf0000000;
			value |= (nibble << 1) >> bits;
			word <<= 4;
			nibblesWritten--;
			bits -= 3;
		} while (nibble & 0x80000000);
		nonzeros = value;
		numPixelsToDecode -= nonzeros;
		for (; nonzeros; nonzeros--) {
			// int positive = DecodeVLE();  // nonzero value
			value = 0;
			bits = 29;
			do {
				if (!nibblesWritten) {
					word = *pBuffer++;
					nibblesWritten = 8;
				}
				nibble = word & 0xf0000000;
				value |= (nibble << 1) >> bits;
				word <<= 4;
				nibblesWritten--;
				bits -= 3;
			} while (nibble & 0x80000000);
			positive = value;
			delta = (positive >> 1) ^ -(positive & 1);
			current = previous + delta;
			*output++ = current;
			previous = current;
		}
	}
}

//############################################################################################################
//######################################## ffmpeg video decompression ########################################
//############################################################################################################

shared_ptr<proto::FrameData> getSingleProto(std::list <shared_ptr<proto::FrameData>>& protobuffer)
{
	while (protobuffer.empty()) {
		std::this_thread::sleep_for(10ms);
	}
	g_protobufflock.lock();
	auto tmp = protobuffer.front();
	protobuffer.pop_front();
	g_protobufflock.unlock();
	return tmp;
}

// encoded packet gets send to decoder, receive decoded frame, convert frame format, push to own buffer
static void decode(AVCodecContext* codecx, AVFrame* frame, AVPacket* packet, std::list <shared_ptr<cv::Mat>>& imagebuffer, AVFrame* convFrame, struct SwsContext* scaler, bool isColor)
{
	int ret;
	ret = avcodec_send_packet(codecx, packet);
	if (ret < 0) {
		fprintf(stderr, "Error sending a packet for decoding\n");
		exit(1);
	}

	while (ret >= 0) {
		ret = avcodec_receive_frame(codecx, frame);
		if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
			return;
		}
		else if (ret < 0) {
			fprintf(stderr, "Error during decoding\n");
			exit(1);
		}

		// now decoded!!
		//printf("saving frame %3d\n", codecx->frame_number);
		//fflush(stdout);

		// convert format to bgr
		sws_scale(scaler, frame->data, frame->linesize, 0, frame->height, convFrame->data, convFrame->linesize);
		
		// fill in cv mat, push to buffer
		if (isColor) {
			Mat color(Size(convFrame->width, convFrame->height), CV_8UC3, (void*)convFrame->data[0], Mat::AUTO_STEP);

			g_imagebufflock.lock();
			imagebuffer.push_back(make_shared<cv::Mat>(color));
			g_imagebufflock.unlock();
		}
		else {
			Mat depth(Size(convFrame->width, convFrame->height), CV_16U);
			for (int y = 0; y < depth.rows; y++) {
				uint16_t* pixel = depth.ptr<uint16_t>(y); // point to first pixel in row
				for (int x = 0; x < depth.cols; x++) {
					uint8_t d1 = convFrame->data[0][y * convFrame->linesize[0] + 3 * x]; //b
					uint8_t d2 = convFrame->data[0][y * convFrame->linesize[0] + 3 * x + 1]; //g
					*pixel++ = (d2 << 8) | d1;
				}
			}

			g_imagebufflock.lock();
			imagebuffer.push_back(make_shared<cv::Mat>(depth));
			g_imagebufflock.unlock();
		}

	}
}


void decompressFFmpeg(list <shared_ptr<proto::FrameData>>* protobuffer, list <shared_ptr<cv::Mat>>* colorbuffer, list <shared_ptr<cv::Mat>>* depthbuffer, std::atomic<bool>* stop) //todo output, maybe opencv mat?
{
	av_log_set_level(AV_LOG_VERBOSE); // for debug
	int ret;

	const AVCodec* codec;
	AVCodecContext* codecx = NULL;
	const AVCodec* codec2;
	AVCodecContext* codecx2 = NULL;

	AVPacket* pktColor;
	AVFrame* avColor;
	AVFrame* avColorConv;
	AVPacket* pktDepth;
	AVFrame* avDepth;
	AVFrame* avDepthConv;

	int widthCol = 640, heightCol = 480;
	int widthDep = 640, heightDep = 576;

	// ################### color alloc ###################
	// alloc frame
	avColor = av_frame_alloc();
	if (!avColor)
		cout << "could not allocate color frame" << endl;

	// alloc converted frame
	avColorConv = av_frame_alloc();
	if (!avColorConv)
		cout << "couldn't allocate avColorConv" << endl;
	avColorConv->format = AV_PIX_FMT_BGR24;
	avColorConv->width = widthCol;
	avColorConv->height = heightCol;
	av_image_alloc(avColorConv->data, avColorConv->linesize, avColorConv->width, avColorConv->height, (AVPixelFormat)avColorConv->format, 16);
	
	// Packet
	pktColor = av_packet_alloc();
	av_init_packet(pktColor);

	// ################### depth alloc ###################
	// alloc frame
	avDepth = av_frame_alloc();
	if (!avDepth)
		cout << "could not allocate depth frame" << endl;

	// alloc converted frame
	avDepthConv = av_frame_alloc();
	if (!avDepthConv)
		cout << "couldn't allocate avDepthConv" << endl;
	avDepthConv->format = AV_PIX_FMT_BGR24; //treat depth as bgr
	avDepthConv->width = widthDep;
	avDepthConv->height = heightDep;
	av_image_alloc(avDepthConv->data, avDepthConv->linesize, avDepthConv->width, avDepthConv->height, (AVPixelFormat)avDepthConv->format, 16);

	// Packet
	pktDepth = av_packet_alloc();
	av_init_packet(pktDepth);

	//########################################### codec ################################################
	// set codec
	codec = avcodec_find_decoder(AV_CODEC_ID_H264);
	codecx = avcodec_alloc_context3(codec);
	codec2 = avcodec_find_decoder(AV_CODEC_ID_H264);
	codecx2 = avcodec_alloc_context3(codec2);

	// open it
	avcodec_open2(codecx, codec, NULL);
	avcodec_open2(codecx2, codec2, NULL);


	//##################################### decompress loop #############################################

	// for convert gbrp to bgra
	struct SwsContext* scalerCol = sws_getContext(widthCol, heightCol, AV_PIX_FMT_GBRP, widthCol, heightCol, AV_PIX_FMT_BGR24, SWS_POINT, NULL, NULL, NULL); // nearest neighbor // SWS_POINT | SWS_ACCURATE_RND
	struct SwsContext* scalerDep = sws_getContext(widthDep, heightDep, AV_PIX_FMT_GBRP, widthDep, heightDep, AV_PIX_FMT_BGR24, SWS_POINT, NULL, NULL, NULL);

	// decode loop
	while (!(*stop))
	{
		auto framedata = getSingleProto(*protobuffer); //this locks if protobuffer empty

		pktColor->data = (uint8_t*)framedata->colorimage().data().data();
		pktColor->size = framedata->colorimage().size();
		pktDepth->data = (uint8_t*)framedata->depthimage().data().data();
		pktDepth->size = framedata->depthimage().size();


		decode(codecx, avColor, pktColor, *colorbuffer, avColorConv, scalerCol, true);
		decode(codecx2, avDepth, pktDepth, *depthbuffer, avDepthConv, scalerDep, false);

	}

	/* flush the decoder */
	decode(codecx, avColor, NULL, *colorbuffer, avColorConv, scalerCol, true);
	decode(codecx2, avDepth, NULL, *colorbuffer, avDepthConv, scalerDep, false);

	avcodec_free_context(&codecx);
	av_frame_free(&avColor);
	av_packet_free(&pktColor);

	avcodec_free_context(&codecx2);
	av_frame_free(&avDepth);
	av_packet_free(&pktDepth);

}

void decompressFFmpegRVL(list <shared_ptr<proto::FrameData>>* protobuffer, list <shared_ptr<cv::Mat>>* colorbuffer, list <shared_ptr<vector<short>>>* depthbuffer, std::atomic<bool>* stop) //todo output, maybe opencv mat?
{
	av_log_set_level(AV_LOG_VERBOSE); // for debug
	int ret;

	const AVCodec* codec;
	AVCodecContext* codecx = NULL;


	AVPacket* pktColor;
	AVFrame* avColor;
	AVFrame* avColorConv;

	int widthCol = 640, heightCol = 480;

	// ################### color alloc ###################
	// alloc frame
	avColor = av_frame_alloc();
	if (!avColor)
		cout << "could not allocate color frame" << endl;

	// alloc converted frame
	avColorConv = av_frame_alloc();
	if (!avColorConv)
		cout << "couldn't allocate avColorConv" << endl;
	avColorConv->format = AV_PIX_FMT_BGR24;
	avColorConv->width = widthCol;
	avColorConv->height = heightCol;
	av_image_alloc(avColorConv->data, avColorConv->linesize, avColorConv->width, avColorConv->height, (AVPixelFormat)avColorConv->format, 16);

	// Packet
	pktColor = av_packet_alloc();
	av_init_packet(pktColor);


	//########################################### codec ################################################
	// set codec
	codec = avcodec_find_decoder(AV_CODEC_ID_H264);
	codecx = avcodec_alloc_context3(codec);

	// open it
	avcodec_open2(codecx, codec, NULL);


	//##################################### decompress loop #############################################

	// for convert gbrp to bgra
	struct SwsContext* scalerCol = sws_getContext(widthCol, heightCol, AV_PIX_FMT_GBRP, widthCol, heightCol, AV_PIX_FMT_BGR24, SWS_POINT, NULL, NULL, NULL); // nearest neighbor // SWS_POINT | SWS_ACCURATE_RND

	// decode loop
	while (!(*stop))
	{
		auto framedata = getSingleProto(*protobuffer); //this locks if protobuffer empty

		pktColor->data = (uint8_t*)framedata->colorimage().data().data();
		pktColor->size = framedata->colorimage().size();

		// decode color
		decode(codecx, avColor, pktColor, *colorbuffer, avColorConv, scalerCol, true);

		// decode depth
		shared_ptr<vector<short>> depth = make_shared<vector<short>>();
		depth->reserve(framedata->depthimage().width() * framedata->depthimage().height() * 2);
		ff_librvldepth_decompress_rvl(framedata->depthimage().data().data(), depth->data(), framedata->depthimage().width() * framedata->depthimage().height());
		depthbuffer->push_back(depth);

	}

	/* flush the decoder */
	decode(codecx, avColor, NULL, *colorbuffer, avColorConv, scalerCol, true);

	avcodec_free_context(&codecx);
	av_frame_free(&avColor);
	av_packet_free(&pktColor);

}