#include "clientCamera.h"
#include "core/threadvars.h"
#include "ptimer.h"
#include "protos/framedata.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/util/delimited_message_util.h>
#include "genKps.h"
#include "compressutil.h"
#include "../../utils/coreutil.h"

static k4a::calibration calib;
static SOCKET ClientSocket;
//################################# Client Camera #####################################
template <typename Proto>
int recvProto(SOCKET* csock, std::shared_ptr<Proto> proto)
{
	char buffer[sizeof(google::protobuf::uint32)]; //4 byte buffer for storing size of message
	memset(buffer, '\0', sizeof(google::protobuf::uint32));

	int bytecount = 0;

	do {
		//Peek into the socket to get the header with the size of the message
		int bytecount = recv(*csock, buffer, sizeof(google::protobuf::uint32), MSG_PEEK); //does not remove from queue
		if (bytecount > 0) {
			*proto = *recvDeserializedMessage<Proto>(*csock, getMessageSize(buffer)); //second read with info about size of package
		}
		else if (bytecount == 0) {
			printf("No bytes received...\n");
			//std::this_thread::sleep_for(std::chrono::milliseconds(20));
		}
		else {
			printf("recv failed with error: %d\n", WSAGetLastError());
			return 1;
		}
	} while (false);

	return 0;
}


//get sockaddr, ipv4 or ipv6
void* get_in_addr(struct sockaddr* sa)
{
	if (sa->sa_family == AF_INET) {
		return &(((struct sockaddr_in*)sa)->sin_addr);
	}
	return &(((struct sockaddr_in6*)sa)->sin6_addr);
}


//Get size of proto message from 4 byte header
google::protobuf::uint32 getMessageSize(char* buf)
{
	google::protobuf::uint32 msgSize;
	google::protobuf::io::ArrayInputStream ais(buf, sizeof(google::protobuf::uint32));
	google::protobuf::io::CodedInputStream input(&ais);
	input.ReadVarint32(&msgSize);//Decode the header and get the size
	return msgSize;
}
//Receive entire proto message from client, de-serialize and return it
template <typename Proto>
std::shared_ptr<Proto> recvDeserializedMessage(SOCKET& csock, google::protobuf::uint32 msgSize)
{
	int bytecount;
	auto recvSize = msgSize + sizeof(google::protobuf::uint32); //size of entire message including 4 byte message length header
	std::shared_ptr<Proto> message = std::make_shared<Proto>();
	char* buffer = new char[recvSize];

	//Read the entire buffer including the 4 byte header
	if ((bytecount = recv(csock, buffer, recvSize, MSG_WAITALL)) == -1) { //MSG_WAITALL get complete message
		printf("recv failed with error: %d\n", WSAGetLastError()); //todo proper error handling
	}
	//cout << "Bytes received: " << bytecount << endl;

	//Assign ArrayInputStream with enough memory. Deserialize
	google::protobuf::io::ArrayInputStream ais(buffer, recvSize);
	google::protobuf::io::CodedInputStream input(&ais);

	//Read an unsigned integer with Varint encoding, truncating to 32 bits.
	input.ReadVarint32(&msgSize);

	//After the message's length is read, PushLimit() is used to prevent the CodedInputStream 
	//from reading beyond that length.Limits are used when parsing length-delimited 
	//embedded messages
	google::protobuf::io::CodedInputStream::Limit msgLimit = input.PushLimit(msgSize);

	//De-Serialize
	message->ParseFromCodedStream(&input);

	//Once the embedded message has been parsed, PopLimit() is called to undo the limit
	input.PopLimit(msgLimit);

	//Print the message for debug
	//cout << "Message is " << framedata->DebugString();

	delete[] buffer;
	return message;

}

int recvHandler(SOCKET* csock, list <shared_ptr<proto::FrameData>>* protobuffer, std::atomic<bool>* stop)
{
	char buffer[sizeof(google::protobuf::uint32)]; //4 byte buffer for storing size of message
	memset(buffer, '\0', sizeof(google::protobuf::uint32));

	int bytecount = 0;

	while (!(*stop))
	{
		//Peek into the socket to get the header with the size of the message
	//	cout << "\nreceiving... " << endl;
		bytecount = recv(*csock, buffer, sizeof(google::protobuf::uint32), MSG_PEEK);
		if (bytecount > 0) {
			auto framedata = recvDeserializedMessage<proto::FrameData>(*csock, getMessageSize(buffer)); //recv proto

			//cout << "colorImg: " << framedata->colorimage().ByteSize() << endl;
			//cout << "depthImg: " << framedata->depthimage().ByteSize() << endl;

			//cout << "received color image " << framedata->colorimage().timestamp() << endl;
			//cout << "received depth image " << framedata->depthimage().timestamp() << endl;

			g_protobufflock.lock();
			protobuffer->push_back(framedata);
			g_protobufflock.unlock();
		}
		else if (bytecount == 0) {
			printf("No bytes received...\n");
			*stop = true;
			//std::this_thread::sleep_for(std::chrono::milliseconds(20));
		}
		else {
			printf("recv failed with error: %d\n", WSAGetLastError());
			return 1;
		}
	}

	return 0;
}

shared_ptr<cv::Mat> getSingleImage(std::list <shared_ptr<cv::Mat>>& imagebuffer)
{
	while (imagebuffer.empty()) {
		std::this_thread::sleep_for(10ms);
	}
	g_imagebufflock.lock();
	auto tmp = imagebuffer.front();
	imagebuffer.pop_front();
	g_imagebufflock.unlock();
	return tmp;
}

shared_ptr<vector<short>> getSingleImage(list <shared_ptr<vector<short>>>& imagebuffer)
{
	while (imagebuffer.empty()) {
		std::this_thread::sleep_for(10ms);
	}
	g_imagebufflock.lock();
	auto tmp = imagebuffer.front();
	imagebuffer.pop_front();
	g_imagebufflock.unlock();
	return tmp;
}


int initializeServer() {
	//#################################################### Initialize sockets ###############################################################
	WSADATA wsaData; //version-data
	int iResult; // helper-variable
	SOCKET ListenSocket = INVALID_SOCKET;
	ClientSocket = INVALID_SOCKET;

	struct addrinfo* result = NULL;
	struct addrinfo hints;
	struct sockaddr_storage client_addr; //connector's address information
	socklen_t client_size; //sizeof client_addr
	char ipclient[INET6_ADDRSTRLEN]; //for printing client's ip addr

	// Initialize Winsock
	iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (iResult != 0) {
		printf("WSAStartup failed with error: %d\n", iResult);
		return 1;
	}

	ZeroMemory(&hints, sizeof(hints));	// Make sure the struct is empty
	hints.ai_family = AF_INET;			// IPv4
	hints.ai_socktype = SOCK_STREAM;	// TCP stream sockets
	hints.ai_protocol = IPPROTO_TCP;	// TCP Protocol
	hints.ai_flags = AI_PASSIVE;		// fill in my IP for me

	// Resolve the server address and port
	iResult = getaddrinfo(NULL, DEFAULT_PORT, &hints, &result);
	if (iResult != 0) {
		printf("getaddrinfo failed with error: %d\n", iResult);
		WSACleanup();
		return 1;
	}

	// Create a SOCKET for connecting to server
	ListenSocket = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
	if (ListenSocket == INVALID_SOCKET) {
		printf("socket failed with error: %ld\n", WSAGetLastError());
		freeaddrinfo(result);
		WSACleanup();
		return 1;
	}

	// Setup the TCP listening socket. Bind binds to local port
	iResult = ::bind(ListenSocket, result->ai_addr, (int)result->ai_addrlen);
	if (iResult == SOCKET_ERROR) {
		printf("bind failed with error: %d\n", WSAGetLastError());
		freeaddrinfo(result);
		closesocket(ListenSocket);
		WSACleanup();
		return 1;
	}
	freeaddrinfo(result);

	//###################################################### Listen & accept ###############################################################
	iResult = listen(ListenSocket, SOMAXCONN);
	if (iResult == SOCKET_ERROR) {
		printf("listen failed with error: %d\n", WSAGetLastError());
		closesocket(ListenSocket);
		WSACleanup();
		return 1;
	}
	cout << "Waiting for connection...\n";

	// Accept a client socket
	client_size = sizeof(client_addr);
	ClientSocket = accept(ListenSocket, (struct sockaddr*) & client_addr, &client_size);
	if (ClientSocket == INVALID_SOCKET) {
		printf("accept failed with error: %d\n", WSAGetLastError());
		closesocket(ListenSocket);
		WSACleanup();
		return 1;
	}
	//only used to get printable ip adress
	inet_ntop(client_addr.ss_family, get_in_addr((struct sockaddr*) & client_addr), ipclient, sizeof(ipclient)); //ntop == network to presentation/printable
	cout << "Server: connected to " << ipclient << endl;

	//====================================================== moved to serverInitialise()================================

	// No longer need server socket
	closesocket(ListenSocket);

	//disable Nagles algorithm as it causes 500ms delay (tested on localhost connection)
	BOOL opt = FALSE;
	int optlen = sizeof(BOOL);
	iResult = setsockopt(ClientSocket, IPPROTO_TCP, TCP_NODELAY, (char*)&opt, optlen);
	if (iResult == SOCKET_ERROR) {
		wprintf(L"setsockopt for TCP_NODELAY failed with error: %u\n", WSAGetLastError());
	}


	//####################################################### Receive Calibration ###############################################################

	// receive raw calibration
	cout << "receiving raw calibration..." << endl;
	shared_ptr<proto::Calibration> rawCalib = make_shared<proto::Calibration>();
	if (recvProto<proto::Calibration>(&ClientSocket, rawCalib) == 1) {
		return 1;
	}

	//################################################## Convert Calibration data ###########################################################
// calibration for transform object
	vector<uint8_t> vcalib(rawCalib->data().begin(), rawCalib->data().end());
	vcalib.push_back('\0');
	calib = calib.get_from_raw(vcalib, K4A_DEPTH_MODE_NFOV_UNBINNED, K4A_COLOR_RESOLUTION_1536P);
	k4a::transformation transform(calib);

	// convert intrinsics
	//take the color intrinsic since depth img is warped to color. Note: This assumes the camera image is a pinhole camera which is an approximation (todo maybe)
	auto& ccc = calib.color_camera_calibration;
	auto& param = calib.color_camera_calibration.intrinsics.parameters.param;
	camera::PinholeCameraIntrinsic intrinsic(ccc.resolution_width, ccc.resolution_height, param.fx, param.fy, param.cx, param.cy);

	//note that the input pictures have to be scalled to this resolution
	g_intrinsic = getScaledIntr(intrinsic, g_resx, g_resy);
	g_intrinsic_cuda = open3d::cuda::PinholeCameraIntrinsicCuda(g_intrinsic);
	g_lowIntr = getScaledIntr(g_intrinsic, g_lowx, g_lowy); //todo get rid of getlowintr


}


void StartClientThreads(list <shared_ptr<Frame>>& framebuffer, std::atomic<bool>& stop) {
	using namespace cv;
	//##### equivalent to core threads
	cout << "receiving image data..." << endl;
	list <shared_ptr<proto::FrameData>> protobuffer; //protobuffer contains deserialized proto image data and gets filled by recvThread->recvHandler
	std::thread recvThread(recvHandler, &ClientSocket, &protobuffer, &stop); // fills protobuffer with deserialized proto image data, loops until stop = true

	//####################################################### decompress thread ###############################################################
	list <shared_ptr<cv::Mat>> colorbuffer;
	//list <shared_ptr<cv::Mat>> depthbuffer;
	list <shared_ptr<vector<short>>> depthbuffer;
	std::thread decompThread(decompressFFmpegRVL, &protobuffer, &colorbuffer, &depthbuffer, &stop);  // gets proto from buffer and decompresses it, pushes to imagebuffer
	int iResult; // helper-variable

	//#######framebuffer stuff

	// convert proto image data and push to framebuffer
	while (!stop)
	{
		// get image from imagebuffer
		auto cvCol = getSingleImage(colorbuffer);
		auto depth = getSingleImage(depthbuffer);

		//// create k4a depth img
		//k4a::image depth_k4a = depth_k4a.create_from_buffer(K4A_IMAGE_FORMAT_DEPTH16, depth->cols, depth->rows, depth->step[0],
		//	(uint8_t*)depth->data, depth->step[0] * depth->rows, 0, 0);

		k4a::image depth_k4a = depth_k4a.create_from_buffer(K4A_IMAGE_FORMAT_DEPTH16, 640, 576, 640 * 2,
			(uint8_t*)depth->data(), 640 * 576 * 2, 0, 0); // res due to nfov unbinned

		// align depth to color
		k4a::transformation transform(calib);
		k4a::image depth_aligned = transform.depth_image_to_color_camera(depth_k4a); // todo don't scale up and scale down again

		// resize images to resx * resy
		Mat cvDepth = Mat(Size(depth_aligned.get_width_pixels(), depth_aligned.get_height_pixels()), CV_16U, (void*)depth_aligned.get_buffer(), Mat::AUTO_STEP);
		resize(cvDepth, cvDepth, Size(g_resx, g_resy), 0, 0, INTER_NEAREST); //no interpol since this will give artifacts for depth images

		//open3d images
		shared_ptr<geometry::Image> color_image_ptr = make_shared<geometry::Image>();
		shared_ptr<geometry::Image> depth_image_ptr = make_shared<geometry::Image>();

		//convert to open3d image
		color_image_ptr->Prepare(cvCol->cols, cvCol->rows, 3, 1);
		//memcpy doesnt work here since color channels are the wrong way round
		//if (cvCol3.isContinuous()) {
		//	memcpy(color_image_8bit.data_.data(), cvCol3.data, cvCol3.total() * cvCol3.elemSize());
		//}
		//else {
#pragma omp parallel for 
		for (int y = 0; y < cvCol->rows; y++) {
			uint8_t* pixel = cvCol->ptr<uint8_t>(y); // point to first color in row
			for (int x = 0; x < cvCol->cols; x++) {
				*color_image_ptr->PointerAt<uint8_t>(x, y, 2) = *pixel++;
				*color_image_ptr->PointerAt<uint8_t>(x, y, 1) = *pixel++;
				*color_image_ptr->PointerAt<uint8_t>(x, y, 0) = *pixel++;
			}
		}
		//}

		depth_image_ptr->Prepare(cvDepth.cols, cvDepth.rows, 1, 2);
		if (cvDepth.isContinuous()) {
			memcpy(depth_image_ptr->data_.data(), cvDepth.data, cvDepth.total() * cvDepth.elemSize());
		}
		else {
#pragma omp parallel for
			for (int y = 0; y < cvDepth.rows; y++) {
				uint16_t* pixel_d = cvDepth.ptr<uint16_t>(y); //point to first pixel in row
				for (int x = 0; x < cvDepth.cols; x++) {
					*depth_image_ptr->PointerAt<uint16_t>(x, y) = *pixel_d++;
				}
			}
		}


		auto tmp = std::make_shared<Frame>();
		//6-15ms
		generateFrame(color_image_ptr, depth_image_ptr, tmp);
		generateOrbKeypoints(tmp, *cvCol);


		g_bufferlock.lock();
		framebuffer.push_back(tmp);
		g_bufferlock.unlock();

	}

	recvThread.join();
	decompThread.join();


	//########################################################## shutdown ###################################################################
	// shutdown the connection since we're done
	cout << "Shutting down connection" << endl;
	iResult = shutdown(ClientSocket, SD_RECEIVE);
	if (iResult == SOCKET_ERROR) {
		printf("shutdown failed with error: %d\n", WSAGetLastError());
		closesocket(ClientSocket);
		WSACleanup();
		return;
	}

	// cleanup
	cout << "Performing cleanup" << endl;
	closesocket(ClientSocket);
	WSACleanup();

	return;
}

//################################# End Client Camera #####################################

