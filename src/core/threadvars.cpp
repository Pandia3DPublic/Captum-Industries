#include "threadvars.h"
#include <atomic>
#include <mutex>

using namespace std;

std::atomic<bool> stopMeshing(false); //signaling bool to stop the meshing thread
std::atomic<bool> g_current_slam_finished(false); // signals that the current slam is finished, integration might be lacking.
std::atomic<bool> g_trackingLost(false); //indicates if tracking is lost
std::atomic<bool> g_reconThreadFinished(false); //indicates that the recon thread and all its subthreads finished
std::atomic<bool> g_wholeMesh(false); //indicates that the recon thread and all its subthreads finished
std::mutex seglock; //segmentation lock
std::mutex meshlock; //mesh lock
std::mutex solverlock; //gpu solver lock
std::mutex currentposlock; //just for the current position

//solver communication
std::atomic<bool> g_solverRunning(false);

//gui comm
std::atomic<bool> g_take_dataCam(false);
std::atomic<int> g_programState = gui_READY;



