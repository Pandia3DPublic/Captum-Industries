#include "segmentation.h"
#include "utils/visutil.h"



void segmentationThread(list <shared_ptr<Frame>>* segframebuffer, std::atomic<bool>* stop, std::atomic<bool>* loaded) {
	//py::scoped_interpreter guard{}; //initializes the python interpreter
	py::initialize_interpreter();
	py::module segmentModule = py::module::import("segment"); //this is alway in build folder due to cmake post build event.
	py::object segModel = segmentModule.attr("init_model")("ade20k", "slow"); //this initializes the tensorflow model on the gpu
	*loaded = true;
	while (!(*stop)) {
		while (segframebuffer->empty()) {
			std::this_thread::sleep_for(10ms);
			if (*stop)
				return;
		}
		seglock.lock();
		auto f = segframebuffer->front();
		segframebuffer->pop_front();
		seglock.unlock();
		//auto floatimg = f->rgb->CreateFloatImageMultiChannel();
		auto floatimg = f->rgbd->color_.CreateFloatImageMultiChannel(); //note untested
		auto tmpres = getSegmentedImagePython(*floatimg, segmentModule, segModel);
		auto resizedsegmap = resizeImage(tmpres, floatimg->width_, floatimg->height_, "nointerpol");
		seglock.lock();
		f->segmentationImage = resizedsegmap;
		seglock.unlock();
	}
	py::finalize_interpreter();
}
