#include "ptimer.h"
#include <Open3D/Open3D.h>

using namespace std;
std::unordered_map<std::string,std::pair<double,int>> Timer::averages;

Timer::Timer() {
	start = std::chrono::high_resolution_clock::now();
}
Timer::Timer(string a, TimeUnit unit) {
	content = a;
	unit_ = unit;
	start = std::chrono::high_resolution_clock::now();
	if (averages.count(content) == 0) {
		averages.insert({content,make_pair(0,0)});
	} 

}

Timer::Timer(string a) {
	content = a;
	start = std::chrono::high_resolution_clock::now();
	if (averages.count(content) == 0) {
		averages.insert({content, make_pair(0,0)});
	} 

}

Timer::~Timer() {
	end = std::chrono::high_resolution_clock::now();
	auto& average  =  averages[content].first;
	auto& sum = averages[content].second;
	average = (average * sum + std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / (sum+1);
	sum+=1;
	if (!done && !averageoutput) {
		if (unit_ == millisecond) {
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
			open3d::utility::LogInfo("Time elapsed in miliseconds in {} : {} \n", content.c_str(), duration.count());
		}
		if (unit_ == microsecond) {
			auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
			open3d::utility::LogInfo("Time elapsed in microseconds in {} : {} \n", content.c_str(), duration.count());
		}
		if (unit_ == nanosecond) {
			auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
			open3d::utility::LogInfo("Time elapsed in microseconds in {} : {} \n", content.c_str(), duration.count());
		}
	}
	done = true;
	//std::cout <<"Time elapsed in miliseconds in " << content << " : "<< duration.count() << std::endl;
}

void Timer::printAverage() {
	if (content.compare("scope") != 0) {
		open3d::utility::LogInfo("Average Time spend in {} in microseconds: {} \n", content.c_str(), averages[content].first);
	} else {
		open3d::utility::LogInfo("No average possible for unnamed Timer! \n");
	}
	averageoutput = true;
	
}