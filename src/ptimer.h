#pragma once
#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>

enum TimeUnit { millisecond, microsecond, nanosecond };

struct Timer {
	std::chrono::time_point<std::chrono::steady_clock> start, end;
	std::string content = "scope";
	bool done = false;
	bool averageoutput =false;
	TimeUnit unit_ = microsecond;
	Timer();
	Timer(std::string a, TimeUnit unit);
	Timer(std::string a);
	~Timer();

	static std::unordered_map<std::string,std::pair<double,int>> averages; //first entry is average, second number of contributions
	//gets the average time for the timer with the same name;
	void printAverage();
};
