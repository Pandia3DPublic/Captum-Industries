#include "genCors.h"



bool matchsort(match& i, match& j) {
	return i.d < j.d;
}
//returns sorted correspondences between two keypointunits
//the query index (p1 in match) alwayss refer to the frame with the lower unique_id
//todo Sollte auf gpu laufen. Sollte kd-tree artige strutktur fuer binaere daten nutzen
void getCors(std::shared_ptr<KeypointUnit> f1,std::shared_ptr<KeypointUnit> f2, std::vector<match> &out)
{
	if (f1->unique_id > f2->unique_id) { //cannot happen atm
		auto tmp = f1;
		f1 = f2;
		f2 = tmp;
		cout <<"Frames switched in getCors due to unique ids!: " <<  f1->unique_id << " 2 " << f2->unique_id << endl;;
	}
	vector<cv::DMatch> matches;
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming"); //todo this should be flann-hamming, but no idea how to do that in c++
	matcher->match(f1->orbDescriptors, f2->orbDescriptors, matches, cv::Mat());

	//translate to custom format
	out = std::vector<match>(); //start with empty in case stuff is left. Important if two or more frame were removed.
	out.reserve(matches.size());
	match tmp;
	for (auto m : matches) {
		tmp.indeces << m.queryIdx, m.trainIdx; //get the indeces
		tmp.d = static_cast<double>(m.distance); // get the distance
		tmp.p1 = f1->orbKeypoints[m.queryIdx].p;
		tmp.p2 = f2->orbKeypoints[m.trainIdx].p;
		out.push_back(tmp);

	}
	std::sort(out.begin(), out.end(), matchsort);
}
