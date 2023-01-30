#pragma once
#include "match.h"
#include <vector>
#include <unordered_map> 
#include <string>

//base class that contains all variables needed for intra and interchunk alignment
using namespace std;



class Optimizable
{
public:
	Optimizable();
	~Optimizable();
	//variables
	c_umap<string, vector<match>> rawmatches; //vector containing a further vector with matches for each pair of frames. todo superflous, only for debugging
	c_umap<string, vector<match>> filteredmatches; //vector containing a further vector with filtered matches for each pair of frames (kabsch filter).
	c_umap<string, pairTransform> pairTransforms;
	vector<rmatch> efficientMatches; // contains only the cors that are used for sparse alignment. Indeces only


private:

};

