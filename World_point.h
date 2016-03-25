#pragma once
#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include "DataInterface.h"
#include "opencv2/core.hpp"

using namespace std;
using namespace cv;


/*
* Temporary class to store data
*/
class World_point
{
public:
	World_point(int i) { 
		pt_idx_3D = i; 
		point.SetPoint((float)-1, (float)-1, (float)-1);
	};
	World_point()
	{
		pt_idx_3D = -1;
	};
	~World_point()
	{
	};

	int pt_idx_3D; // 3D point index of this object
	Point3D point; // 3D location of this point
	vector<Point2D> measurements; // Projections
	vector<int> camera_idx; // Camera index
	vector<int> pt_idx; // 3D point index

};


