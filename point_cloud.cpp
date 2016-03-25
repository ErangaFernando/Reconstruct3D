#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "PointCloud.h"

using namespace std;
using namespace cv;

PointCloud::PointCloud(){}
PointCloud::PointCloud(std::string im_1_name, std::string im_2_name) {
	img_1 = im_1_name;
	img_2 = im_2_name;
}



