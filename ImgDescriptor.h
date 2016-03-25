#pragma once
#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"


using namespace std;
using namespace cv;

class ImgDescriptor
{
public:
	std::string im_name;
	std::vector<cv::KeyPoint> keypoints;
	std::vector<Vec3b> colors;
	std::vector<Point2f> points;
	Mat descriptor;

	ImgDescriptor() {};
	~ImgDescriptor() {};
};