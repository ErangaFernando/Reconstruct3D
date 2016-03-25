#pragma once
#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"


using namespace std;
using namespace cv;

class PointCloud
{
public:
	std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
	std::vector<cv::Point2f> points_1, points_2;
	std::vector<cv::Point3f> points_3D;
	std::vector<int> point_idx_3D, point_idx_2D_1, point_idx_2D_2;
	std::vector<int> ptcolor;
	std::vector<int> inliers_1, inliers_2;
	std::vector<DMatch> good_matches;
	int keyframe;
	int second_cam;
	double scale;
	std::string img_1, img_2;
	cv::Mat descriptors_1, descriptors_2;
	cv::Mat R, T;
	


	PointCloud();
	PointCloud(std::string im_1_name, std::string im_2_name);
	PointCloud(const PointCloud& rhs) {
		keypoints_1 = rhs.keypoints_1;
		keypoints_2 = rhs.keypoints_2;
		points_1 = rhs.points_1;
		points_2 = rhs.points_2;
		points_3D = rhs.points_3D;
		point_idx_2D_1 = rhs.point_idx_2D_1;
		point_idx_2D_2 = rhs.point_idx_2D_2;
		point_idx_3D = rhs.point_idx_3D;
		scale = rhs.scale;
		img_1 = rhs.img_1;
		img_2 = rhs.img_2;
		descriptors_1 = rhs.descriptors_1.clone();
		descriptors_2 = rhs.descriptors_2.clone();
		R = rhs.R.clone();
		T = rhs.T.clone();
		inliers_1 = rhs.inliers_1;
		inliers_2 = rhs.inliers_2;
		ptcolor = rhs.ptcolor;
		good_matches = rhs.good_matches;
		keyframe = rhs.keyframe;
		second_cam = rhs.second_cam;

	};
};