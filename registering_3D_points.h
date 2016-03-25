#pragma once
#include "stdafx.h"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "PointCloud.h"
#include "DataInterface.h"
#include "World_point.h"


using namespace std;
using namespace cv;

/*
* Finding the matches between 3D points
*/
void register_point_cloud(vector<PointCloud>& ptcloud);
/*
* Matching the features
*/
void find_corresponding_points(vector<PointCloud>& ptcloud, vector<World_point>& temp_array);
/*
* Updating the point cloud
*/
void record_point_cloud(vector<PointCloud> ptcloud);
/*
* Arranging data according to MBA standard
*/
void convert_to_NVM(vector<PointCloud>& ptcloud, vector<World_point>& wdpt,
	vector<CameraT>& camera_data, vector<Point3D>& point_data,
	vector<Point2D>& measurements, vector<int>& ptidx, vector<int>& camidx,
	vector<string>& names, vector<int>& ptc);

/*
*This function will transfer all data to the 1st camera coordinate system
*/
void transfer_to_1st_cam(vector<PointCloud>& ptcloud);

