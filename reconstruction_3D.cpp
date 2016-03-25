// reconstruction_3D.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "stereo_image_processsing.h"
#include "PointCloud.h"
#include "registering_3D_points.h"
#include "ImgDescriptor.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define ESSENTIAL_MAT_CAL_FAILED	25
#define NOT_ENOUGH_INLIERS			30
#define DETECT_AND_MATCH			123
#define MATCH_ONLY					125

bool read_cal_data(Mat& cam_mat, Mat& dist_coef);
bool read_image_list(vector<string>& image_list);
bool est_3D_pts(ImgDescriptor& key_descriptor, ImgDescriptor& second_descriptor,
	Mat& R, Mat& T, vector<Point3f>& points_3D,
	vector<int>& inliers_1, vector<int>& inliers_2, vector<DMatch>& good_matches);

// Camera matrix and distortion coefficients are global
Mat camera_matrix;
Mat distortion_coeff;

int main()
{

	if (!read_cal_data(camera_matrix, distortion_coeff)) {
		cout << "Loading camera calibration data failed";
	}

	cout << "-------------- Camera Matrix -------------------" <<endl<< camera_matrix << endl;
	cout << "----------Distortion coefficients---------------" << endl << distortion_coeff << endl;

	vector<string> image_list;
	read_image_list(image_list);
	int max_iterations = (int)image_list.size();
	vector<ImgDescriptor> features_surf;
	if (!detect_and_record_features(image_list, features_surf)) {
		cout << "Reading Image failed !" << endl;
	}

	

	// Key frame number and number of images to match
	int keyframe = 0;
	int no_of_frames = 25;
	int second_img = 1;
	int count = 0;
	Mat R(3, 3, CV_32FC1);
	Mat T(1, 3, CV_32FC1);
	vector<PointCloud> data;


	for (int i = 0; count < no_of_frames; i++)
	{

		vector<Point3f> points_3D;
		vector<int> inliers_1, inliers_2;
		vector<DMatch> good_matches;

		bool returnval = est_3D_pts(features_surf[keyframe], features_surf[second_img], R, T, points_3D, inliers_1, inliers_2, good_matches);
		if (returnval) {	
			Point3f s(0.0f, 0.0f, -1.0f);
			int temp_count = 0;
			vector<Point3f> temp_vec(features_surf[keyframe].keypoints.size(),s );

			for (size_t i = 0; i < features_surf[keyframe].keypoints.size(); i++)
			{
				if (inliers_1[i] == 1)
				{
					temp_vec[i] = points_3D[temp_count];
					temp_count++;
				}
			}
			PointCloud PC1(features_surf[keyframe].im_name, features_surf[second_img].im_name);
			PC1.keypoints_1 = features_surf[keyframe].keypoints;
			PC1.keypoints_2 = features_surf[second_img].keypoints;
			PC1.points_1 = features_surf[keyframe].points;
			PC1.points_2 = features_surf[second_img].points;
			PC1.points_3D = temp_vec;
			PC1.T = T.clone();
			PC1.R = R.clone();
			PC1.descriptors_1 = features_surf[keyframe].descriptor.clone();
			PC1.descriptors_2 = features_surf[second_img].descriptor.clone();
			PC1.keyframe = keyframe;
			PC1.good_matches = good_matches;
			PC1.second_cam = count;
			vector<int> temp_idx(PC1.keypoints_1.size(), -1);
			PC1.point_idx_2D_1 = temp_idx;
			PC1.point_idx_2D_2 = temp_idx;
			PC1.point_idx_3D = temp_idx;
			PC1.inliers_1 = inliers_1;
			PC1.inliers_2 = inliers_2;
			data.push_back(PC1);
			cout << "##########################################" << endl << features_surf[keyframe].points.size() << endl << features_surf[second_img].points.size() << endl << "##############################################" << endl;
			//cout << PC1.img_1 << PC1.img_2 << endl;
			count++;
			
		}
		second_img++;
		waitKey(500);
		
	}

	register_point_cloud(data);



    return 0;
}

/* 
* Reading the calibration data from file
*/
bool read_cal_data(Mat& cam_mat, Mat& dist_coef)
{
	std::string file_name = "out_camera_data.xml";
	FileStorage fs(file_name, FileStorage::READ);

	if (!fs.isOpened()) {
		cout << "Could not open the configuration file: \"" << file_name << "\"" << endl;
		return false;
	}

	else {
		Mat cam_matrix, dist_coefficients;
		fs["camera_matrix"] >> cam_matrix;
		fs["distortion_coefficients"] >> dist_coefficients;

		cam_mat = cam_matrix;
		dist_coef = dist_coefficients;

		return true;
	}
	return true;
}


/*
* Reading the image list from file
*/
bool read_image_list(vector<string>& image_list)
{
	FileStorage fs("imd10.xml", FileStorage::READ);
	FileNode n = fs.getFirstTopLevelNode();
	FileNodeIterator it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
	{
		image_list.push_back((string)*it);
	}
	return true;
}


/*
* Estimating the 3D positions new
*/
bool est_3D_pts(ImgDescriptor& key_descriptor, ImgDescriptor& second_descriptor, 
	Mat& R, Mat& T, vector<Point3f>& points_3D, 
	vector<int>& inliers_1, vector<int>& inliers_2, vector<DMatch>& good_matches)
{
	calc_matches(key_descriptor.keypoints, second_descriptor.keypoints, key_descriptor.descriptor, second_descriptor.descriptor, good_matches);

	if (good_matches.size()<30)
	{
		cout << "------ Not Enough Features to continue! -------------" << endl;
		return false;
	}

	// Creating inliers
	for (size_t i = 0; i < key_descriptor.keypoints.size(); i++)
	{
		inliers_1.push_back(0);
	}

	for (size_t i = 0; i < second_descriptor.keypoints.size(); i++)
	{
		inliers_2.push_back(0);
	}


	vector<Point2f> matched_pt_1, matched_pt_2;
	for (size_t i = 0; i < good_matches.size(); i++)
	{
		matched_pt_1.push_back(key_descriptor.keypoints[good_matches[i].queryIdx].pt);
		inliers_1[good_matches[i].queryIdx] = 1;
		matched_pt_2.push_back(second_descriptor.keypoints[good_matches[i].trainIdx].pt);
		inliers_2[good_matches[i].trainIdx] = 1;
	}
	

	Mat essential_mat;
	vector<uchar> inliers;
	essential_mat_cal(matched_pt_1, matched_pt_2, camera_matrix, essential_mat, inliers);
	int inliers_num = countNonZero(inliers);
	if (inliers_num < 20)
	{
		cout << "------- Not enough inliers to calculate the Essential Matrix! -----------" << endl;
		return false;
	}


	/////////////////////////////////////////////////////////////////////////
	// Modifying the inliers
	int temp_count = 0;
	for (size_t i = 0; i < key_descriptor.keypoints.size(); i++)
	{
		if (inliers_1[i] == 1)
		{
			if (inliers[temp_count] == 0)
			{
				inliers_1[i] = 0;
			}
			temp_count++;
		}
	}
	temp_count = 0;
	for (size_t i = 0; i < second_descriptor.keypoints.size(); i++)
	{
		if (inliers_2[i] == 1)
		{
			if (inliers[temp_count] == 0)
			{
				inliers_2[i] = 0;
			}
			temp_count++;
		}
	}
	//////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////
	// Evaluating the Essential Matrix
	//Check the condition of essential matrix rank should be 2
	if (fabs(determinant(essential_mat)) > 1e-07)
	{
		cout << "det(essential_mat) !=0" << endl;
		return false;
	}
	// Refining the points with inliers
	keep_vector_by_status_points(matched_pt_1, matched_pt_2, inliers);

	/* Recover the pose of second camera */

	bool recovered = recover_pose(camera_matrix, essential_mat, matched_pt_1, matched_pt_2, R, T);
	
	if (!recovered)
	{
		cout << "------- Camera Pose recovery failed -------------------------" << endl;
		return false;
	}

	// Projection matrices
	Mat P = Mat::eye(3, 4, CV_32FC1);
	Mat P1;
	hconcat(R, T, P1);
	
	// 3D points
	
	bool temp  = calc_3D_points_and_reproject(camera_matrix, matched_pt_1, matched_pt_2, P, P1, points_3D, key_descriptor.im_name, second_descriptor.im_name, inliers, 0);
	if (!temp)
	{
		cout << "--------- Large reprojection Errors ---------------------------" << endl;
		return false;
	}
	//////////////////////////////////////////////////////////////////////////

	cout << "Size of 3d points:" << points_3D.size()<<endl;


	




	return true;
}




