/*
* stereo_image_processing.cpp
*
* Functions in this file are used in processing the stero images
* - Feature detection
* - Feature matching
* - Calculating the essential matrix
* - Calculating the fundamental matrix
* - Triangulating the points ( Part of the code obtained from code written by roy_shilkrot)
*/

#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "stereo_image_processsing.h"
#include "ImgDescriptor.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define DETECT_AND_MATCH 123

/*
* feature_detection will detect SURF features in the images
*/
void feature_detection_matching(Mat& img_1, Mat& img_2, vector<KeyPoint>& keypoints_1, vector<KeyPoint>& keypoints_2, Mat& descriptors_1, Mat& descriptors_2, vector<Point2f>& points_1, vector<Point2f>& points_2, int flag)
{
	if (flag = DETECT_AND_MATCH) {
		if (!img_1.data || !img_2.data)
		{
			std::cout << " --(!) Error reading images " << std::endl; getchar();

		}

		//-- Step 1: Detect the keypoints using SURF Detector
		int minHessian = 400;
		Ptr<SURF> detector = SURF::create(minHessian);
		
		detector->detect(img_1, keypoints_1);
		detector->detect(img_2, keypoints_2);

		/* ORB Detector*/
		//Ptr<ORB> orb = ORB::create(500, 1.2f, 8, 11, 0, 2, ORB::HARRIS_SCORE, 11);
		//orb->detect(img_1, keypoints_1);
		//orb->detect(img_2, keypoints_2);

	}
	
	//-- Step 2: Creating the descriptor for the keypoints
	Ptr<SURF> extractor = SURF::create(400);
	extractor->compute(img_1, keypoints_1, descriptors_1);
	extractor->compute(img_2, keypoints_2, descriptors_2);
	//cout << keypoints_1.size() << endl;
	//-- Step 3: Matching the keypoints
	//-- Brute force matching
	BFMatcher matcher(NORM_L2);
	std::vector< DMatch > matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	std::vector< DMatch > good_matches;
	vector<uchar> status(keypoints_1.size(),0);

	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist)
		{
			good_matches.push_back(matches[i]);
			status[i] = 1;
		}
	}

	// Drawing the good matches
	//draw_matches(img_1, img_2, keypoints_1, keypoints_2, good_matches);

	vector<KeyPoint> new_kp_1, new_kp_2;
	points_1.clear();
	points_2.clear();
	for (int i = 0; i < (int)good_matches.size(); i++)
	{
		new_kp_1.push_back(keypoints_1[good_matches[i].queryIdx]);
		points_1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
		new_kp_2.push_back(keypoints_2[good_matches[i].trainIdx]);
		points_2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
		
	}

	keypoints_1.clear();
	keypoints_1.clear();

	for (int i = 0; i < (int)good_matches.size(); i++)
	{
		keypoints_1.push_back(new_kp_1[i]);
		keypoints_2.push_back(new_kp_2[i]);

	}

	
	//cout << keypoints_1.size() << endl;
	//cout << points_1.size() << endl;

}
/*
* Drawing the good matches
*/
void draw_matches(Mat& img_1, Mat& img_2, vector<KeyPoint>& keypoints_1, vector<KeyPoint>& keypoints_2, vector<DMatch> good_matches)
{
	Mat img_matches;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//-- Show detected matches
	namedWindow("Good Matches", 0);
	imshow("Good Matches", img_matches);
	//waitKey(200);
}

/*
* Calculating the Esssential Matrix
*/
void essential_mat_cal(vector<Point2f>& points_1, vector<Point2f>& points_2, Mat& camera_mat, Mat& essential_mat, vector<uchar>& inliers) {
	double minVal, maxVal;
	minMaxIdx(points_1, &minVal, &maxVal);
	Mat fundamental_matrix = findFundamentalMat(points_1, points_2, FM_RANSAC, 0.006 * maxVal, 0.99, inliers);
	essential_mat = camera_mat.t()*fundamental_matrix*camera_mat;
	
}

/*
* Recover the pose of the second camera
*/
bool recover_pose(Mat& camera_mat, Mat& essential_mat, vector<Point2f>& points_1, vector<Point2f>& points_2, Mat& R, Mat& T)
{
	Mat R1(3, 3, CV_32FC1);
	Mat R2(3, 3, CV_32FC1);
	Mat t1(1, 3, CV_32FC1);
	Mat t2(1, 3, CV_32FC1);

	if (!DecomposeEtoRandT(essential_mat, R1, R2, t1, t2)) { return false; }
	if (determinant(R1) + 1.0 < 1e-09) {
		//according to http://en.wikipedia.org/wiki/Essential_matrix#Showing_that_it_is_valid
		cout << "det(R) == -1 [" << determinant(R1) << "]: flip E's sign" << endl;
		essential_mat = -essential_mat;
		if (!DecomposeEtoRandT(essential_mat, R1, R2, t1, t2)) return false;
	}
	if (fabs(determinant(R1)) - 1.0 > 1e-07) {
		cerr << "det(R) != +-1.0, this is not a rotation matrix";
		return false;
	}

	Mat P = Mat::eye(3, 4, CV_32FC1);

	Mat P1(3,4,CV_32FC1);
	hconcat(R1, t1, P1);
	R = R1;
	T = t1;
	bool triangulationSucceeded = true;
	if (!triangulate_points(camera_mat, points_1, points_2, P, P1))
	{
		hconcat(R1, t2, P1);
		R = R1;
		T = t2;
		if (!triangulate_points(camera_mat, points_1, points_2, P, P1))
		{
			hconcat(R2, t1, P1);
			R = R2;
			T = t1;
			if (!triangulate_points(camera_mat, points_1, points_2, P, P1))
			{
				hconcat(R2, t2, P1);
				R = R2;
				T = t2;
				if (!triangulate_points(camera_mat, points_1, points_2, P, P1))
				{
					cerr << "can't find the right P matrix\n";
					triangulationSucceeded = false;
				}
			}
		}
	}
	//cout << "R" << R << endl << "T" << T << endl;
	return triangulationSucceeded;
}

/* 
* Triangulate the points 
*/
bool triangulate_points(Mat& camMat, vector<Point2f>& points_1, vector<Point2f>& points_2, const Mat& P, const Mat& P1)
{
	// Undistor
	Mat normalized_pt1, normalized_pt2;
	undistortPoints(points_1, normalized_pt1, camMat, Mat());
	undistortPoints(points_2, normalized_pt2, camMat, Mat());

	Mat points_3D_h(4, points_1.size(), CV_32FC1);
	triangulatePoints(P, P1, normalized_pt1, normalized_pt2, points_3D_h);

	Mat pt_3d; 
	convertPointsFromHomogeneous(Mat(points_3D_h.t()).reshape(4, 1), pt_3d);

	vector<uchar> status(pt_3d.rows, 0);
	for (int i = 0; i<pt_3d.rows; i++) {
		status[i] = (pt_3d.at<Point3f>(i).z > 0) ? 1 : 0;
	}
	int count = countNonZero(status);
	double percentage = ((double)count / (double)pt_3d.rows);
	if (percentage < 0.95)
	{
		return false;
	}
	cout << count << "/" << pt_3d.rows << " = " << percentage*100.0 << "% are in front of camera" << endl;
	return true;
}
/*
* Calculating the 3D points and projecting the points back to the image
*/
bool calc_3D_points_and_reproject(Mat& camMat, vector<Point2f>& points_1, vector<Point2f>& points_2, const Mat& P, const Mat& P1, vector<Point3f>& points_3D, string& im_name_1,string& im_name_2, vector<uchar>& inliers, int display_flag)
{
	Mat img_1 = imread(im_name_1, CV_LOAD_IMAGE_COLOR);
	Mat img_2 = imread(im_name_2, CV_LOAD_IMAGE_COLOR);

	// Undistor
	Mat normalized_pt1, normalized_pt2;
	undistortPoints(points_1, normalized_pt1, camMat, Mat());
	undistortPoints(points_2, normalized_pt2, camMat, Mat());

	Mat points_3D_h(4, points_1.size(), CV_32FC1);
	triangulatePoints(P, P1, normalized_pt1, normalized_pt2, points_3D_h);

	convertPointsFromHomogeneous(Mat(points_3D_h.t()).reshape(4, 1), points_3D);

	Mat R = P(Rect(0,0, 3, 3));
	Vec3f rvec;// (0, 0, 0);
	Rodrigues(R ,rvec);
	Vec3f tvec= P.col(3);
	vector<Point2f> reprojected_pt_set_1;
	
	projectPoints(points_3D, rvec, tvec, camMat, Mat(), reprojected_pt_set_1);
	
	double reprojErr = norm(reprojected_pt_set_1, points_1, NORM_L2)/(double)points_1.size();
	

	cout << "reprojectionError: " << reprojErr << endl;
	//cout << "rep," << reprojected_pt_set_1.size() << ",pot," << points_1.size() << endl;
	for (size_t i = 0; i < reprojected_pt_set_1.size(); i++)
	{
		circle(img_1, reprojected_pt_set_1[i], 3, Scalar(255, 0, 0), 1, 8, 0);
		circle(img_1, points_1[i], 5, Scalar(0, 255, 0), 1, 8, 0);
	}

	imshow("reproj", img_1);
	if (reprojErr >0.4)
	{
		return false;
	}

	return true;
}
/*
* Decomposing the essential matrix
*/
bool DecomposeEtoRandT(Mat& E, Mat& R1, Mat& R2, Mat& t1, Mat& t2)
{

	//Using HZ E decomposition	
	SVD svd(E, SVD::MODIFY_A);
	Mat w, u, vt;
	svd.compute(E, w, u, vt, SVD::MODIFY_A);
	
	//check if first and second singular values are the same (as they should be)
	double singular_values_ratio = fabsf(w.at<double>(0) / w.at<double>(1));

	if (singular_values_ratio>1.0) singular_values_ratio = 1.0 / singular_values_ratio; // flip ratio to keep it [0,1]
	if (singular_values_ratio < 0.7) {
		cerr << "singular values of essential matrix are too far apart\n";
		return false;
	}

	Matx33d W(0, -1, 0,   //HZ 9.13
		1, 0, 0,
		0, 0, 1);
	Matx33d Wt(0, 1, 0,
		-1, 0, 0,
		0, 0, 1);
	Mat R1_double, R2_double, t1_double, t2_double;
	R1_double = u * Mat(W) * vt; //HZ 9.19
	R2_double = u * Mat(Wt) * vt; //HZ 9.19
	t1_double = u.col(2); //u3
	t2_double = -u.col(2); //u3

	R1_double.convertTo(R1, CV_32FC1);
	R2_double.convertTo(R2, CV_32FC1);
	t1_double.convertTo(t1, CV_32FC1);
	t2_double.convertTo(t2, CV_32FC1);
	return true;
}
/*
* Calculate the good matches
*/

void calc_matches(vector<KeyPoint>& keypoints_1, vector<KeyPoint>& keypoints_2, Mat& descriptor_1, Mat& descriptor_2, vector< DMatch>& good_matches)
{
	BFMatcher matcher(NORM_L2);
	std::vector< DMatch > matches;
	matcher.match(descriptor_1, descriptor_2, matches);
	double max_dist = 0; double min_dist = 100;
	good_matches.clear();
	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i <keypoints_1.size(); i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	for (int i = 0; i < descriptor_1.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}
}

/*
*Refining the keypoints
*/
void keep_vector_by_status_keypoints(vector<KeyPoint>& kp1, vector<KeyPoint>& kp2, vector<uchar>& status)
{
	vector<KeyPoint> old_kp1 = kp1;
	vector<KeyPoint> old_kp2 = kp2;

	kp1.clear();
	kp2.clear();

	for (size_t i = 0; i < status.size(); ++i) {
		if (status[i])
		{
			kp1.push_back(old_kp1[i]);
			kp2.push_back(old_kp2[i]);
		}
	}
}

void keep_vector_by_status_points(vector<Point2f>& kp1, vector<Point2f>& kp2, vector<uchar>& status)
{
	vector<Point2f> old_kp1 = kp1;
	vector<Point2f> old_kp2 = kp2;

	kp1.clear();
	kp2.clear();

	for (size_t i = 0; i < status.size(); ++i) {
		if (status[i])
		{
			kp1.push_back(old_kp1[i]);
			kp2.push_back(old_kp2[i]);
		}
	}
}

/*
* Detect Features and save them
*/
bool detect_and_record_features(vector<string>& name_list,vector<ImgDescriptor>& features_surf) 
{
	features_surf.clear();
	for (int i = 0; i < (int)name_list.size(); i++)
	{
		string im_name = name_list[i];
		Mat img_1 = imread(im_name, CV_LOAD_IMAGE_COLOR);
		if (!img_1.data)
		{
			std::cout << " --(!) Error reading images " << std::endl; getchar();
			return false;
		}

		//-- Step 1: Detect the keypoints using SURF Detector
		int minHessian = 400;
		Ptr<SURF> detector = SURF::create(minHessian);
		vector<KeyPoint> keypoints;
		detector->detect(img_1, keypoints);

		Mat descriptor;
		Ptr<SURF> extractor = SURF::create(400);
		extractor->compute(img_1, keypoints, descriptor);

		ImgDescriptor imdes;
		imdes.im_name = im_name;
		imdes.keypoints = keypoints;
		imdes.descriptor = descriptor.clone();

		for (size_t j = 0; j < keypoints.size(); j++)
		{
			float x, y;
			x = keypoints[j].pt.x;
			y = keypoints[j].pt.y;
			Vec3b intensity = img_1.at<Vec3b>(y, x);
			Point2f pt = keypoints[j].pt;

			imdes.colors.push_back(intensity);
			imdes.points.push_back(pt);
		}

		features_surf.push_back(imdes);
		cout << keypoints.size() << " Features detected on image " << im_name << endl;
	}

	return true;
	
}