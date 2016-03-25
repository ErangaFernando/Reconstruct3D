#pragma once


#include "stdafx.h"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "ImgDescriptor.h"


using namespace std;
using namespace cv;



/*
* feature_detection will detect SURF features in the images
*/
void feature_detection_matching(
								Mat& img_1, 
								Mat& img_2, 
								vector<KeyPoint>& keypoints_1, 
								vector<KeyPoint>& keypoints_2, 
								Mat& descriptors_1, 
								Mat& descriptors_2, 
								vector<Point2f>& points_1, 
								vector<Point2f>& points_2,
								int flag
								);
/*
* Calculating the Esssential Matrix
*/
void essential_mat_cal(vector<Point2f>& points_1, vector<Point2f>& points_2, Mat& camera_mat, Mat& essential_mat, vector<uchar>& inliers);
/*
*Estimating the pose of the second camera
*/
bool recover_pose(Mat& camera_mat, Mat& essential_mat, vector<Point2f>& points_1, vector<Point2f>& points_2, Mat& R, Mat& T);
/*
* Decomposing the essential matrix
*/
bool DecomposeEtoRandT(Mat& E, Mat& R1, Mat& R2, Mat& t1, Mat& t2);
/*``
* Triangulate the points
*/
bool triangulate_points(Mat& camMat, vector<Point2f>& points_1, vector<Point2f>& points_2, const Mat& P, const Mat& P1);
/*
* Calculating the 3D points and projecting the points back to the image
*/
bool calc_3D_points_and_reproject(Mat& camMat, vector<Point2f>& points_1, vector<Point2f>& points_2, const Mat& P, const Mat& P1, vector<Point3f>& points_3D, string& im_name_1, string& im_name_2, vector<uchar>& inliers, int display_flag);
/*
* Calculate the good matches
*/
void calc_matches(vector<KeyPoint>& keypoints_1, vector<KeyPoint>& keypoints_2, Mat& descriptor_1, Mat& descriptor_2, vector< DMatch >& good_matches);

/*
* Detect Features and save them
*/
bool detect_and_record_features(vector<string>& name_list, vector<ImgDescriptor>& features_surf);



/*
* Utilities
*/
void keep_vector_by_status_keypoints(vector<KeyPoint>& kp1, vector<KeyPoint>& kp2, vector<uchar>& status);
void keep_vector_by_status_points(vector<Point2f>& kp1, vector<Point2f>& kp2, vector<uchar>& status);
void draw_matches(Mat& img_1, Mat& img_2, vector<KeyPoint>& keypoints_1, vector<KeyPoint>& keypoints_2, vector<DMatch> good_matches);