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
#include "registering_3D_points.h"
#include "PointCloud.h"
#include "DataInterface.h"
#include "World_point.h"
#include "util.h"
#include "pba.h"


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define MATCH_ONLY					125


void register_point_cloud(vector<PointCloud>& ptcloud)
{
	int max_3D_idx = 0;
	int length = (int)ptcloud.size();
	vector<World_point> temp_array;
	cout << ptcloud[0].inliers_1.size()<<",";
	cout << ptcloud[0].keypoints_1.size()<<endl;

	//record_point_cloud(ptcloud);

	for (int i = 0; i < (int)ptcloud[0].keypoints_1.size(); i++)
	{
		int temp_count = 0;
		for (int j = 0; j < length; j++)
		{  
			if (ptcloud[j].inliers_1[i] == 1)
			{
				ptcloud[j].point_idx_3D[i] = max_3D_idx;				
				temp_count++;
			}
			
		}
		if (temp_count != 0)
		{
			max_3D_idx++;
		}
		
	}

	

	cout << "Different Number of 3D points: " << max_3D_idx << endl;

	for (size_t i = 0; i < length; i++)
	{
		int count = countNonZero(ptcloud[i].inliers_1);
		int count1 = countNonZero(ptcloud[i].inliers_2);
		cout<<ptcloud[i].points_3D.size()<<","<< count<<","<<count1<<endl;
	}

	
	for (int i = 0; i < max_3D_idx; i++)
	{
		World_point wdpt(i);
		temp_array.push_back(wdpt);
	}

	
	find_corresponding_points(ptcloud, temp_array);

	vector<CameraT> camera_data;
	vector<Point3D> point_data;
	vector<Point2D> measurements; 
	vector<int> ptidx;
	vector<int> camidx;
	vector<string> names;
	vector<int> bla;
	convert_to_NVM(ptcloud, temp_array, camera_data, point_data, measurements, ptidx, camidx, names, bla);
	for (size_t i = 0; i < point_data.size(); i++)
	{
		//cout << point_data[i].xyz[0] << "," << point_data[i].xyz[1] << "," << point_data[i].xyz[2] << endl;
	}
	vector<int> ptc(measurements.size(), 0);

	// Running the MBA
	cout << "Initializing MBA................" << endl;
	ParallelBA::DeviceT device = ParallelBA::PBA_CPU_DOUBLE;
	ParallelBA pba(device);          //You should reusing the same object for all new data

	pba.SetCameraData(camera_data.size(), &camera_data[0]);                        //set camera parameters
	pba.SetPointData(point_data.size(), &point_data[0]);                            //set 3D point data
	pba.SetProjection(measurements.size(), &measurements[0], &ptidx[0], &camidx[0]);//set the projections
	cout << "Starting MBA................" << endl;
	pba.RunBundleAdjustment();    //run bundle adjustment, and camera_data/point_data will be modified

	cout << "MBA done................" << endl;
	//Write the optimized system to file
	const char*  outpath = "my_output.nvm";
	SaveModelFile(outpath, camera_data, point_data, measurements, ptidx, camidx, names, ptc);
	cout << "Completed................" << endl;
	
	
}

/*
* Finding the corresponsind points in 3D
*/
void find_corresponding_points(vector<PointCloud>& ptcloud, vector<World_point>& temp_array)
{
	

	float cx = 639.5f;
	float cy = 359.5f;
	float x1, y1, x2, y2;
	size_t length = ptcloud.size();
	
	for (size_t i = 0; i < length; i++)
	{
		cout << "Good Measurement size" << ptcloud[i].good_matches.size() << "," << endl;
		for (size_t j = 0; j < ptcloud[i].good_matches.size(); j++)
		{
			int index_in = ptcloud[i].good_matches[j].queryIdx;
			if (ptcloud[i].inliers_1[index_in]==1)
			{
				int index = ptcloud[i].point_idx_3D[index_in];
				ptcloud[i].point_idx_2D_1[ptcloud[i].good_matches[j].queryIdx] = index;
				ptcloud[i].point_idx_2D_2[ptcloud[i].good_matches[j].trainIdx] = index;

			}

		}
	}

	for (size_t i = 0; i < ptcloud.size(); i++)
	{
		for (size_t j = 0; j < ptcloud[i].keypoints_1.size(); j++)
		{
			if (ptcloud[i].point_idx_3D[j] != -1)
			{
				if (temp_array[ptcloud[i].point_idx_3D[j]].point.xyz[2] == -1 && ptcloud[i].points_3D[j].z > 0)
				{
					temp_array[ptcloud[i].point_idx_3D[j]].point.SetPoint(ptcloud[i].points_3D[j].x, ptcloud[i].points_3D[j].y, ptcloud[i].points_3D[j].z);
				}
			}

			if (ptcloud[i].point_idx_2D_1[j] != -1)
			{
				x1 = ptcloud[i].points_1[j].x; y1 = ptcloud[i].points_1[j].y;
				Point2D p(x1 - cx, y1 - cy);
				temp_array[ptcloud[i].point_idx_2D_1[j]].measurements.push_back(p);
				temp_array[ptcloud[i].point_idx_2D_1[j]].pt_idx.push_back(ptcloud[i].point_idx_2D_1[j]);
				temp_array[ptcloud[i].point_idx_2D_1[j]].camera_idx.push_back(ptcloud[i].keyframe);
			}
			if (ptcloud[i].point_idx_2D_2[j] != -1)
			{
				x1 = ptcloud[i].points_2[j].x; y1 = ptcloud[i].points_2[j].y;
				Point2D p(x1 - cx, y1 - cy);
				temp_array[ptcloud[i].point_idx_2D_2[j]].measurements.push_back(p);
				temp_array[ptcloud[i].point_idx_2D_2[j]].pt_idx.push_back(ptcloud[i].point_idx_2D_2[j]);
				temp_array[ptcloud[i].point_idx_2D_2[j]].camera_idx.push_back(ptcloud[i].second_cam);
			}
		}
	}


	
}
/*
* Reecording the point cloud
*/
void record_point_cloud(vector<PointCloud> ptcloud) {
	ofstream os;
	os.open("Raw Data.txt", ios::out);
	Mat r0 = Mat::eye(3, 3, CV_32FC1);
	Mat t0 = Mat::zeros(3, 1, CV_32FC1);
	for (size_t i = 0; i < ptcloud.size(); i++)
	{
		os << "----------------------- Point Cloud " << i << "------------------------------\n";
		if (i == 0) {
			os << "Rotation Matrix: \n" << r0 << "\nTranslation:" << t0 << "\n";
		}
		else
		{
			os << "Rotation Matrix: \n" << ptcloud[i-1].R << "\nTranslation:" << ptcloud[i - 1].T << "\n";
		}
		os << "Keyframe :" << ptcloud[i].keyframe << "\n";
		os << "Second Cam :" << ptcloud[i].second_cam << "\n";
		os << "Image 1 :" << ptcloud[i].img_1 << "\n";
		os << "Image 2 :" << ptcloud[i].img_2 << "\n";

		os << "Cam 1 details\n";
		for (size_t j = 0; j < ptcloud[i].keypoints_1.size(); j++)
		{
			os << ptcloud[i].points_3D[j] << "," << ptcloud[i].inliers_1[j] << "," << ptcloud[i].points_1[j] << "\n";
		}
		os << "Cam 2 details\n";
		for (size_t j = 0; j < ptcloud[i].keypoints_2.size(); j++)
		{
			os << ptcloud[i].inliers_2[j] << "," << ptcloud[i].points_2[j] << "\n";
		}

		os << "---------------------------------------------------- Matches --------------------------------------------------\n";
		for (size_t j = 0; j < ptcloud[i].good_matches.size(); j++)
		{
			os << ptcloud[i].good_matches[j].queryIdx << "," << ptcloud[i].good_matches[j].trainIdx << "\n";
		}

	}
	os.close();
}

/*
* Arranging data according to MBA standard
*/

void convert_to_NVM(vector<PointCloud>& ptcloud, vector<World_point>& temp_array,
	vector<CameraT>& camera_data, vector<Point3D>& point_data,
	vector<Point2D>& measurements, vector<int>& ptidx, vector<int>& camidx,
	vector<string>& names, vector<int>& ptc)
{
	size_t length = temp_array.size();
	for (size_t i = 0; i < length; i++)
	{
		point_data.push_back(temp_array[i].point);
		for (size_t j = 0; j < temp_array[i].measurements.size(); j++)
		{
			Point2D p(temp_array[i].measurements[j].x, temp_array[i].measurements[j].y);
			measurements.push_back(p);
			camidx.push_back(temp_array[i].camera_idx[j]);
			ptidx.push_back((temp_array[i].pt_idx[j]));
		}
	}
	cout << "3d point data " << point_data.size() << endl << "ptidx" << ptidx.size() << endl;
	cout << "Measu: " << measurements.size() << endl << "camidx: " << camidx.size() << endl;

	// Creating the camera
	length = ptcloud.size();
	float focal_length = 1092.840093468892f;
	CameraT cam0;
	Mat R = Mat::eye(3, 3, CV_32FC1);
	Mat r1;
	Rodrigues(R, r1);
	float r0[3];
	r0[0] = r1.at<float>(0);
	r0[1] = r1.at<float>(1);
	r0[2] = r1.at<float>(2);
	cam0.SetRodriguesRotation(r0);
	float t0[3];
	t0[0] =0.0f;
	t0[1] = 0.0f;
	t0[2] = 0.0f;
	cam0.SetTranslation(t0);
	cam0.f = focal_length;

	camera_data.push_back(cam0);
	
	for (int i = 1; i < length; i++)
	{
		CameraT cam;
		Mat R_n;
		cam.SetConstantCamera();
		Rodrigues(ptcloud[i-1].R, R_n);
		float r[3];
		r[0] = R_n.at<float>(0);
		r[1] = R_n.at<float>(1);
		r[2] = R_n.at<float>(2);
		cam.SetRodriguesRotation(r);
		float t[3];
		t[0] = ptcloud[i-1].T.at<float>(0);
		t[1] = ptcloud[i-1].T.at<float>(1);
		t[2] = ptcloud[i-1].T.at<float>(2);
		cam.SetTranslation(t);
		cam.f = focal_length;
		camera_data.push_back(cam);
		names.push_back(ptcloud[i].img_1);		
	}

	CameraT cam;
	Mat R_n;
	cam.SetConstantCamera();
	Rodrigues(ptcloud[length - 1].R, R_n);
	float r[3];
	r[0] = R_n.at<float>(0);
	r[1] = R_n.at<float>(1);
	r[2] = R_n.at<float>(2);
	cam.SetRodriguesRotation(r);
	float t[3];
	t[0] = ptcloud[length - 1].T.at<float>(0);
	t[1] = ptcloud[length - 1].T.at<float>(1);
	t[2] = ptcloud[length - 1].T.at<float>(2);
	cam.SetTranslation(t);
	cam.f = focal_length;
	camera_data.push_back(cam);
	names.push_back(ptcloud[length-1].img_2);

	cout << "cam data:" << camera_data.size() << endl << "names" << names.size() << endl;
}

/*
*This function will transfer all data to the 1st camera coordinate system
*/
void transfer_to_1st_cam(vector<PointCloud>& ptcloud)
{
	
}






