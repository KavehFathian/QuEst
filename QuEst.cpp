#include <iostream> 
#include <algorithm>
#include <vector> 
#include <random>  
#include <chrono> // time module 

#include <stdio.h> 
#include <stdlib.h> 
#include <cmath> 
#include <ctime> 
#include <time.h> 

#include <opencv2/core/core.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/calib3d.hpp> 
#include <opencv2/features2d.hpp> 
#include <opencv2/xfeatures2d.hpp> 

#include <Eigen/Dense> 
#include <Eigen/Eigenvalues> 
#include <Eigen/QR> 
#include <Eigen/SVD>

#include <armadillo> 

#include <ros/ros.h> 
#include <sensor_msgs/Image.h> 
#include <geometry_msgs/Transform.h> 
#include <message_filters/subscriber.h> 
#include <message_filters/time_synchronizer.h> 
#include <message_filters/sync_policies/approximate_time.h> 
#include <image_transport/image_transport.h> 
#include <cv_bridge/cv_bridge.h> 


using namespace std; 
using namespace cv; 
using namespace cv::xfeatures2d; 
using namespace Eigen; 
using namespace sensor_msgs; 
using namespace message_filters; 
using namespace std::chrono; 

// using Eigen::RowVectorXi; // 1 by n, integer 
// using Eigen::RowVectorXd; // 1 by n, double 
// using Eigen::Matrix2Xi; // 2 by n, integer 
// using Eigen::Matrix3d; // 3 by 3, double 
// using Eigen::Matrix3Xd; // 3 by n, double 
// using Eigen::Matrix4Xd; // 4 by n, double 
// using Eigen::Matrix4Xi; // 4 by n, integer 
// using Eigen::MatrixX4i; // n by 4, integer 
// using Eigen::MatrixXd; // n by m, double 
// using Eigen::MatrixXcd; // n by m, complex double 
// using Eigen::Vector4d; // 4 by 1, double 
// using Eigen::VectorXd; // n by 1, double 

typedef sync_policies::ApproximateTime<Image, Image> MySyncPolicy; 

// define class ----------
class QuEst{
private: 
	ros::NodeHandle nh_; 
	message_filters::Subscriber<Image> sub_1, sub_2; 
	Synchronizer<MySyncPolicy> sync; 
	ros::Publisher pub; 
	geometry_msgs::Transform quest_msg; 

public: 
	QuEst(); 
	void CallBack(const sensor_msgs::ImageConstPtr& input_image_1, const sensor_msgs::ImageConstPtr& input_image_2); 
	void ImageProcess(const Mat& input_image_1, const Mat& input_image_2); 
	void QuEst_RANSAC(const Matrix3Xd& m, const Matrix3Xd& n); 
	void GetKeyPointsAndDescriptors(const Mat& img_1_input, const Mat& img_2_input, 
		vector<KeyPoint>& keypoints_1_raw, vector<KeyPoint>& keypoints_2_raw, 
		Mat& descriptors_1, Mat& descriptors_2); 
	void MatchFeaturePoints(const Mat& descriptors_1, const Mat& descriptors_2, 
						vector<DMatch>& matches_select); 

	static bool ransac(void (*QuEst_fit)(const MatrixXd&, const vector<int>&, VectorXd&), 
				const MatrixXd& data, const vector<int>& ind, VectorXd& test_model, 
			void (*QuEst_distance)(const MatrixXd&, const VectorXd&, const double, VectorXd&, vector<int>&), 
				const MatrixXd& data_1, const VectorXd& ind_1, const double distance_threshold, VectorXd& select_model, vector<int>& select_inliers, 
			bool (*QuEst_degenerate)(const MatrixXd&, const vector<int>&),  
				const MatrixXd& data_2, const vector<int>& ind_2, 
			const int minimumSizeSamplesToFit, 
			VectorXd& best_model, 
			vector<int>& best_inliers); 

	static void QuEst_fit(const MatrixXd& allData, const vector<int>& useIndices, VectorXd& test_model); 

	static void QuEst_distance(const MatrixXd& data, const VectorXd& test_model, const double distance_threshold, 
		VectorXd& select_model, vector<int>& select_inliers); 

	static bool QuEst_degenerate(const MatrixXd& data, const vector<int>& ind); 

	static void Q2R(MatrixXd& R, const Matrix4Xd& Q); 
	static void Q2R_3by3(Matrix3d& R_Q2R, const Vector4d& Q); 

	static void CoefsVer_3_1_1(MatrixXd& Cf, const Matrix3Xd& m1, const Matrix3Xd& m2); 
	static void QuEst_5Pt_Ver5_2(Matrix4Xd& Q ,const Matrix3Xd& m, const Matrix3Xd& n); 
	static void FindTrans(Vector3d& T, const Matrix3Xd& m, const Matrix3Xd& n, const Vector4d& Q); 
	static void QuEst_Ver1_1(Matrix4Xd& Q, const Matrix3Xd& m, const Matrix3Xd& n); 
	static void QuatResidue(RowVectorXd& residu, const Matrix3Xd& m1, const Matrix3Xd& m2, const Matrix4Xd& qSol); 
	
	static void coefsNum(MatrixXd& coefsN, const VectorXd& mx1, const VectorXd& mx2, 
		const VectorXd& my1, const VectorXd& my2, 
		const VectorXd& nx2, const VectorXd& ny2, 
		const VectorXd& r2, const VectorXd& s1, const VectorXd& s2); 
	static void coefsDen(MatrixXd& coefsD, const VectorXd& mx2, const VectorXd& my2, 
		const VectorXd& nx1, const VectorXd& nx2, 
		const VectorXd& ny1, const VectorXd& ny2, 
		const VectorXd& r1, const VectorXd& r2, const VectorXd& s2); 
	static void coefsNumDen(MatrixXd& coefsND, const VectorXd& a1, const VectorXd& a2, const VectorXd& a3, 
		const VectorXd& a4, const VectorXd& a5, const VectorXd& a6, const VectorXd& a7, 
		const VectorXd& a8, const VectorXd& a9, const VectorXd& a10, 
		const VectorXd& b1, const VectorXd& b2, const VectorXd& b3, const VectorXd& b4, 
		const VectorXd& b5, const VectorXd& b6, const VectorXd& b7, const VectorXd& b8, 
		const VectorXd& b9, const VectorXd& b10); 

	// static bool QuEst_degenerate(const CMatrixDouble& allData, const std::vector<size_t>& useIndices); 
	// static void QuEst_fit(const CMatrixDouble& allData, const std::vector<size_t>& useIndices, 
	// 	vector<CMatrixDouble>& fitModels); 
	// static void QuEst_distance(const CMatrixDouble& allData, const vector<CMatrixDouble>& testModels,
	// 	const double distanceThreshold, unsigned int& out_bestModelIndex, std::vector<size_t>& out_inlierIndices); 


}; 

// class constructor ----------
QuEst::QuEst(): 
sub_1(nh_, "/cam1/rgb/image_color",1), // /cam1/rgb/image_raw
sub_2(nh_, "/cam2/rgb/image_color",1), // /cam2/rgb/image_raw
sync(MySyncPolicy(100), sub_1, sub_2){ 

	pub = nh_.advertise<geometry_msgs::Transform>("quest_msg",1); 

	// run cameras 
	sync.registerCallback(boost::bind(&QuEst::CallBack, this, _1, _2)); 

} 

void QuEst::CallBack(const sensor_msgs::ImageConstPtr& input_image_1, const sensor_msgs::ImageConstPtr& input_image_2){

	// read  ROS images into OpenCV type ----------
	Mat img_1_raw, img_2_raw; 
	img_1_raw = cv_bridge::toCvCopy(input_image_1,"bgr8")->image; 
	img_2_raw = cv_bridge::toCvCopy(input_image_2,"bgr8")->image; 

	// convert color to gray ----------
	Mat img_1_gray, img_2_gray; 
	cvtColor(img_1_raw, img_1_gray, CV_BGR2GRAY); 
	cvtColor(img_2_raw, img_2_gray, CV_BGR2GRAY); 

	Mat img_1_iphone = imread( "/home/yujie/catkin_ws/formation_control/Images/iphone_1.jpg", 0); 
	// Mat img_1_iphone_gray; 
	// cvtColor(img_1_iphone, img_1_iphone_gray, CV_RGB2GRAY); 

	Mat img_2_iphone = imread( "/home/yujie/catkin_ws/formation_control/Images/iphone_2.jpg", 0); 
	// Mat img_2_iphone_gray; 
	// cvtColor(img_2_iphone, img_2_iphone_gray, CV_RGB2GRAY); 

	// feature extraction, feature matching, QuEst, and RANSAC ---------- 
	QuEst::ImageProcess(img_1_gray, img_2_gray); 
	// QuEst::ImageProcess(img_1_iphone, img_2_iphone); 

} 

void QuEst::ImageProcess(const Mat& img_1_input, const Mat& img_2_input){

	// generate feature points of two images ---------- 
	vector<KeyPoint> keypoints_1_raw, keypoints_2_raw; 
	Mat descriptors_1, descriptors_2; 
	QuEst::GetKeyPointsAndDescriptors(img_1_input,img_2_input,
		keypoints_1_raw, keypoints_2_raw, 
		descriptors_1, descriptors_2); 

	// generate matches of descriptor pairs ---------- 
	vector<DMatch> matches_select; 
	QuEst::MatchFeaturePoints(descriptors_1, descriptors_2, matches_select); 

	int numPts = 20; 

	Matrix3Xd M1(3,numPts); 
	for(int i=0;i<numPts;i++){
		M1(0,i) = keypoints_1_raw[matches_select[i].queryIdx].pt.x; 
		M1(1,i) = keypoints_1_raw[matches_select[i].queryIdx].pt.y; 
		M1(2,i) = 1; 
	}

	Matrix3Xd M2(3,numPts); 
	for(int i=0;i<numPts;i++){
		M2(0,i) = keypoints_2_raw[matches_select[i].trainIdx].pt.x; 
		M2(1,i) = keypoints_2_raw[matches_select[i].trainIdx].pt.y; 
		M2(2,i) = 1; 
	}

	Matrix3d K1; 
	K1 << 527,0,322,0,532,257,0,0,1; 
	// K1 << 1350,0,999,0,1358,525,0,0,1; 

	Matrix3d K2; 
	K2 << 517,0,306,0,521,263,0,0,1; 
	// K2 << 1413,0,919,0,1422,533,0,0,1; 

	Matrix3d K_iphone; 
	K_iphone << 3000, 0, 2016, 0, 3000, 1512, 0, 0, 1; cout<<K_iphone<<endl;

	// feature points in image coordinate ---------- 
	Matrix3Xd m1 = K1.inverse() * M1; 
// cout<<"m1: "<<endl<<m1<<endl; 
	Matrix3Xd m2 = K2.inverse() * M2; 
// cout<<"m2: "<<endl<<m2<<endl<<endl; 

	// // run QuEst, without RANSAC ---------- 
	// QuEst(m1,m2); 

	Matrix3Xd distance_small_cam1(3,9); 
	distance_small_cam1 << -0.454545454545455,-0.400000000000000,-0.300000000000000,0,0.0833333333333330,0.200000000000000,0.600000000000000,0.777777777777778,0.800000000000000, -0.454545454545455,-0.200000000000000,0.100000000000000,-0.300000000000000,-0.0833333333333335,0.200000000000000,-0.600000000000000,-0.444444444444445,0.200000000000000, 1,1,1,1,1,1,1,1,1 ; 
	Matrix3Xd distance_small_cam2(3,9); 
	distance_small_cam2 << -1.71431175985870,-1.96037308094989,-2.09014117419613,-1.04287888505732,-0.844014090892817,-0.881212226804600,-0.269787498997492,-0.148715314792843,0.0242106438334235, 0.148445880648556,0.544704667098884,1.18938804760797,0.306468877819024,0.578975860221267,1.09070102873107,-0.0173600428217116,0.104767374955274,0.818551643495114, 1,1,1,1,1,1,1,1,1 ; 

	Matrix3Xd distance_middle_cam1(3,9); 
	distance_middle_cam1 << -0.454545454545455,-0.400000000000000,-0.300000000000000,0,0.0833333333333330,0.200000000000000,0.600000000000000,0.777777777777778,0.800000000000000, -0.454545454545455,-0.200000000000000,0.100000000000000,-0.300000000000000,-0.0833333333333335,0.200000000000000,-0.600000000000000,-0.444444444444445,0.200000000000000, 1,1,1,1,1,1,1,1,1 ; 
	Matrix3Xd distance_middle_cam2(3,9); 
	distance_middle_cam2 << -2.95081718945697,-3.83730741019507,-4.65784828720804,-2.10074709667266,-1.70214021370466,-2.27190107612040,-0.807887492189780,-0.754374609414440,-0.690693577089308, 0.189585013414950,0.761868506284116,1.82069043138469,0.392002705683240,0.717312344854991,1.50871025698162,-0.0204560384258938,0.126676598628282,1.03343548455560, 1,1,1,1,1,1,1,1,1 ; 

	Matrix3Xd distance_large_cam1(3,9); 
	distance_large_cam1 << -0.454545454545455,-0.400000000000000,-0.300000000000000,0,0.0833333333333330,0.200000000000000,0.600000000000000,0.777777777777778,0.800000000000000, -0.454545454545455,-0.200000000000000,0.100000000000000,-0.300000000000000,-0.0833333333333335,0.200000000000000,-0.600000000000000,-0.444444444444445,0.200000000000000, 1,1,1,1,1,1,1,1,1 ; 
	Matrix3Xd distance_large_cam2(3,9); 
	distance_large_cam2 << -5.13542256688840,-8.20310560068015,-13.0346980689521,-3.97771139588335,-3.09907497586641,-5.39093200489630,-1.57957549699303,-1.68032888407479,-1.91455774042629, 0.262267889294865,1.26699725872958,3.88024200911560,0.543764465336476,0.942508897113401,2.44621954388548,-0.0248959988438400,0.160172228093006,1.40130103165601, 1,1,1,1,1,1,1,1,1; 

	// run QuEst_RANSAC, with RANSAC to reject outliers ---------- 
	// QuEst::QuEst_RANSAC(distance_small_cam1, distance_small_cam2); 
	// QuEst::QuEst_RANSAC(distance_middle_cam1, distance_middle_cam2); 
	// QuEst::QuEst_RANSAC(distance_large_cam1, distance_large_cam2); 
	QuEst::QuEst_RANSAC(m1,m2); 

	// draw point matching, best 20 points ---------- 
	vector<DMatch> matches_select_draw; 
	for(int i=0;i<numPts;i++){
		matches_select_draw.push_back(matches_select[i]); 
	} 

	Mat img_matches; 
	drawMatches(img_1_input,keypoints_1_raw,
				img_2_input,keypoints_2_raw,
				matches_select_draw,
				img_matches,
				Scalar::all(-1),Scalar::all(-1),
				vector<char>(),
				DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
				); 
	namedWindow("img_matches", WINDOW_NORMAL); 
	imshow("img_matches",img_matches); 
	waitKey(1); 

} 

void QuEst::GetKeyPointsAndDescriptors(const Mat& img_1_input, const Mat& img_2_input, 
	vector<KeyPoint>& keypoints_1_raw, vector<KeyPoint>& keypoints_2_raw, 
	Mat& descriptors_1, Mat& descriptors_2){

	// // use SURF features ---------- 
	// Ptr<Feature2D> detector = SURF::create(
	// 	1600, 	// hessianThreshold 
	// 	3,		// number of pyramid octaves 
	// 	4,		// number of octave layers 
	// 	true,	// extended descriptor 
	// 	false 	// upright flag 
	// 	); 

	// use SIFT features ---------- 
	Ptr<Feature2D> detector = SIFT::create(
		0, 		// nfeatures  
		3,		// nOctaveLayers  
		.04,	// contrastThreshold  
		10,		// edgeThreshold  
		1.6 	// sigma 
		); 

	// // use ORB features ---------- 
	// Ptr<Feature2D> detector = ORB::create(
	// 	500, 				// nfeatures 
	// 	1.2, 				// scaleFactor 
	// 	8, 					// nlevels 
	// 	31, 				// edgeThreshold
	// 	0, 					// firstLevel 
	// 	2, 					// WTA_k 
	// 	1,				 	// HARRIS_SCORE or FAST_SCORE 
	// 	31, 				// patchSize 
	// 	20	 				// fastThreshold 
	// 	); 

	detector->detect(img_1_input,keypoints_1_raw); 

	detector->detect(img_2_input,keypoints_2_raw); 

	// show feature points in each image ---------- 
	Mat image_1_show; 
	drawKeypoints(img_1_input,
				keypoints_1_raw,
				image_1_show,
				Scalar::all(-1),
				4	// draw rich points 
				); 
	namedWindow("keypoints_1", WINDOW_NORMAL); 
	imshow("keypoints_1",image_1_show); 
	waitKey(0); 

	Mat image_2_show; 
	drawKeypoints(img_2_input,
				keypoints_2_raw,
				image_2_show,
				Scalar::all(-1),
				4
				); 
	namedWindow("keypoints_2", WINDOW_NORMAL); 
	imshow("keypoints_2",image_2_show); 	
	waitKey(0); 

	detector->compute(img_1_input,keypoints_1_raw,descriptors_1); 
	detector->compute(img_2_input,keypoints_2_raw,descriptors_2); 
	// descriptors_1.convertTo(descriptors_1, CV_32F);
	// descriptors_2.convertTo(descriptors_2, CV_32F);

} 

void QuEst::MatchFeaturePoints(const Mat& descriptors_1, const Mat& descriptors_2, 
						vector<DMatch>& matches_select){

	// feature match, normal matcher ---------- 
	FlannBasedMatcher matcher; 
	vector<DMatch> matches_raw; 
	matcher.match(descriptors_1, descriptors_2, matches_raw); // cout<<"match numbers: "<<matches_raw.size()<<endl<<endl; 

	// sort distance, select 10 best ones ---------- 
	vector<double> distance; 
	for(int i=0;i<matches_raw.size();i++){
		distance.push_back(matches_raw[i].distance); 
	} 
	
	sort(distance.begin(), distance.end()); 

	for(int i=0;i<matches_raw.size();i++){
		if(matches_raw[i].distance == distance[0]) matches_select.push_back(matches_raw[i]); 
		if(matches_raw[i].distance == distance[1]) matches_select.push_back(matches_raw[i]); 
		if(matches_raw[i].distance == distance[2]) matches_select.push_back(matches_raw[i]); 
		if(matches_raw[i].distance == distance[3]) matches_select.push_back(matches_raw[i]); 
		if(matches_raw[i].distance == distance[4]) matches_select.push_back(matches_raw[i]); 
		if(matches_raw[i].distance == distance[5]) matches_select.push_back(matches_raw[i]); 
		if(matches_raw[i].distance == distance[6]) matches_select.push_back(matches_raw[i]); 
		if(matches_raw[i].distance == distance[7]) matches_select.push_back(matches_raw[i]); 
		if(matches_raw[i].distance == distance[8]) matches_select.push_back(matches_raw[i]); 
		if(matches_raw[i].distance == distance[9]) matches_select.push_back(matches_raw[i]); 
		if(matches_raw[i].distance == distance[10]) matches_select.push_back(matches_raw[i]); 
		if(matches_raw[i].distance == distance[11]) matches_select.push_back(matches_raw[i]); 
		if(matches_raw[i].distance == distance[12]) matches_select.push_back(matches_raw[i]); 
		if(matches_raw[i].distance == distance[13]) matches_select.push_back(matches_raw[i]); 
		if(matches_raw[i].distance == distance[14]) matches_select.push_back(matches_raw[i]); 
		if(matches_raw[i].distance == distance[15]) matches_select.push_back(matches_raw[i]); 
		if(matches_raw[i].distance == distance[16]) matches_select.push_back(matches_raw[i]); 
		if(matches_raw[i].distance == distance[17]) matches_select.push_back(matches_raw[i]); 
		if(matches_raw[i].distance == distance[18]) matches_select.push_back(matches_raw[i]); 
		if(matches_raw[i].distance == distance[19]) matches_select.push_back(matches_raw[i]); 
	} 

} 

// contain two parts: recover quaternion first, then get translation ---------- 
// m: (3 by 10) 
// n: (3 by 10) 
void QuEst::QuEst_RANSAC(const Matrix3Xd& x1, const Matrix3Xd& x2){

	int numPts = x1.cols(); 

	// normalize x1 and x2 ---------- 
	RowVectorXd x1n(1,numPts); 
	for(int i=0; i<numPts; i++){
		x1n(0,i) = abs(x1(0,i)) + abs(x1(1,i)) + abs(x1(2,i)); 
	} 
	Matrix3Xd x1_norm(3,numPts); 
	for(int i=0; i<numPts; i++){
		x1_norm(0,i) = x1(0,i)/x1n(0,i); 
		x1_norm(1,i) = x1(1,i)/x1n(0,i); 
		x1_norm(2,i) = x1(2,i)/x1n(0,i); 
	} 
// cout<<"x1_norm: "<<endl<<x1_norm<<endl<<endl; 

	RowVectorXd x2n(1,numPts); 
	for(int i=0; i<numPts; i++){
		x2n(0,i) = abs(x2(0,i)) + abs(x2(1,i)) + abs(x2(2,i)); 
	} 
	Matrix3Xd x2_norm(3,numPts); 
	for(int i=0; i<numPts; i++){
		x2_norm(0,i) = x2(0,i)/x2n(0,i); 
		x2_norm(1,i) = x2(1,i)/x2n(0,i); 
		x2_norm(2,i) = x2(2,i)/x2n(0,i); 
	} 
// cout<<"x2_norm: "<<endl<<x2_norm<<endl<<endl; 

	// RANSAC written by Yujie --------------------
	// RANSAC written by Yujie --------------------
	// RANSAC written by Yujie --------------------

	// formulate data ---------- 
	MatrixXd data(6,numPts); 
	for(int i=0; i<numPts; i++){
		data(0,i) = x1_norm(0,i); 
		data(1,i) = x1_norm(1,i); 
		data(2,i) = x1_norm(2,i); 
		data(3,i) = x2_norm(0,i); 
		data(4,i) = x2_norm(1,i); 
		data(5,i) = x2_norm(2,i); 
		// data(0,i) = x1(0,i); 
		// data(1,i) = x1(1,i); 
		// data(2,i) = x1(2,i); 
		// data(3,i) = x2(0,i); 
		// data(4,i) = x2(1,i); 
		// data(5,i) = x2(2,i); 
	}
// cout<<"data: "<<endl<<data<<endl<<endl; 

	// minimum samples for fittingfn 
	const int minimumSizeSamplesToFit = 6; 

	// distance threshold between data and model 
	const double distance_threshold = 1e-6; 

	// output model and inliers 
	VectorXd test_model(7); 
	VectorXd select_model(7), best_model(7); 
	vector<int> select_inliers, best_inliers; 

	// random indicies in range (0, minimumSizeSamplesToFit-1) 
	vector<int> ind; 
	for(int i=0; i<minimumSizeSamplesToFit; i++){
		ind.push_back(i); 
	}
	unsigned seed = chrono::system_clock::now().time_since_epoch().count(); 
	std::shuffle(ind.begin(), ind.end(), default_random_engine(seed)); 
// cout<<"ind: "; 
// for(int i=0; i<ind.size(); i++){
// 	cout<<ind.at(i)<<' '; 
// }
// cout<<endl<<endl; 

// high_resolution_clock::time_point t1 = high_resolution_clock::now(); // start time 

	// run RANSAC by Yujie ---------- 
	QuEst::ransac(&QuEst_fit, data, ind, test_model, 
		&QuEst_distance, data, test_model, distance_threshold, select_model, select_inliers, 
		&QuEst_degenerate, data, ind, 
		minimumSizeSamplesToFit, 
		best_model, 
		best_inliers); 

// high_resolution_clock::time_point t2 = high_resolution_clock::now(); // end time 
// auto duration = duration_cast<microseconds>(t2-t1).count(); // running time for RANSAC 
// cout<<"RANSAC takes: "<<duration<<" microseconds"<<endl<<endl; 

cout<<"best_model: "<<endl<<best_model<<endl<<endl; 

	double model_trans_x_sat; 		if(best_model(0)>5)	model_trans_x_sat=5; 		else if(best_model(0)<-5) model_trans_x_sat = -5; 		else model_trans_x_sat = best_model(0);  
	double model_trans_y_sat; 		if(best_model(1)>5)	model_trans_y_sat=5; 		else if(best_model(1)<-5) model_trans_y_sat = -5; 		else model_trans_y_sat = best_model(1);  
	double model_trans_z_sat; 		if(best_model(2)>5)	model_trans_z_sat=5; 		else if(best_model(2)<-5) model_trans_z_sat = -5; 		else model_trans_z_sat = best_model(2); 
	double model_rotation_w_sat; 	if(best_model(3)>1)	model_rotation_w_sat=1; 	else if(best_model(3)<-1) model_rotation_w_sat = -1; 	else model_rotation_w_sat = best_model(3);  
	double model_rotation_x_sat; 	if(best_model(4)>1)	model_rotation_x_sat=1; 	else if(best_model(4)<-1) model_rotation_x_sat = -1; 	else model_rotation_x_sat = best_model(4);  
	double model_rotation_y_sat; 	if(best_model(5)>1)	model_rotation_y_sat=1; 	else if(best_model(5)<-1) model_rotation_y_sat = -1; 	else model_rotation_y_sat = best_model(5);  
	double model_rotation_z_sat; 	if(best_model(6)>1)	model_rotation_z_sat=1; 	else if(best_model(6)<-1) model_rotation_z_sat = -1; 	else model_rotation_z_sat = best_model(6);  

	quest_msg.translation.x = model_trans_x_sat; 
	quest_msg.translation.y = model_trans_y_sat; 
	quest_msg.translation.z = model_trans_z_sat; 
	quest_msg.rotation.w = model_rotation_w_sat; 
	quest_msg.rotation.x = model_rotation_x_sat; 
	quest_msg.rotation.y = model_rotation_y_sat; 
	quest_msg.rotation.z = model_rotation_z_sat; 

	pub.publish(quest_msg); 

// 	// MRPT version of RANSAC, deprecated --------------------
// 	// MRPT version of RANSAC, deprecated --------------------
// 	// MRPT version of RANSAC, deprecated --------------------
// 	// stack x1_norm and x2_norm together ---------- 
// 	CMatrixDouble data(6, numPts); 
// 	for(int i=0; i<numPts; i++){
// 		data(0,i) = x1_norm(0,i); 
// 		data(1,i) = x1_norm(1,i); 
// 		data(2,i) = x1_norm(2,i); 
// 		data(3,i) = x2_norm(0,i); 
// 		data(4,i) = x2_norm(1,i); 
// 		data(5,i) = x2_norm(2,i); 
// 	} 

// 	// run RANSAC ---------- 
// 	CMatrixDouble best_model; 
// 	std::vector<size_t> best_inliers; 
// 	const double DIST_THRESHOLD = 0.0001;  // 1e-4
// 	const unsigned int minimumSizeSamplesToFit = 20; 
// 	const double prob_good_sample = 0.8; 
// 	const size_t maxIter = 20; 

// 	math::RANSAC myransac; 
// 	myransac.execute(data, 
// 		QuEst_fit, 
// 		QuEst_distance,
// 		QuEst_degenerate, 
// 		DIST_THRESHOLD,
// 		minimumSizeSamplesToFit, 
// 		best_inliers, 
// 		best_model, 
// 		prob_good_sample,
// 		maxIter); 

// // cout<<"best_model: "<<endl<<best_model<<endl<<endl; 

} 

bool QuEst::ransac(void (*QuEst_fit)(const MatrixXd&, const vector<int>&, VectorXd&), 
				const MatrixXd& data, const vector<int>& ind, VectorXd& test_model, 
			void (*QuEst_distance)(const MatrixXd&, const VectorXd&, const double, VectorXd&, vector<int>&), 
				const MatrixXd& data_1, const VectorXd& ind_1, const double distance_threshold, VectorXd& select_model, vector<int>& select_inliers, 
			bool (*QuEst_degenerate)(const MatrixXd&, const vector<int>&),  
				const MatrixXd& data_2, const vector<int>& ind_2, 
			const int minimumSizeSamplesToFit, 
			VectorXd& best_model, 
			vector<int>& best_inliers){	

	const int Npts = data.cols(); // number of points in data 

	const double p = 0.99; // desired probability of choosing at least one sample free from outliers 

	const int maxDataTrials = 100; // max number to select non-degenerate data set 

	const int maxIter = 1000; // max number of iterations 
	
	int trialcount = 0; 

	int bestscore = 0; 

	double N = 1; // dummy initialisation for number of trials 

	while (N > trialcount){

		bool degenerate = true; 
		int count = 1; 

		while(degenerate){

			// test that these points are not a degenerate configuration 
			degenerate = QuEst_degenerate(data_2, ind_2); 

			if(!degenerate){
				// fit model to this random selection of data points 

				vector<int> ind_degen; 
				for(int i=0; i<Npts; i++){
					ind_degen.push_back(i); 
				}
				unsigned seed_degen = chrono::system_clock::now().time_since_epoch().count(); 
				std::shuffle(ind_degen.begin(), ind_degen.end(), default_random_engine(seed_degen)); 

				vector<int> ind_degen_select; 
				for(int i=0; i<minimumSizeSamplesToFit; i++){
					ind_degen_select.push_back(ind_degen[i]); 
				} 
// cout<<"index: "; 
// for(int i=0; i<ind_degen_select.size(); i++){
// 	cout<<ind_degen_select.at(i)<<" "; 
// } 
// cout<<endl<<endl; 

// high_resolution_clock::time_point t1 = high_resolution_clock::now(); 
				(*QuEst_fit)(data, ind_degen_select, test_model); // cout<<"test_model: "<<endl<<test_model<<endl<<endl; 
// high_resolution_clock::time_point t2 = high_resolution_clock::now(); 
// auto duration = duration_cast<microseconds>(t2-t1).count(); 
// cout<<"QuEst takes:  "<<duration<<" microseconds"<<endl<<endl; 

			}

			if(++count > maxDataTrials){
				// safeguard against being stuck in this loop forever 
				cout<<"Unable to select a nondegenerate data set"<<endl<<endl; 
				break; 
			}

		}

		// once we are out here, we should have some kind of model
		// evaluate distances between points and model 
		// returning the indices of elements that are inliers 
		(*QuEst_distance)(data, test_model, distance_threshold, select_model, select_inliers); 

		// find the number of inliers to this model 
		int ninliers = select_inliers.size(); // cout<<"ninliers: "<<ninliers<<endl<<endl; 

		if(ninliers > bestscore){

			bestscore = ninliers; 
			best_inliers = select_inliers; 
			best_model = select_model; 

			// update estimate of N, the number of trials,   
			// to ensure we pick with probability p, a data set with no outliers 
			double fracinliers = ninliers / static_cast<double>(Npts); 
			double pNoOutliers = 1 - pow(fracinliers, static_cast<double>(minimumSizeSamplesToFit)); 

			// avoid division by -Inf
			pNoOutliers = std::max(std::numeric_limits<double>::epsilon(), pNoOutliers); 
			// avoid division by 0 
			pNoOutliers = std::min(1.0-std::numeric_limits<double>::epsilon(), pNoOutliers); 

			// update N 
			N = log(1-p) / log(pNoOutliers); // cout<<"N: "<<N<<endl<<endl; 

		} 

		trialcount = trialcount + 1; // cout<<"trialcount: "<<trialcount<<endl<<endl; 

		if(trialcount > maxIter){
			cout<<"RANSAC reached the maximum number of trials. "<<endl<<endl; 
			break; 
		}

	} 
cout<<"total trialcount: "<<trialcount<<endl<<endl; 

		if(best_model.rows() > 0){
			return true; 
		} 
		else{
			return false; 
		} 

} 

bool QuEst::QuEst_degenerate(const MatrixXd& data, const vector<int>& ind){
	return false; 
}

void QuEst::QuEst_fit(const MatrixXd& allData, const vector<int>& useIndices, VectorXd& test_model){

	int numPts = useIndices.size(); 

	// take allData out, into x1 and x2 ---------- 
	Matrix3Xd x1(3,numPts); 
	for(int i=0; i<numPts; i++){
		x1(0,i) = allData(0,useIndices[i]); 
		x1(1,i) = allData(1,useIndices[i]); 
		x1(2,i) = allData(2,useIndices[i]); 
	} 
	Matrix3Xd x2(3,numPts); 
	for(int i=0; i<numPts; i++){
		x2(0,i) = allData(3,useIndices[i]); 
		x2(1,i) = allData(4,useIndices[i]); 
		x2(2,i) = allData(5,useIndices[i]); 
	} 
// cout<<"x1: "<<endl<<x1<<endl<<endl; 
// cout<<"x2: "<<endl<<x2<<endl<<endl; 

	// take first 5 points of x1 and x2, feeding into QuEst ---------- 
	Matrix3Xd x1_5pts(3,5); 
	for(int i=0; i<5; i++){
		x1_5pts(0,i) = x1(0,i); 
		x1_5pts(1,i) = x1(1,i); 
		x1_5pts(2,i) = x1(2,i); 
	} 
	Matrix3Xd x2_5pts(3,5); 
	for(int i=0; i<5; i++){
		x2_5pts(0,i) = x2(0,i); 
		x2_5pts(1,i) = x2(1,i); 
		x2_5pts(2,i) = x2(2,i); 
	} 

	// run QuEst algorithm, get 35 candidates ---------- 
	Matrix4Xd Q(4,35); 
	// Matrix3Xd T(3,35); 
	QuEst_Ver1_1(Q,x1_5pts,x2_5pts); 
// cout<<"Q transpose: "<<endl<<Q.transpose()<<endl<<endl; 

	// score function, pick the best estimated pose solution ----------
	RowVectorXd res = RowVectorXd::Zero(35); 
	QuatResidue(res,x1,x2,Q); 
// cout<<"res: "<<endl<<res<<endl<<endl; 

	int mIdx = 0; 
	for(int i=1; i<35; i++){
		if(res(0,i) < res(0,mIdx))
			mIdx = i; 
	} 
// cout<<"mIdx: "<<endl<<mIdx<<endl<<endl; 
// cout<<"Quaternion: "<<endl<<Q(0,mIdx)<<"   "<<Q(1,mIdx)<<"   "<<Q(2,mIdx)<<"   "<<Q(3,mIdx)<<endl<<endl; 

	Vector4d Q_select; 
	Q_select(0,0) = Q(0,mIdx); 
	Q_select(1,0) = Q(1,mIdx); 
	Q_select(2,0) = Q(2,mIdx); 
	Q_select(3,0) = Q(3,mIdx); 

	Vector3d T; 
	FindTrans(T,x1_5pts,x2_5pts,Q_select); 
// cout<<"Translation: "<<endl<<T(0,0)<<"   "<<T(1,0)<<"   "<<T(2,0)<<endl<<endl; 

	double T_norm = sqrt(T(0,0)*T(0,0) + T(1,0)*T(1,0) + T(2,0)*T(2,0)); 

	// test_model(0) = T(0,0) / T_norm; 
	// test_model(1) = T(1,0) / T_norm; 
	// test_model(2) = T(2,0) / T_norm; 
	test_model(0) = T(0,0); 
	test_model(1) = T(1,0); 
	test_model(2) = T(2,0); 
	test_model(3) = Q_select(0,0); 
	test_model(4) = Q_select(1,0); 
	test_model(5) = Q_select(2,0); 
	test_model(6) = Q_select(3,0); 

} 

void QuEst::QuEst_distance(const MatrixXd& data, const VectorXd& test_model, const double distance_threshold, 
	VectorXd& select_model, vector<int>& select_inliers){

	int numPts = data.cols(); 

	// extract all feature points 
	Matrix3Xd x1(3,numPts); 
	for(int i=0; i<numPts; i++){
		x1(0,i) = data(0,i); 
		x1(1,i) = data(1,i); 
		x1(2,i) = data(2,i); 
	} 
	Matrix3Xd x2(3,numPts); 
	for(int i=0; i<numPts; i++){
		x2(0,i) = data(3,i); 
		x2(1,i) = data(4,i); 
		x2(2,i) = data(5,i); 
	} 
// cout<<"in dist function, x1: "<<endl<<x1<<endl; 
// cout<<"in dist function, x2: "<<endl<<x2<<endl<<endl; 
	
	// rotation matrix 
	Vector4d q(4,1); 
	q(0,0) = test_model(3,0); 
	q(1,0) = test_model(4,0); 
	q(2,0) = test_model(5,0); 
	q(3,0) = test_model(6,0); 
	Matrix3d R; 
	Q2R_3by3(R,q); 
// cout<<"R: "<<endl<<R<<endl<<endl; 

	// skew matrix 
	double t_norm = sqrt(test_model(0,0)*test_model(0,0)+test_model(1,0)*test_model(1,0)+test_model(2,0)*test_model(2,0)); 

	Vector3d t(3,1); 
	t(0,0) = test_model(0,0) / t_norm; 
	t(1,0) = test_model(1,0) / t_norm; 
	t(2,0) = test_model(2,0) / t_norm; 

	Matrix3d Tx; 
	Tx(0,0) = 0; 
	Tx(0,1) = -t(2,0); 
	Tx(0,2) = t(1,0); 
	Tx(1,0) = t(2,0); 
	Tx(1,1) = 0; 
	Tx(1,2) = -t(0,0); 
	Tx(2,0) = -t(1,0); 
	Tx(2,1) = t(0,0); 
	Tx(2,2) = 0; 
// cout<<"Tx: "<<endl<<Tx<<endl<<endl; 

	// fundamental matrix ---------- 
	Matrix3d F(3,3); 
	F = Tx * R; 
// cout<<"F: "<<endl<<F<<endl<<endl; 

	RowVectorXd x2tFx1 = RowVectorXd::Zero(numPts); 
// cout<<x2tFx1<<endl; 
	for(int i=0; i<numPts; i++){
		x2tFx1(0,i)=(x2(0,i)*F(0,0)+x2(1,i)*F(1,0)+x2(2,i)*F(2,0))*x1(0,i)+(x2(0,i)*F(0,1)+x2(1,i)*F(1,1)+x2(2,i)*F(2,1))*x1(1,i)+(x2(0,i)*F(0,2)+x2(1,i)*F(1,2)+x2(2,i)*F(2,2))*x1(2,i); 
	} 
// cout<<"x2tFx1: "<<endl<<x2tFx1<<endl<<endl; 
	
	// evaluate distance ---------- 
	Matrix3Xd Fx1(3,numPts); 
	Fx1 = F * x1; 
// cout<<"Fx1: "<<endl<<Fx1<<endl<<endl; 

	Matrix3Xd Ftx2(3,numPts); 
	Ftx2 = F.transpose() * x2; 
// cout<<"Ftx2: "<<endl<<Ftx2<<endl<<endl; 

	select_inliers.clear(); 

	for(int i=0; i<numPts; i++){
		double d = x2tFx1(0,i)*x2tFx1(0,i)/(Fx1(0,i)*Fx1(0,i)+Fx1(1,i)*Fx1(1,i)+Ftx2(0,i)*Ftx2(0,i)+Ftx2(1,i)*Ftx2(1,i)); 
// cout<<"d: "<<d<<endl; 

		if(abs(d)<distance_threshold) 
			select_inliers.push_back(i); 
	} 
// cout<<endl; 
// cout<<"select_inliers: "<<endl<<select_inliers.size()<<endl<<endl; 

	select_model = test_model; 

} 

// bool QuEst::QuEst_degenerate(const CMatrixDouble& allData, const std::vector<size_t>& useIndices){
// 	return false; 
// } 

// void QuEst::QuEst_fit(const CMatrixDouble& allData, const std::vector<size_t>& useIndices, 
// 	vector<CMatrixDouble>& fitModels){

// // cout<<"fit function start ---------------------------------------"<<endl<<endl; 
// // cout<<"useIndices: "<<endl; 
// // for(int i=0; i<useIndices.size(); i++){
// // 	cout<<useIndices.at(i)<<endl; 
// // }
// // cout<<endl; 

// 	int numPts = 20; 

// 	// take allData out, into x1 and x2 ---------- 
// 	Matrix3Xd x1(3,numPts); 
// 	for(int i=0; i<numPts; i++){
// 		x1(0,i) = allData(0,useIndices[i]); 
// 		x1(1,i) = allData(1,useIndices[i]); 
// 		x1(2,i) = allData(2,useIndices[i]); 
// 	} 
// 	Matrix3Xd x2(3,numPts); 
// 	for(int i=0; i<numPts; i++){
// 		x2(0,i) = allData(3,useIndices[i]); 
// 		x2(1,i) = allData(4,useIndices[i]); 
// 		x2(2,i) = allData(5,useIndices[i]); 
// 	} 
// // cout<<"x1: "<<endl<<x1<<endl<<endl; 
// // cout<<"x2: "<<endl<<x2<<endl<<endl; 

// 	// take first 5 points of x1 and x2, feeding into QuEst ---------- 
// 	Matrix3Xd x1_5pts(3,5); 
// 	for(int i=0; i<5; i++){
// 		x1_5pts(0,i) = x1(0,i); 
// 		x1_5pts(1,i) = x1(1,i); 
// 		x1_5pts(2,i) = x1(2,i); 
// 	} 
// 	Matrix3Xd x2_5pts(3,5); 
// 	for(int i=0; i<5; i++){
// 		x2_5pts(0,i) = x2(0,i); 
// 		x2_5pts(1,i) = x2(1,i); 
// 		x2_5pts(2,i) = x2(2,i); 
// 	} 

// 	// run QuEst algorithm, get 35 candidates ---------- 
// 	Matrix4Xd Q(4,35); 
// 	// Matrix3Xd T(3,35); 
// 	QuEst_Ver1_1(Q,x1_5pts,x2_5pts); 
// // cout<<"Q transpose: "<<endl<<Q.transpose()<<endl<<endl; 
// // cout<<"T transpose: "<<endl<<T.transpose()<<endl<<endl; 

// 	// score function, pick the best estimated pose solution ----------
// 	RowVectorXd res = RowVectorXd::Zero(35); 
// 	QuatResidue(res,x1,x2,Q); 
// // cout<<"res: "<<endl<<res<<endl<<endl; 

// 	int mIdx = 0; 
// 	for(int i=1; i<35; i++){
// 		if(res(0,i) < res(0,mIdx))
// 			mIdx = i; 
// 	} 
// // cout<<"mIdx: "<<endl<<mIdx<<endl<<endl;  
// // cout<<"Quaternion: "<<endl<<Q(0,mIdx)<<"   "<<Q(1,mIdx)<<"   "<<Q(2,mIdx)<<"   "<<Q(3,mIdx)<<endl<<endl; 
// // cout<<"Translation: "<<endl<<T(0,mIdx)<<endl<<T(1,mIdx)<<endl<<T(2,mIdx)<<endl<<endl; 

// 	Vector4d Q_select; 
// 	Q_select(0,0) = Q(0,mIdx); 
// 	Q_select(1,0) = Q(1,mIdx); 
// 	Q_select(2,0) = Q(2,mIdx); 
// 	Q_select(3,0) = Q(3,mIdx); 

// 	Vector3d T; 
// 	FindTrans(T,x2_5pts,x1_5pts,Q_select); 

// 	fitModels.resize(1); 
	
// 	CMatrixDouble& M = fitModels[0]; 
// 	M.setSize(7,1); 
// 	M(0,0) = T(0,0); 
// 	M(1,0) = T(1,0); 
// 	M(2,0) = T(2,0); 
// 	M(3,0) = Q_select(0,0); 
// 	M(4,0) = Q_select(1,0); 
// 	M(5,0) = Q_select(2,0); 
// 	M(6,0) = Q_select(3,0); 

// // cout<<"fit function end -------------------------------"<<endl<<endl; 

// } 

// void QuEst::QuEst_distance(const CMatrixDouble& allData, const vector<CMatrixDouble>& testModels,
// 	const double distanceThreshold,unsigned int& out_bestModelIndex,std::vector<size_t>& out_inlierIndices){

// 	out_bestModelIndex = 0; 
// 	const CMatrixDouble& M = testModels[0]; 

// 	int numPts = allData.cols(); 

// 	// extract all feature points 
// 	Matrix3Xd x1(3,numPts); 
// 	for(int i=0; i<numPts; i++){
// 		x1(0,i) = allData(0,i); 
// 		x1(1,i) = allData(1,i); 
// 		x1(2,i) = allData(2,i); 
// 	} 
// 	Matrix3Xd x2(3,numPts); 
// 	for(int i=0; i<numPts; i++){
// 		x2(0,i) = allData(3,i); 
// 		x2(1,i) = allData(4,i); 
// 		x2(2,i) = allData(5,i); 
// 	} 
// // cout<<"in dist function, x1: "<<endl<<x1<<endl; 
// // cout<<"in dist function, x2: "<<endl<<x2<<endl<<endl; 
	
// 	// rotation matrix 
// 	Vector4d q(4,1); 
// 	q(0,0) = M(3,0); 
// 	q(1,0) = M(4,0); 
// 	q(2,0) = M(5,0); 
// 	q(3,0) = M(6,0); 
// 	Matrix3d R; 
// 	Q2R_3by3(R,q); 

// 	// skew matrix 
// 	double t_norm = sqrt(M(0,0)*M(0,0)+M(1,0)*M(1,0)+M(2,0)*M(2,0)); 

// 	Vector3d t(3,1); 
// 	t(0,0) = M(0,0) / t_norm; 
// 	t(1,0) = M(1,0) / t_norm; 
// 	t(2,0) = M(2,0) / t_norm; 

// 	Matrix3d Tx; 
// 	Tx(0,0) = 0; 
// 	Tx(0,1) = -t(2,0); 
// 	Tx(0,2) = t(1,0); 
// 	Tx(1,0) = t(2,0); 
// 	Tx(1,1) = 0; 
// 	Tx(1,2) = -t(0,0); 
// 	Tx(2,0) = -t(1,0); 
// 	Tx(2,1) = t(0,0); 
// 	Tx(2,2) = 0; 

// 	// fundamental matrix ---------- 
// 	Matrix3d F(3,3); 
// 	F = Tx * R; 
// // cout<<"F: "<<endl<<F<<endl<<endl; 

// 	RowVectorXd x2tFx1 = RowVectorXd::Zero(numPts); 
// // cout<<x2tFx1<<endl; 
// 	for(int i=0; i<numPts; i++){
// 		x2tFx1(0,i)=(x2(0,i)*F(0,0)+x2(1,i)*F(1,0)+x2(2,i)*F(2,0))*x1(0,i)+(x2(0,i)*F(0,1)+x2(1,i)*F(1,1)+x2(2,i)*F(2,1))*x1(1,i)+(x2(0,i)*F(0,2)+x2(1,i)*F(1,2)+x2(2,i)*F(2,2))*x1(2,i); 
// 	} 
	
// 	// evaluate distance ---------- 
// 	Matrix3Xd Fx1(3,numPts); 
// 	Fx1 = F * x1; 

// 	Matrix3Xd Ftx2(3,numPts); 
// 	Ftx2 = F.transpose() * x2; 

// 	out_inlierIndices.clear(); 
// 	out_inlierIndices.reserve(numPts); 	
// 	for(int i=0; i<numPts; i++){
// 		double d = x2tFx1(0,i)*x2tFx1(0,i)/(Fx1(0,i)*Fx1(0,i)+Fx1(1,i)*Fx1(1,i)+Ftx2(0,i)*Ftx2(0,i)+Ftx2(1,i)*Ftx2(1,i)); 
// // cout<<"d: "<<d<<endl; 

// 		if(abs(d)<distanceThreshold) 
// 			out_inlierIndices.push_back(i); 
// 	} 
// // cout<<endl; 
// // cout<<"out_inlierIndices: "<<endl<<out_inlierIndices.size()<<endl<<endl; 

// }

// contain two parts: recover quaternion and recover translation 
// void QuEst::QuEst_Ver1_1(Matrix4Xd& Q, Matrix3Xd& T, const Matrix3Xd& m, const Matrix3Xd& n){
void QuEst::QuEst_Ver1_1(Matrix4Xd& Q, const Matrix3Xd& m, const Matrix3Xd& n){

	QuEst::QuEst_5Pt_Ver5_2(Q,m,n); 
	// QuEst::FindTrans(T,m,n,Q); 

}


// recover translation, 3 by 35 ----------
void QuEst::FindTrans(Vector3d& T, const Matrix3Xd& m, const Matrix3Xd& n, const Vector4d& Q){

// cout<<"Q: "<<endl<<Q<<endl<<endl; 
// cout<<"m: "<<endl<<m<<endl<<endl; 
// cout<<"n: "<<endl<<n<<endl<<endl; 

	int numCols = Q.cols(); 

	MatrixXd R = MatrixXd::Zero(9, numCols); 
	QuEst::Q2R(R,Q); // convert quaternion into rotation matrix 
// cout<<"R transpose: "<<endl<<R.transpose()<<endl<<endl; 

	int numPts = m.cols(); 
	int numInp = R.cols(); 

	for(int k=0; k<numInp; k++){

		MatrixXd C = MatrixXd::Zero(3*numPts, 2*numPts+3); 

		for(int i=1; i<=numPts; i++){
			C((i-1)*3,  0) = 1; 
			C((i-1)*3+1,1) = 1; 
			C((i-1)*3+2,2) = 1; 

			C((i-1)*3,  (i-1)*2+3) = R(0,k)*m(0,i-1) + R(1,k)*m(1,i-1) + R(2,k)*m(2,i-1); 
			C((i-1)*3+1,(i-1)*2+3) = R(3,k)*m(0,i-1) + R(4,k)*m(1,i-1) + R(5,k)*m(2,i-1); 
			C((i-1)*3+2,(i-1)*2+3) = R(6,k)*m(0,i-1) + R(7,k)*m(1,i-1) + R(8,k)*m(2,i-1); 

			C((i-1)*3,  (i-1)*2+4) = -n(0,i-1); 
			C((i-1)*3+1,(i-1)*2+4) = -n(1,i-1); 
			C((i-1)*3+2,(i-1)*2+4) = -n(2,i-1); 
		} 
cout<<"C: "<<endl<<C<<endl<<endl; 

		// use SVD to find singular vectors 
		// BDCSVD<MatrixXd> svd(C, ComputeFullV); 
		JacobiSVD<MatrixXd> svd(C, ComputeThinV); 
		MatrixXd N = svd.matrixV(); 
cout<<"N: "<<endl<<N<<endl<<endl; 

// 		// start armadillo 	----------------------------------------

// 		arma::Mat<double> C_arma(3*numPts, 2*numPts+3); 
// 		for(int i=0;i<3*numPts;i++){
// 			for(int j=0;j<2*numPts+3;j++){
// 				C_arma(i,j) = C(i,j); 
// 			}
// 		}		
// // cout<<"C_arma: "<<endl<<C_arma<<endl<<endl; 

// 		arma::mat U_arma; 
// 		arma::vec s_arma; 
// 		arma::mat N_arma; 
// 		arma::svd_econ(U_arma,s_arma,N_arma,C_arma); 

// 		MatrixXd N(13,13); 
// 		for(int i=0; i<13; i++){
// 			for(int j=0; j<13; j++){
// 				N(i,j) = N_arma(i,j); 
// 			} 
// 		} 		

// 		// end armadillo 	----------------------------------------

		// adjust the sign 
		int numPos=0, numNeg=0; 
		for(int i=0;i<2*numPts;i++){
			if(N(i+3,2*numPts+2)>0) numPos++; 
			if(N(i+3,2*numPts+2)<0) numNeg++; 
		}
// cout<<"numPos: "<<numPos<<"   "<<"numNeg: "<<numNeg<<endl<<endl; 

		if(numPos<numNeg){
			T(0,k) = -N(0,2*numPts+2); 
			T(1,k) = -N(1,2*numPts+2); 
			T(2,k) = -N(2,2*numPts+2); 
		}
		else{
			T(0,k) = N(0,2*numPts+2); 
			T(1,k) = N(1,2*numPts+2); 
			T(2,k) = N(2,2*numPts+2); 
		} 
// cout<<"T transpose: "<<endl<<T.transpose()<<endl<<endl; 

	}

} 

void QuEst::QuatResidue(RowVectorXd& residu, const Matrix3Xd& m1, const Matrix3Xd& m2, const Matrix4Xd& qSol){

	int numPts = m1.cols(); 

	int numEq = numPts*(numPts-1)*(numPts-2)/6; 
	
	MatrixXd C0(numEq,35); 
	CoefsVer_3_1_1(C0,m1,m2); // coefficient matrix such that C * x = c 
// cout<<"C0: "<<endl<<C0<<endl<<endl; 

	MatrixXd xVec(35,35); 
	for(int i=0; i<35; i++){
		xVec(0,i) = qSol(0,i) * qSol(0,i) * qSol(0,i) * qSol(0,i); 
		xVec(1,i) = qSol(0,i) * qSol(0,i) * qSol(0,i) * qSol(1,i); 
		xVec(2,i) = qSol(0,i) * qSol(0,i) * qSol(1,i) * qSol(1,i); 
		xVec(3,i) = qSol(0,i) * qSol(1,i) * qSol(1,i) * qSol(1,i); 
		xVec(4,i) = qSol(1,i) * qSol(1,i) * qSol(1,i) * qSol(1,i); 
		xVec(5,i) = qSol(0,i) * qSol(0,i) * qSol(0,i) * qSol(2,i); 
		xVec(6,i) = qSol(0,i) * qSol(0,i) * qSol(1,i) * qSol(2,i); 
		xVec(7,i) = qSol(0,i) * qSol(1,i) * qSol(1,i) * qSol(2,i); 
		xVec(8,i) = qSol(1,i) * qSol(1,i) * qSol(1,i) * qSol(2,i); 
		xVec(9,i) = qSol(0,i) * qSol(0,i) * qSol(2,i) * qSol(2,i); 
		xVec(10,i) = qSol(0,i) * qSol(1,i) * qSol(2,i) * qSol(2,i); 
		xVec(11,i) = qSol(1,i) * qSol(1,i) * qSol(2,i) * qSol(2,i); 
		xVec(12,i) = qSol(0,i) * qSol(2,i) * qSol(2,i) * qSol(2,i); 
		xVec(13,i) = qSol(1,i) * qSol(2,i) * qSol(2,i) * qSol(2,i); 
		xVec(14,i) = qSol(2,i) * qSol(2,i) * qSol(2,i) * qSol(2,i); 
		xVec(15,i) = qSol(0,i) * qSol(0,i) * qSol(0,i) * qSol(3,i); 
		xVec(16,i) = qSol(0,i) * qSol(0,i) * qSol(1,i) * qSol(3,i); 
		xVec(17,i) = qSol(0,i) * qSol(1,i) * qSol(1,i) * qSol(3,i); 
		xVec(18,i) = qSol(1,i) * qSol(1,i) * qSol(1,i) * qSol(3,i); 
		xVec(19,i) = qSol(0,i) * qSol(0,i) * qSol(2,i) * qSol(3,i); 
		xVec(20,i) = qSol(0,i) * qSol(1,i) * qSol(2,i) * qSol(3,i); 
		xVec(21,i) = qSol(1,i) * qSol(1,i) * qSol(2,i) * qSol(3,i); 
		xVec(22,i) = qSol(0,i) * qSol(2,i) * qSol(2,i) * qSol(3,i); 
		xVec(23,i) = qSol(1,i) * qSol(2,i) * qSol(2,i) * qSol(3,i); 
		xVec(24,i) = qSol(2,i) * qSol(2,i) * qSol(2,i) * qSol(3,i); 
		xVec(25,i) = qSol(0,i) * qSol(0,i) * qSol(3,i) * qSol(3,i); 
		xVec(26,i) = qSol(0,i) * qSol(1,i) * qSol(3,i) * qSol(3,i); 
		xVec(27,i) = qSol(1,i) * qSol(1,i) * qSol(3,i) * qSol(3,i); 
		xVec(28,i) = qSol(0,i) * qSol(2,i) * qSol(3,i) * qSol(3,i); 
		xVec(29,i) = qSol(1,i) * qSol(2,i) * qSol(3,i) * qSol(3,i); 
		xVec(30,i) = qSol(2,i) * qSol(2,i) * qSol(3,i) * qSol(3,i); 
		xVec(31,i) = qSol(0,i) * qSol(3,i) * qSol(3,i) * qSol(3,i); 
		xVec(32,i) = qSol(1,i) * qSol(3,i) * qSol(3,i) * qSol(3,i); 
		xVec(33,i) = qSol(2,i) * qSol(3,i) * qSol(3,i) * qSol(3,i); 
		xVec(34,i) = qSol(3,i) * qSol(3,i) * qSol(3,i) * qSol(3,i); 
	} 
// cout<<"xVec: "<<endl<<xVec<<endl<<endl; 

	MatrixXd residuMat(numEq,35); 
	residuMat = C0 * xVec; 
// cout<<"residuMat: "<<endl<<residuMat<<endl<<endl; 

	for(int i=0; i<35; i++){
		for(int j=0; j<numEq; j++){
			residu(0,i) = residu(0,i) + abs(residuMat(j,i)); 
		}
	} 
// cout<<"residu: "<<endl<<residu<<endl<<endl; 

}


// recover quaternion, 4 by 35 ---------
void QuEst::QuEst_5Pt_Ver5_2(Matrix4Xd& Q ,const Matrix3Xd& m, const Matrix3Xd& n){

// cout<<"QuEst_5Pt_Ver5_2 start -------------------------"<<endl<<endl; 

	int numPts = m.cols(); 

	Matrix4Xi Idx(4,35); 
	Idx << 1,2,5,11,21,3,6,12,22,8,14,24,17,27,31,4,7,13,23,9,15,25,18,28,32,10,16,26,19,29,33,20,30,34,35, 
		2,5,11,21,36,6,12,22,37,14,24,39,27,42,46,7,13,23,38,15,25,40,28,43,47,16,26,41,29,44,48,30,45,49,50, 
		3,6,12,22,37,8,14,24,39,17,27,42,31,46,51,9,15,25,40,18,28,43,32,47,52,19,29,44,33,48,53,34,49,54,55, 
		4,7,13,23,38,9,15,25,40,18,28,43,32,47,52,10,16,26,41,19,29,44,33,48,53,20,30,45,34,49,54,35,50,55,56; 

	RowVectorXi idx_w(35); 
	idx_w << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35; 

	RowVectorXi idx_w0(21); 
	idx_w0 << 36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56; 

	MatrixX4i Idx1(20,4); 
	Idx1 << 1,2,3,4, 
		2,5,6,7,
		3,6,8,9,
		4,7,9,10,
		5,11,12,13,
		6,12,14,15,
		7,13,15,16,
		8,14,17,18,
		9,15,18,19,
		10,16,19,20,
		11,21,22,23,
		12,22,24,25,
		13,23,25,26,
		14,24,27,28,
		15,25,28,29,
		16,26,29,30,
		17,27,31,32,
		18,28,32,33,
		19,29,33,34,
		20,30,34,35; 

	MatrixX4i Idx2(15,4); 
	Idx2 <<	21,1,2,3, 
			22,2,4,5, 
			23,3,5,6, 
			24,4,7,8, 
			25,5,8,9, 
			26,6,9,10, 
			27,7,11,12, 
			28,8,12,13, 
			29,9,13,14, 
			30,10,14,15, 
			31,11,16,17, 
			32,12,17,18, 
			33,13,18,19, 
			34,14,19,20, 
			35,15,20,21; 

	MatrixXd Bx = MatrixXd::Zero(35,35); 

	int numEq = numPts*(numPts-1)*(numPts-2)/6; 

	MatrixXd Cf(numEq,35); // coefficient matrix such that Cf * V = 0 
	CoefsVer_3_1_1(Cf,m,n); 
// cout<<"Cf: "<<endl<<Cf<<endl<<endl<<endl; 

	MatrixXd A = MatrixXd::Zero(4*numEq, 56); // coefficient matrix such that A * X = 0 
	for(int i=0; i<numEq; i++){
		for(int j=0; j<35; j++){
			int idx_0 = Idx(0,j); 
			A(i,idx_0-1) = Cf(i,j); 
			
			int idx_1 = Idx(1,j); 
			A(i+numEq,idx_1-1) = Cf(i,j); 
			
			int idx_2 = Idx(2,j); 
			A(i+numEq*2,idx_2-1) = Cf(i,j); 
			
			int idx_3 = Idx(3,j); 
			A(i+numEq*3,idx_3-1) = Cf(i,j); 
		}
	} 
// cout<<"A: "<<endl<<A<<endl<<endl; 

	MatrixXd A1(4*numEq,35); // split A into A1 and A2, A1 contains term w 
	for(int i=0; i<40; i++){
		for(int j=0; j<35; j++){
			A1(i,j) = A(i,j); 
		}
	} 
// cout<<"A1: "<<endl<<A1<<endl<<endl; 

	MatrixXd A2(4*numEq,21); // A2 doesn't contains term w 
	for(int i=0; i<40; i++){
		for(int j=0; j<21; j++){
			A2(i,j) = A(i,j+35); 
		}
	} 
// cout<<"A2: "<<endl<<A2<<endl<<endl; 

	// MatrixXd A2_pseudo = A2.completeOrthogonalDecomposition().pseudoInverse(); 
	// MatrixXd Bbar = -A2_pseudo*A1; 
	MatrixXd Bbar = -A2.completeOrthogonalDecomposition().solve(A1); 
// cout<<"Bbar: "<<endl<<Bbar<<endl<<endl; 

	for(int i=0; i<20; i++){
		Bx(Idx1(i,0)-1,Idx1(i,1)-1) = 1; 
	} 
	for(int i=0; i<15; i++){
		for(int j=0; j<35; j++){
			Bx(20+i,j) = Bbar(i,j); 
		}
	} 
// cout<<"Bx: "<<endl<<Bx<<endl<<endl; 

// 	// Eigen version of eigensolver, deprecated ----------
// 	EigenSolver<MatrixXd> es; 
// 	es.compute(Bx, true); 
// 	MatrixXcd Ve = es.eigenvectors(); 
// // cout<<"Ve: "<<endl<<Ve<<endl<<endl; 

// start armadillo ------------------------------------------------------------ 
	arma::Mat<double> Bx_arma(35,35); 
	for(int i=0;i<35;i++){
		for(int j=0;j<35;j++){
			Bx_arma(i,j) = Bx(i,j); 
		}
	}
// cout<<"Bx in arma: "<<endl<<Bx_arma<<endl<<endl; 
	
	arma::cx_vec eigval; 
	arma::cx_mat Ve_arma; 
	eig_gen(eigval, Ve_arma, Bx_arma, "balance"); 
// cout<<"Ve_arma in arma: "<<endl<<Ve_arma<<endl<<endl; 
// cout<<"e_values in arma: "<<endl<<eigval<<endl<<endl; 

	arma::mat V_arma; 
	V_arma = real(Ve_arma); 
// cout<<"V_arma in arma: "<<endl<<V_arma<<endl<<endl; 

// end armadillo  ------------------------------------------------------------ 

	MatrixXd V(35,35); 
	for(int i=0; i<35; i++){
		for(int j=0; j<35; j++){
			V(i,j) = V_arma(i,j); 
		} 
	} 
// cout<<"V in eigen: "<<endl<<V<<endl<<endl; 

	// correct sign of each column, the first element is always positive 
	MatrixXd V_1(35,35); 
	for(int i=0; i<35; i++){ // i represents column 
		for(int j=0; j<35; j++){ // j represent row 

			if(V(0,i)<0)
				V_1(j,i) = V(j,i) * (-1); 
			else 
				V_1(j,i) = V(j,i) * (1); 				

		} 
	} 
// cout<<"V_1: "<<endl<<V_1<<endl<<endl; 

	// recover quaternion elements 
	RowVectorXd w(35); 
	for(int i=0; i<35; i++){
		w(0,i) = sqrt(sqrt(V_1(0,i))); 
	} 
// cout<<"w: "<<endl<<w<<endl<<endl; 

 	RowVectorXd w3(35); 
	for(int i=0; i<35; i++){
		w3(0,i) = w(0,i)*w(0,i)*w(0,i); 
	} 
// cout<<"w3: "<<endl<<w3<<endl<<endl; 

	Matrix4Xd Q_0(4,35); 
	for(int i=0; i<35; i++){
		Q_0(0,i) = w(0,i); 
		Q_0(1,i) = V_1(1,i) / w3(0,i); 
		Q_0(2,i) = V_1(2,i) / w3(0,i); 
		Q_0(3,i) = V_1(3,i) / w3(0,i); 
	} 
// cout<<"Q_0: "<<endl<<Q_0<<endl<<endl; 

	RowVectorXd QNrm(1,35); 
	for(int i=0; i<35; i++){
		QNrm(0,i) = sqrt(Q_0(0,i)*Q_0(0,i)+Q_0(1,i)*Q_0(1,i)+Q_0(2,i)*Q_0(2,i)+Q_0(3,i)*Q_0(3,i)); 
	}
// cout<<"QNrm: "<<endl<<QNrm<<endl<<endl; 

	// normalize each column 
	for(int i=0; i<35; i++){
		Q(0,i) = Q_0(0,i) / QNrm(0,i); 
		Q(1,i) = Q_0(1,i) / QNrm(0,i); 
		Q(2,i) = Q_0(2,i) / QNrm(0,i); 
		Q(3,i) = Q_0(3,i) / QNrm(0,i); 
	} 
// cout<<"Q: "<<endl<<Q<<endl<<endl; 

// cout<<"QuEst_5Pt_Ver5_2 end -----------------------------------"<<endl<<endl; 

} 

// convert quaternion into rotation matrix ---------- 
void QuEst::Q2R(MatrixXd& R_Q2R, const Matrix4Xd& Q){

	int numInp_Q2R = Q.cols(); 

	for(int i=0; i<numInp_Q2R; i++){

		Vector4d q; 
		q(0,0) = Q(0,i); 
		q(1,0) = Q(1,i); 
		q(2,0) = Q(2,i); 
		q(3,0) = Q(3,i); 

		R_Q2R(0,i) = 1 - 2*q(2,0)*q(2,0) - 2*q(3,0)*q(3,0); 
		R_Q2R(1,i) = 2*q(1,0)*q(2,0) - 2*q(3,0)*q(0,0); 
		R_Q2R(2,i) = 2*q(1,0)*q(3,0) + 2*q(0,0)*q(2,0); 
		R_Q2R(3,i) = 2*q(1,0)*q(2,0) + 2*q(3,0)*q(0,0); 
		R_Q2R(4,i) = 1 - 2*q(1,0)*q(1,0) - 2*q(3,0)*q(3,0); 
		R_Q2R(5,i) = 2*q(2,0)*q(3,0) - 2*q(1,0)*q(0,0); 
		R_Q2R(6,i) = 2*q(1,0)*q(3,0) - 2*q(0,0)*q(2,0); 
		R_Q2R(7,i) = 2*q(2,0)*q(3,0) + 2*q(1,0)*q(0,0); 
		R_Q2R(8,i) = 1 - 2*q(1,0)*q(1,0) - 2*q(2,0)*q(2,0); 

	}
// cout<<"R_Q2R: "<<endl<<R_Q2R.transpose()<<endl<<endl; 

} 

void QuEst::Q2R_3by3(Matrix3d& R_Q2R, const Vector4d& Q){

		R_Q2R(0,0) = 1 - 2*Q(2,0)*Q(2,0) - 2*Q(3,0)*Q(3,0); 
		R_Q2R(0,1) = 2*Q(1,0)*Q(2,0) - 2*Q(3,0)*Q(0,0); 
		R_Q2R(0,2) = 2*Q(1,0)*Q(3,0) + 2*Q(0,0)*Q(2,0); 
		R_Q2R(1,0) = 2*Q(1,0)*Q(2,0) + 2*Q(3,0)*Q(0,0); 
		R_Q2R(1,1) = 1 - 2*Q(1,0)*Q(1,0) - 2*Q(3,0)*Q(3,0); 
		R_Q2R(1,2) = 2*Q(2,0)*Q(3,0) - 2*Q(1,0)*Q(0,0); 
		R_Q2R(2,0) = 2*Q(1,0)*Q(3,0) - 2*Q(0,0)*Q(2,0); 
		R_Q2R(2,1) = 2*Q(2,0)*Q(3,0) + 2*Q(1,0)*Q(0,0); 
		R_Q2R(2,2) = 1 - 2*Q(1,0)*Q(1,0) - 2*Q(2,0)*Q(2,0); 

}

// m1:(3 by n), m2:(3 by n), n is number of points 
void QuEst::CoefsVer_3_1_1(MatrixXd& Cf, const Matrix3Xd& m1, const Matrix3Xd& m2){

	int numPts = m1.cols(); 

	int numCols = numPts*(numPts-1)/2 - 1; 

	Matrix2Xi idxBin1(2,numCols); 
	int counter = 0; 
	for(int i=1; i<=numPts-2; i++){
		for(int j=i+1; j<=numPts; j++){
			counter = counter + 1; 
			idxBin1(0,counter-1) = i; 
			idxBin1(1,counter-1) = j; 
		}
	} 
// cout<<idxBin1<<endl; 

	VectorXd mx1(numCols,1); 
	VectorXd my1(numCols,1); 
	VectorXd s1(numCols,1); 
	VectorXd nx1(numCols,1); 
	VectorXd ny1(numCols,1); 
	VectorXd r1(numCols,1); 
	VectorXd mx2(numCols,1); 
	VectorXd my2(numCols,1); 
	VectorXd s2(numCols,1); 
	VectorXd nx2(numCols,1); 
	VectorXd ny2(numCols,1); 
	VectorXd r2(numCols,1); 
	for(int i=0; i<numCols; i++){
		int index_1 = idxBin1(0,i); 
		mx1(i,0) = m1(0,index_1-1); 
		my1(i,0) = m1(1,index_1-1); 
		s1(i,0) = m1(2,index_1-1); 
		nx1(i,0) = m2(0,index_1-1); 
		ny1(i,0) = m2(1,index_1-1); 
		r1(i,0) = m2(2,index_1-1); 

		int index_2 = idxBin1(1,i); 
		mx2(i,0) = m1(0,index_2-1); 
		my2(i,0) = m1(1,index_2-1); 
		s2(i,0) = m1(2,index_2-1); 
		nx2(i,0) = m2(0,index_2-1); 
		ny2(i,0) = m2(1,index_2-1); 
		r2(i,0) = m2(2,index_2-1); 
	} 
// cout<<mx1<<endl<<my1<<endl<<s1<<endl<<nx1<<endl<<ny1<<endl<<r1<<endl<<endl; 
// cout<<mx2<<endl<<my2<<endl<<s2<<endl<<nx2<<endl<<ny2<<endl<<r2<<endl<<endl; 

	MatrixXd coefsN(numCols, 10); 
	coefsNum(coefsN,mx1,mx2,my1,my2,nx2,ny2,r2,s1,s2); 	
// cout<<coefsN<<endl; 

	MatrixXd coefsD(numCols, 10); 
	coefsDen(coefsD,mx2,my2,nx1,nx2,ny1,ny2,r1,r2,s2); 	
// cout<<coefsD<<endl; 

	int numEq = numPts*(numPts-1)*(numPts-2)/6; 
	
	Matrix2Xi idxBin2(2,numEq); 
	int counter_bin2_1 = 0; 
	int counter_bin2_2 = 0; 
	for(int i=numPts-1; i>=2; i--){
		for(int j=1+counter_bin2_2; j<=i-1+counter_bin2_2; j++){
			for(int k=j+1; k<=i+counter_bin2_2; k++){
				counter_bin2_1 = counter_bin2_1 + 1; 
				idxBin2(0,counter_bin2_1-1) = j; 
				idxBin2(1,counter_bin2_1-1) = k; 
			}
		}
		counter_bin2_2 = i + counter_bin2_2; 
	} 
// cout<<idxBin2<<endl; 

	int numEqDouble = 2 * numEq; 
	
	VectorXd a1(numEqDouble,1); 
	VectorXd a2(numEqDouble,1); 
	VectorXd a3(numEqDouble,1); 
	VectorXd a4(numEqDouble,1); 
	VectorXd a5(numEqDouble,1); 
	VectorXd a6(numEqDouble,1); 
	VectorXd a7(numEqDouble,1); 
	VectorXd a8(numEqDouble,1); 
	VectorXd a9(numEqDouble,1); 
	VectorXd a10(numEqDouble,1); 
	VectorXd b1(numEqDouble,1); 
	VectorXd b2(numEqDouble,1); 
	VectorXd b3(numEqDouble,1); 
	VectorXd b4(numEqDouble,1); 
	VectorXd b5(numEqDouble,1); 
	VectorXd b6(numEqDouble,1); 
	VectorXd b7(numEqDouble,1); 
	VectorXd b8(numEqDouble,1); 
	VectorXd b9(numEqDouble,1); 
	VectorXd b10(numEqDouble,1); 
	for(int i=0; i<numEq; i++){ 

		int index_1 = idxBin2(0,i); 
		a1(i,0) = coefsN(index_1-1,0); 
		a1(i+numEq,0) = coefsD(index_1-1,0);
		a2(i,0) = coefsN(index_1-1,1); 
		a2(i+numEq,0) = coefsD(index_1-1,1); 
		a3(i,0) = coefsN(index_1-1,2); 
		a3(i+numEq,0) = coefsD(index_1-1,2); 
		a4(i,0) = coefsN(index_1-1,3); 
		a4(i+numEq,0) = coefsD(index_1-1,3); 
		a5(i,0) = coefsN(index_1-1,4); 
		a5(i+numEq,0) = coefsD(index_1-1,4); 
		a6(i,0) = coefsN(index_1-1,5); 
		a6(i+numEq,0) = coefsD(index_1-1,5); 
		a7(i,0) = coefsN(index_1-1,6); 
		a7(i+numEq,0) = coefsD(index_1-1,6); 
		a8(i,0) = coefsN(index_1-1,7); 
		a8(i+numEq,0) = coefsD(index_1-1,7); 
		a9(i,0) = coefsN(index_1-1,8); 
		a9(i+numEq,0) = coefsD(index_1-1,8); 
		a10(i,0) = coefsN(index_1-1,9); 
		a10(i+numEq,0) = coefsD(index_1-1,9); 

		int index_2 = idxBin2(1,i); 
		b1(i,0) = coefsD(index_2-1,0); 
		b1(i+numEq,0) = coefsN(index_2-1,0); 
		b2(i,0) = coefsD(index_2-1,1); 
		b2(i+numEq,0) = coefsN(index_2-1,1); 
		b3(i,0) = coefsD(index_2-1,2); 
		b3(i+numEq,0) = coefsN(index_2-1,2); 
		b4(i,0) = coefsD(index_2-1,3); 
		b4(i+numEq,0) = coefsN(index_2-1,3); 
		b5(i,0) = coefsD(index_2-1,4); 
		b5(i+numEq,0) = coefsN(index_2-1,4); 
		b6(i,0) = coefsD(index_2-1,5); 
		b6(i+numEq,0) = coefsN(index_2-1,5); 
		b7(i,0) = coefsD(index_2-1,6); 
		b7(i+numEq,0) = coefsN(index_2-1,6); 
		b8(i,0) = coefsD(index_2-1,7); 
		b8(i+numEq,0) = coefsN(index_2-1,7); 
		b9(i,0) = coefsD(index_2-1,8); 
		b9(i+numEq,0) = coefsN(index_2-1,8); 
		b10(i,0) = coefsD(index_2-1,9); 
		b10(i+numEq,0) = coefsN(index_2-1,9); 
	} 
// cout<<a1<<endl<<a2<<endl<<a3<<endl<<a4<<endl<<a5<<endl<<a6<<endl<<a7<<endl<<a8<<endl<<a9<<endl<<a10<<endl<<endl; 
// cout<<b1<<endl<<b2<<endl<<b3<<endl<<b4<<endl<<b5<<endl<<b6<<endl<<b7<<endl<<b8<<endl<<b9<<endl<<b10<<endl<<endl; 

	MatrixXd coefsND(numEqDouble, 35); 
	coefsNumDen(coefsND,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10); 
// cout<<"coefsND: "<<endl<<coefsND<<endl<<endl; 

	for(int i=0; i<numEq; i++){
		Cf(i,0) = coefsND(i,0) - coefsND(i+numEq,0); 		
		Cf(i,1) = coefsND(i,1) - coefsND(i+numEq,1); 		
		Cf(i,2) = coefsND(i,2) - coefsND(i+numEq,2); 		
		Cf(i,3) = coefsND(i,3) - coefsND(i+numEq,3); 		
		Cf(i,4) = coefsND(i,4) - coefsND(i+numEq,4); 		
		Cf(i,5) = coefsND(i,5) - coefsND(i+numEq,5); 		
		Cf(i,6) = coefsND(i,6) - coefsND(i+numEq,6); 		
		Cf(i,7) = coefsND(i,7) - coefsND(i+numEq,7); 		
		Cf(i,8) = coefsND(i,8) - coefsND(i+numEq,8); 		
		Cf(i,9) = coefsND(i,9) - coefsND(i+numEq,9); 		
		Cf(i,10) = coefsND(i,10) - coefsND(i+numEq,10); 		
		Cf(i,11) = coefsND(i,11) - coefsND(i+numEq,11); 		
		Cf(i,12) = coefsND(i,12) - coefsND(i+numEq,12); 		
		Cf(i,13) = coefsND(i,13) - coefsND(i+numEq,13); 		
		Cf(i,14) = coefsND(i,14) - coefsND(i+numEq,14); 		
		Cf(i,15) = coefsND(i,15) - coefsND(i+numEq,15); 		
		Cf(i,16) = coefsND(i,16) - coefsND(i+numEq,16); 		
		Cf(i,17) = coefsND(i,17) - coefsND(i+numEq,17); 		
		Cf(i,18) = coefsND(i,18) - coefsND(i+numEq,18); 		
		Cf(i,19) = coefsND(i,19) - coefsND(i+numEq,19); 		
		Cf(i,20) = coefsND(i,20) - coefsND(i+numEq,20); 		
		Cf(i,21) = coefsND(i,21) - coefsND(i+numEq,21); 		
		Cf(i,22) = coefsND(i,22) - coefsND(i+numEq,22); 		
		Cf(i,23) = coefsND(i,23) - coefsND(i+numEq,23); 		
		Cf(i,24) = coefsND(i,24) - coefsND(i+numEq,24); 		
		Cf(i,25) = coefsND(i,25) - coefsND(i+numEq,25); 		
		Cf(i,26) = coefsND(i,26) - coefsND(i+numEq,26); 		
		Cf(i,27) = coefsND(i,27) - coefsND(i+numEq,27); 		
		Cf(i,28) = coefsND(i,28) - coefsND(i+numEq,28); 		
		Cf(i,29) = coefsND(i,29) - coefsND(i+numEq,29); 		
		Cf(i,30) = coefsND(i,30) - coefsND(i+numEq,30); 		
		Cf(i,31) = coefsND(i,31) - coefsND(i+numEq,31); 		
		Cf(i,32) = coefsND(i,32) - coefsND(i+numEq,32); 		
		Cf(i,33) = coefsND(i,33) - coefsND(i+numEq,33); 		
		Cf(i,34) = coefsND(i,34) - coefsND(i+numEq,34); 		
	} 
// cout<<"Cf: "<<endl<<Cf<<endl<<endl; 


}

void QuEst::coefsNum(MatrixXd& coefsN, const VectorXd& mx1, const VectorXd& mx2, 
	const VectorXd& my1, const VectorXd& my2, 
	const VectorXd& nx2, const VectorXd& ny2, 
	const VectorXd& r2, const VectorXd& s1, const VectorXd& s2){

	int numPts = mx1.rows(); 

	VectorXd t2(numPts,1); 
	VectorXd t3(numPts,1); 
	VectorXd t4(numPts,1); 
	VectorXd t5(numPts,1); 
	VectorXd t6(numPts,1); 
	VectorXd t7(numPts,1); 
	VectorXd t8(numPts,1); 
	VectorXd t9(numPts,1); 
	VectorXd t10(numPts,1); 
	VectorXd t11(numPts,1); 
	VectorXd t12(numPts,1); 
	VectorXd t13(numPts,1); 
	for(int i=0; i<numPts; i++){
		t2(i,0) = mx1(i,0)*my2(i,0)*r2(i,0); 
		t3(i,0) = mx2(i,0)*ny2(i,0)*s1(i,0); 
		t4(i,0) = my1(i,0)*nx2(i,0)*s2(i,0); 
		t5(i,0) = mx1(i,0)*nx2(i,0)*s2(i,0)*2.0; 
		t6(i,0) = my1(i,0)*ny2(i,0)*s2(i,0)*2.0; 
		t7(i,0) = mx1(i,0)*my2(i,0)*nx2(i,0)*2.0; 
		t8(i,0) = my2(i,0)*r2(i,0)*s1(i,0)*2.0; 
		t9(i,0) = mx2(i,0)*my1(i,0)*r2(i,0); 
		t10(i,0) = mx1(i,0)*ny2(i,0)*s2(i,0); 
		t11(i,0) = mx2(i,0)*my1(i,0)*ny2(i,0)*2.0; 
		t12(i,0) = mx2(i,0)*r2(i,0)*s1(i,0)*2.0; 
		t13(i,0) = my2(i,0)*nx2(i,0)*s1(i,0); 
// cout<<t2<<endl<<t3<<endl<<t4<<endl<<t5<<endl<<t6<<endl<<t7<<endl<<t8<<endl<<t9<<endl<<t10<<endl<<t11<<endl<<t12<<endl<<t13<<endl<<endl; 
	
		coefsN(i,0) = t2(i,0)+t3(i,0)+t4(i,0)-mx2(i,0)*my1(i,0)*r2(i,0)-mx1(i,0)*ny2(i,0)*s2(i,0)-my2(i,0)*nx2(i,0)*s1(i,0); 
		coefsN(i,1) = t11(i,0)+t12(i,0)-mx1(i,0)*my2(i,0)*ny2(i,0)*2.0-mx1(i,0)*r2(i,0)*s2(i,0)*2.0; 
		coefsN(i,2) = t7(i,0)+t8(i,0)-mx2(i,0)*my1(i,0)*nx2(i,0)*2.0-my1(i,0)*r2(i,0)*s2(i,0)*2.0; 
		coefsN(i,3) = t5(i,0)+t6(i,0)-mx2(i,0)*nx2(i,0)*s1(i,0)*2.0-my2(i,0)*ny2(i,0)*s1(i,0)*2.0; 
		coefsN(i,4) = -t2(i,0)-t3(i,0)+t4(i,0)+t9(i,0)+t10(i,0)-my2(i,0)*nx2(i,0)*s1(i,0); 
		coefsN(i,5) = -t5(i,0)+t6(i,0)+mx2(i,0)*nx2(i,0)*s1(i,0)*2.0-my2(i,0)*ny2(i,0)*s1(i,0)*2.0; 
		coefsN(i,6) = t7(i,0)-t8(i,0)-mx2(i,0)*my1(i,0)*nx2(i,0)*2.0+my1(i,0)*r2(i,0)*s2(i,0)*2.0; 
		coefsN(i,7) = -t2(i,0)+t3(i,0)-t4(i,0)+t9(i,0)-t10(i,0)+t13(i,0); 
		coefsN(i,8) = -t11(i,0)+t12(i,0)+mx1(i,0)*my2(i,0)*ny2(i,0)*2.0-mx1(i,0)*r2(i,0)*s2(i,0)*2.0; 
		coefsN(i,9) = t2(i,0)-t3(i,0)-t4(i,0)-t9(i,0)+t10(i,0)+t13(i,0); 
	} 
// cout<<coefsN<<endl<<endl; 

}

void QuEst::coefsDen(MatrixXd& coefsD, const VectorXd& mx2, const VectorXd& my2, 
	const VectorXd& nx1, const VectorXd& nx2, 
	const VectorXd& ny1, const VectorXd& ny2, 
	const VectorXd& r1, const VectorXd& r2, const VectorXd& s2){

	int numPts = mx2.rows(); 

	VectorXd t2_D(numPts,1); 
	VectorXd t3_D(numPts,1); 
	VectorXd t4_D(numPts,1); 
	VectorXd t5_D(numPts,1); 
	VectorXd t6_D(numPts,1); 
	VectorXd t7_D(numPts,1); 
	VectorXd t8_D(numPts,1); 
	VectorXd t9_D(numPts,1); 
	VectorXd t10_D(numPts,1); 
	VectorXd t11_D(numPts,1); 
	VectorXd t12_D(numPts,1); 
	VectorXd t13_D(numPts,1); 
	for(int i=0; i<numPts; i++){
		t2_D(i,0) = mx2(i,0)*ny1(i,0)*r2(i,0); 
		t3_D(i,0) = my2(i,0)*nx2(i,0)*r1(i,0); 
		t4_D(i,0) = nx1(i,0)*ny2(i,0)*s2(i,0); 
		t5_D(i,0) = mx2(i,0)*nx2(i,0)*r1(i,0)*2.0; 
		t6_D(i,0) = my2(i,0)*ny2(i,0)*r1(i,0)*2.0; 
		t7_D(i,0) = mx2(i,0)*nx2(i,0)*ny1(i,0)*2.0; 
		t8_D(i,0) = ny1(i,0)*r2(i,0)*s2(i,0)*2.0; 
		t9_D(i,0) = my2(i,0)*nx1(i,0)*r2(i,0); 
		t10_D(i,0) = nx2(i,0)*ny1(i,0)*s2(i,0); 
		t11_D(i,0) = my2(i,0)*nx1(i,0)*ny2(i,0)*2.0; 
		t12_D(i,0) = nx1(i,0)*r2(i,0)*s2(i,0)*2.0; 
		t13_D(i,0) = mx2(i,0)*ny2(i,0)*r1(i,0); 
// cout<<t2_D<<endl<<t3_D<<endl<<t4_D<<endl<<t5_D<<endl<<t6_D<<endl<<t7_D<<endl<<t8_D<<endl<<t9_D<<endl<<t10_D<<endl<<t11_D<<endl<<t12_D<<endl<<t13_D<<endl<<endl; 

		coefsD(i,0) = t2_D(i,0)+t3_D(i,0)+t4_D(i,0)-mx2(i,0)*ny2(i,0)*r1(i,0)-my2(i,0)*nx1(i,0)*r2(i,0)-nx2(i,0)*ny1(i,0)*s2(i,0); 
		coefsD(i,1) = t11_D(i,0)+t12_D(i,0)-my2(i,0)*nx2(i,0)*ny1(i,0)*2.0-nx2(i,0)*r1(i,0)*s2(i,0)*2.0; 
		coefsD(i,2) = t7_D(i,0)+t8_D(i,0)-mx2(i,0)*nx1(i,0)*ny2(i,0)*2.0-ny2(i,0)*r1(i,0)*s2(i,0)*2.0; 
		coefsD(i,3) = t5_D(i,0)+t6_D(i,0)-mx2(i,0)*nx1(i,0)*r2(i,0)*2.0-my2(i,0)*ny1(i,0)*r2(i,0)*2.0; 
		coefsD(i,4) = t2_D(i,0)-t3_D(i,0)-t4_D(i,0)+t9_D(i,0)+t10_D(i,0)-mx2(i,0)*ny2(i,0)*r1(i,0); 
		coefsD(i,5) = t5_D(i,0)-t6_D(i,0)-mx2(i,0)*nx1(i,0)*r2(i,0)*2.0+my2(i,0)*ny1(i,0)*r2(i,0)*2.0; 
		coefsD(i,6) = -t7_D(i,0)+t8_D(i,0)+mx2(i,0)*nx1(i,0)*ny2(i,0)*2.0-ny2(i,0)*r1(i,0)*s2(i,0)*2.0; 
		coefsD(i,7) = -t2_D(i,0)+t3_D(i,0)-t4_D(i,0)-t9_D(i,0)+t10_D(i,0)+t13_D(i,0); 
		coefsD(i,8) = t11_D(i,0)-t12_D(i,0)-my2(i,0)*nx2(i,0)*ny1(i,0)*2.0+nx2(i,0)*r1(i,0)*s2(i,0)*2.0; 
		coefsD(i,9) = -t2_D(i,0)-t3_D(i,0)+t4_D(i,0)+t9_D(i,0)-t10_D(i,0)+t13_D(i,0); 
	} 	
// cout<<coefsD<<endl<<endl; 

}

void QuEst::coefsNumDen(MatrixXd& coefsND, const VectorXd& a1, const VectorXd& a2, const VectorXd& a3, 
	const VectorXd& a4, const VectorXd& a5, const VectorXd& a6, const VectorXd& a7, 
	const VectorXd& a8, const VectorXd& a9, const VectorXd& a10, 
	const VectorXd& b1, const VectorXd& b2, const VectorXd& b3, const VectorXd& b4, 
	const VectorXd& b5, const VectorXd& b6, const VectorXd& b7, const VectorXd& b8, 
	const VectorXd& b9, const VectorXd& b10){

	int numPts = a1.rows(); 

	for(int i=0; i<numPts; i++){

		coefsND(i,0) = a1(i,0)*b1(i,0); 
		coefsND(i,1) = a1(i,0)*b2(i,0)+a2(i,0)*b1(i,0); 
		coefsND(i,2) = a2(i,0)*b2(i,0)+a1(i,0)*b5(i,0)+a5(i,0)*b1(i,0); 
		coefsND(i,3) = a2(i,0)*b5(i,0)+a5(i,0)*b2(i,0); 
		coefsND(i,4) = a5(i,0)*b5(i,0); 
		coefsND(i,5) = a1(i,0)*b3(i,0)+a3(i,0)*b1(i,0); 
		coefsND(i,6) = a2(i,0)*b3(i,0)+a3(i,0)*b2(i,0)+a1(i,0)*b6(i,0)+a6(i,0)*b1(i,0); 
		coefsND(i,7) = a2(i,0)*b6(i,0)+a3(i,0)*b5(i,0)+a5(i,0)*b3(i,0)+a6(i,0)*b2(i,0); 
		coefsND(i,8) = a5(i,0)*b6(i,0)+a6(i,0)*b5(i,0); 
		coefsND(i,9) = a3(i,0)*b3(i,0)+a1(i,0)*b8(i,0)+a8(i,0)*b1(i,0); 
		coefsND(i,10) = a3(i,0)*b6(i,0)+a6(i,0)*b3(i,0)+a2(i,0)*b8(i,0)+a8(i,0)*b2(i,0); 
		coefsND(i,11) = a6(i,0)*b6(i,0)+a5(i,0)*b8(i,0)+a8(i,0)*b5(i,0); 
		coefsND(i,12) = a3(i,0)*b8(i,0)+a8(i,0)*b3(i,0); 
		coefsND(i,13) = a6(i,0)*b8(i,0)+a8(i,0)*b6(i,0); 
		coefsND(i,14) = a8(i,0)*b8(i,0); 
		coefsND(i,15) = a1(i,0)*b4(i,0)+a4(i,0)*b1(i,0); 
		coefsND(i,16) = a2(i,0)*b4(i,0)+a4(i,0)*b2(i,0)+a1(i,0)*b7(i,0)+a7(i,0)*b1(i,0); 
		coefsND(i,17) = a2(i,0)*b7(i,0)+a4(i,0)*b5(i,0)+a5(i,0)*b4(i,0)+a7(i,0)*b2(i,0); 
		coefsND(i,18) = a5(i,0)*b7(i,0)+a7(i,0)*b5(i,0); 
		coefsND(i,19) = a3(i,0)*b4(i,0)+a4(i,0)*b3(i,0)+a1(i,0)*b9(i,0)+a9(i,0)*b1(i,0); 
		coefsND(i,20) = a3(i,0)*b7(i,0)+a4(i,0)*b6(i,0)+a6(i,0)*b4(i,0)+a7(i,0)*b3(i,0)+a2(i,0)*b9(i,0)+a9(i,0)*b2(i,0); 
		coefsND(i,21) = a6(i,0)*b7(i,0)+a7(i,0)*b6(i,0)+a5(i,0)*b9(i,0)+a9(i,0)*b5(i,0); 
		coefsND(i,22) = a3(i,0)*b9(i,0)+a4(i,0)*b8(i,0)+a8(i,0)*b4(i,0)+a9(i,0)*b3(i,0); 
		coefsND(i,23) = a6(i,0)*b9(i,0)+a7(i,0)*b8(i,0)+a8(i,0)*b7(i,0)+a9(i,0)*b6(i,0); 
		coefsND(i,24) = a8(i,0)*b9(i,0)+a9(i,0)*b8(i,0); 
		coefsND(i,25) = a4(i,0)*b4(i,0)+a1(i,0)*b10(i,0)+a10(i,0)*b1(i,0); 
		coefsND(i,26) = a4(i,0)*b7(i,0)+a7(i,0)*b4(i,0)+a2(i,0)*b10(i,0)+a10(i,0)*b2(i,0); 
		coefsND(i,27) = a7(i,0)*b7(i,0)+a5(i,0)*b10(i,0)+a10(i,0)*b5(i,0); 
		coefsND(i,28) = a3(i,0)*b10(i,0)+a4(i,0)*b9(i,0)+a9(i,0)*b4(i,0)+a10(i,0)*b3(i,0); 
		coefsND(i,29) = a6(i,0)*b10(i,0)+a7(i,0)*b9(i,0)+a9(i,0)*b7(i,0)+a10(i,0)*b6(i,0); 
		coefsND(i,30) = a8(i,0)*b10(i,0)+a9(i,0)*b9(i,0)+a10(i,0)*b8(i,0); 
		coefsND(i,31) = a4(i,0)*b10(i,0)+a10(i,0)*b4(i,0); 
		coefsND(i,32) = a7(i,0)*b10(i,0)+a10(i,0)*b7(i,0); 
		coefsND(i,33) = a9(i,0)*b10(i,0)+a10(i,0)*b9(i,0); 
		coefsND(i,34) = a10(i,0)*b10(i,0); 
	}

}

int main(int argc, char** argv){

	ros::init(argc, argv, "QuEst_RANSAC"); 

	QuEst q; 

	ros::spin(); 

	return 0; 
} 
