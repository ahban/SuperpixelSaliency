#if !defined(_OPTICALFLOW_H_INCLUDED_)
#define _OPTICALFLOW_H_INCLUDED_

#include <vector>
#include <opencv2/opencv.hpp>

#define move_th 0.5
#define back_th 10.0f

class FLOW
{
public:
	int FlowType();

	void calcOpticalFlow(
		cv::Mat prevImg,
		cv::Mat nextImg,
		cv::Mat& flowMat,
		std::vector<cv::Point2f>& prevPt,
		std::vector<cv::Point2f>& nextPt,
		int mode);

	void Farneback(
		cv::Mat prevImg,
		cv::Mat nextImg,
		cv::Mat& flowmat,
		std::vector<cv::Point2f>& prevPt,
		std::vector<cv::Point2f>& nextPt);

	void PyramidLK(
		cv::Mat prevImg,
		cv::Mat nextImg,
		cv::Mat& flowmat,
		std::vector<cv::Point2f>& prevPt,
		std::vector<cv::Point2f>& nextPt);

	void SimpleFlow(
		cv::Mat prevImg,
		cv::Mat nextImg,
		cv::Mat& flowMat,
		std::vector<cv::Point2f>& prevPt,
		std::vector<cv::Point2f>& nextPt);

	void TVL1(
		cv::Mat prevImg,
		cv::Mat nextImg,
		cv::Mat& flowMat,
		std::vector<cv::Point2f>& prevPt,
		std::vector<cv::Point2f>& nextPt);

	void DrawFlow(
		cv::Mat& img,
		cv::Mat flowMat);
};

#endif