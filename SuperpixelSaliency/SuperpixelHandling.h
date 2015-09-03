#if !defined(_SUPERPIXELHANDLING_H_INCLUDED_)
#define _SUPERPIXELHANDLING_H_INCLUDED_

#include <vector>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

typedef struct spdata{
	int orilabel;
	int relabel;
	int size;
	cv::Point center;
	cv::Vec3f colors;
    float intensity;
    float rg;
    float by;
	cv::Point2f flow;
	std::vector<cv::Point2f> raw_flow;
	bool status;

	spdata() {
		orilabel = -1;
		relabel = -1;
		size = 0;
		center = cv::Point(0, 0);
		colors = cv::Vec3f(0.0f, 0.0f, 0.0f);
        intensity = 0.0f;
        rg = 0.0f;
        by = 0.0f;
		flow = cv::Point2f(0.0, 0.0);
		status = false;
	}
}sp;

class SPD {
public:
	void SP_Feature(
		std::vector<sp>& sp_data,
		cv::Mat src,
		cv::Mat flowMat,
		int* klabels);

	void reLabeling(
		cv::Mat src,
		cv::Mat& dst,
		int* klabels,
		std::vector<sp>& sp_data);

	void SP_Clustering(
		cv::Mat& dst,
		std::vector<sp> sp_data,
		std::vector<sp>& cclst,
		int* klabels,
		int numlabels);

	void SaveSuperpixelData(
		std::vector<sp> sp_data,
		int frame);

private:
	double calcVariance(std::vector<cv::Point2f> raw_flow);

	double CalcDistance(sp data1, sp data2);
};

#endif