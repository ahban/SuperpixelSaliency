#include "OpticalFlow.h"

int FLOW::FlowType()
{
	int mode = 0;
	while(mode != 1 && mode != 2 && mode != 3 && mode != 4)
	{
		std::cout << "FB:1, PyrLK:2, SF:3, TVL1:4" << std::endl;
		std::cin >> mode;
	}

	return mode;
}

void FLOW::calcOpticalFlow(cv::Mat prevImg,
						   cv::Mat nextImg,
						   cv::Mat& flowMat,
						   std::vector<cv::Point2f>& prevPt,
						   std::vector<cv::Point2f>& nextPt,
						   int mode)
{
	switch(mode) {
	case 1:
		Farneback(prevImg, nextImg, flowMat, prevPt, nextPt);
		break;

	case 2:
		PyramidLK(prevImg, nextImg, flowMat, prevPt, nextPt);
		break;

	case 3:
		SimpleFlow(prevImg, nextImg, flowMat, prevPt, nextPt);
		break;

	case 4:
		TVL1(prevImg, nextImg, flowMat, prevPt, nextPt);
		break;

	default:
		break;
	}
}

void FLOW::Farneback(cv::Mat prevImg,
					 cv::Mat nextImg,
					 cv::Mat& flowMat,
					 std::vector<cv::Point2f>& prevPt,
					 std::vector<cv::Point2f>& nextPt)
{
	int width = prevImg.cols;
	int height = prevImg.rows;

	/* parameters of Farneback */
	double pyrScale = 0.5;
	int levels = 5;
	int winsize = 10;
	int iterations = 10;
	int poly_n = 7;
	double poly_sigma = 1.5;
	int flags = cv::OPTFLOW_FARNEBACK_GAUSSIAN;

	cv::Mat prevGrayImg;
	cv::Mat nextGrayImg;
	cv::Mat backflowMat(height, width, CV_32FC2);

	cv::cvtColor(prevImg, prevGrayImg, CV_BGR2GRAY);
	cv::cvtColor(nextImg, nextGrayImg, CV_BGR2GRAY);
	cv::calcOpticalFlowFarneback(prevGrayImg, nextGrayImg, flowMat, pyrScale, levels, winsize, iterations, poly_n, poly_sigma, flags);
	cv::calcOpticalFlowFarneback(nextGrayImg, prevGrayImg, backflowMat, pyrScale, levels, winsize, iterations, poly_n, poly_sigma, flags);

	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			float e_x = flowMat.at<cv::Point2f>(y, x).x + backflowMat.at<cv::Point2f>(y, x).x;
			float e_y = flowMat.at<cv::Point2f>(y, x).y + backflowMat.at<cv::Point2f>(y, x).y;

			if((std::abs(flowMat.at<cv::Point2f>(y, x).x) > move_th || std::abs(flowMat.at<cv::Point2f>(y, x).y) > move_th) && e_x * e_x + e_y * e_y < back_th) {
				float x1 = (float)x + flowMat.at<cv::Point2f>(y, x).x;
				float y1 = (float)y + flowMat.at<cv::Point2f>(y, x).y;

				prevPt.push_back(cv::Point2f((float)x, (float)y));
				nextPt.push_back(cv::Point2f(x1, y1));
			}

			else
				flowMat.at<cv::Point2f>(y, x) = cv::Point2f(0.0, 0.0);
		}
	}
}

void FLOW::PyramidLK(cv::Mat prevImg,
					 cv::Mat nextImg,
					 cv::Mat& flowMat,
					 std::vector<cv::Point2f>& prevPt,
					 std::vector<cv::Point2f>& nextPt)
{
	int width = prevImg.cols;
	int height = prevImg.rows;

	/* parameters of PyramidLK */
	std::vector<cv::Point2f> prevGrid;
	std::vector<cv::Point2f> nextGrid;
	std::vector<cv::Point2f> primeGrid;
	std::vector<unsigned char> status1;
	std::vector<float> err1;
	std::vector<unsigned char> status2;
	std::vector<float> err2;
	cv::Size winSize(15, 15);
	int maxLevel = 3;
	cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.0001);

	cv::Mat prevGrayImg;
	cv::Mat nextGrayImg;
	cv::cvtColor(prevImg, prevGrayImg, CV_BGR2GRAY);
	cv::cvtColor(nextImg, nextGrayImg, CV_BGR2GRAY);

	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			prevGrid.push_back(cv::Point2f((float)x, (float)y));
			flowMat.at<cv::Point2f>(y, x) = cv::Point2f(0.0, 0.0);
		}
	}

	cv::calcOpticalFlowPyrLK(prevGrayImg, nextGrayImg, prevGrid, nextGrid, status1, err1, winSize, maxLevel, criteria);
	cv::calcOpticalFlowPyrLK(nextGrayImg, prevGrayImg, nextGrid, primeGrid, status2, err2, winSize, maxLevel, criteria);

	int i = 0, j = 0;
	for(i = 0; i < (int)prevGrid.size(); i++) {
		if(status1[i] == 1 && status2[i] == 1) {
			double dx = std::abs(nextGrid[i].x - prevGrid[i].x);
			double dy = std::abs(nextGrid[i].y - prevGrid[i].y);
			float e_x = prevGrid[i].x - primeGrid[i].x;
			float e_y = prevGrid[i].y - primeGrid[i].y;

			if((dx > move_th || dy > move_th) && e_x * e_x + e_y * e_y < back_th) {
				prevPt.push_back(prevGrid[i]);
				nextPt.push_back(nextGrid[i]);
				flowMat.at<cv::Point2f>((int)prevPt[j].y, (int)prevPt[j].x) = prevPt[j] - nextPt[j];
				j++;
			}
		}
	}
}

void FLOW::SimpleFlow(cv::Mat prevImg,
					  cv::Mat nextImg,
					  cv::Mat& flowMat,
					  std::vector<cv::Point2f>& prevPt,
					  std::vector<cv::Point2f>& nextPt)
{
	int width = prevImg.cols;
	int height = prevImg.rows;

	/* parameters of SimpleFlow */
	int layers = 5;
	int averaging_block_size = 3;
	int max_flow = 3;

	cv::Mat backflowMat(height, width, CV_32FC2);

	cv::calcOpticalFlowSF(prevImg, nextImg, flowMat, layers, averaging_block_size, max_flow);
	cv::calcOpticalFlowSF(nextImg, prevImg, backflowMat, 3, 2, 4);

	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			float e_x = flowMat.at<cv::Point2f>(y, x).x + backflowMat.at<cv::Point2f>(y, x).x;
			float e_y = flowMat.at<cv::Point2f>(y, x).y + backflowMat.at<cv::Point2f>(y, x).y;

			if((std::abs(flowMat.at<cv::Point2f>(y, x).x) > move_th || std::abs(flowMat.at<cv::Point2f>(y, x).y) > move_th) && e_x * e_x + e_y * e_y < back_th) {
				float x1 = (float)x + flowMat.at<cv::Point2f>(y, x).x;
				float y1 = (float)y + flowMat.at<cv::Point2f>(y, x).y;

				prevPt.push_back(cv::Point2f((float)x, (float)y));
				nextPt.push_back(cv::Point2f(x1, y1));
			}

			else
				flowMat.at<cv::Point2f>(y, x) = cv::Point2f(0.0, 0.0);
		}
	}
}

void FLOW::TVL1(cv::Mat prevImg,
				cv::Mat nextImg,
				cv::Mat& flowMat,
				std::vector<cv::Point2f>& prevPt,
				std::vector<cv::Point2f>& nextPt)
{
	int width = prevImg.cols;
	int height = prevImg.rows;

	cv::Mat prevGrayImg;
	cv::Mat nextGrayImg;
	cv::cvtColor(prevImg, prevGrayImg, CV_BGR2GRAY);
	cv::cvtColor(nextImg, nextGrayImg, CV_BGR2GRAY);

	cv::Mat backflowMat;

	cv::Ptr<cv::DenseOpticalFlow> tvl1 = cv::createOptFlow_DualTVL1();
	tvl1->calc(prevGrayImg, nextGrayImg, flowMat);
	tvl1->calc(nextGrayImg, prevGrayImg, backflowMat);

	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			float e_x = flowMat.at<cv::Point2f>(y, x).x + backflowMat.at<cv::Point2f>(y, x).x;
			float e_y = flowMat.at<cv::Point2f>(y, x).y + backflowMat.at<cv::Point2f>(y, x).y;

			if((std::abs(flowMat.at<cv::Point2f>(y, x).x) > move_th || std::abs(flowMat.at<cv::Point2f>(y, x).y) > move_th) && e_x * e_x + e_y * e_y < back_th) {
				float x1 = (float)x + flowMat.at<cv::Point2f>(y, x).x;
				float y1 = (float)y + flowMat.at<cv::Point2f>(y, x).y;

				prevPt.push_back(cv::Point2f((float)x, (float)y));
				nextPt.push_back(cv::Point2f(x1, y1));
			}

			else
				flowMat.at<cv::Point2f>(y, x) = cv::Point2f(0.0, 0.0);
		}
	}
}

void FLOW::DrawFlow(cv::Mat& img, cv::Mat flowMat)
{
	int width = img.cols;
	int height = img.rows;

	cv::cvtColor(img, img, CV_BGR2HSV);

	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			if(flowMat.at<cv::Point2f>(y, x) != cv::Point2f(0.0, 0.0)) {
				double dx = flowMat.at<cv::Point2f>(y, x).x;
				double dy = flowMat.at<cv::Point2f>(y, x).y;
				double theta = atan2(dy, dx);

				if(theta < 0.0)
					theta += 2.0 * CV_PI;

				int deg = cvRound(theta * 180.0 / CV_PI) / 2;

				img.at<cv::Vec3b>(y, x) = cv::Vec3b(deg, 255, 255);
			}
		}
	}

	cv::cvtColor(img, img, CV_HSV2BGR);
}