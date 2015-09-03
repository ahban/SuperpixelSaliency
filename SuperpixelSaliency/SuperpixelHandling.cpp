#include "SuperpixelHandling.h"

// public
void SPD::SP_Feature(std::vector<sp>& sp_data,
					 cv::Mat src,
					 cv::Mat flowMat,
					 int* klabels)
{
    cv::Mat_<cv::Vec3f> img = src / 255.0f;
	int width = img.cols;
	int height = img.rows;
	std::vector<int> m_size(sp_data.size());
    
    // luminosity map
    std::vector<cv::Mat_<float>> colors;
    cv::split(img, colors);
    cv::Mat_<float> I = (colors[0] + colors[1] + colors[2]) / 3.0f;
    
    // normalize rgb
    double minval, maxval;
    cv::minMaxLoc(I, &minval, &maxval);
    cv::Mat_<float> r(height, width, 0.0f);
    cv::Mat_<float> g(height, width, 0.0f);
    cv::Mat_<float> b(height, width, 0.0f);
    for(int j = 0; j < height; j++) {
        for(int i = 0; i < width; i++) {
            if(I(j, i) < 0.1f * maxval) // if _ < max / 10.0f => 0.0f
                continue;
            
            r(j, i) = colors[2](j, i) / I(j, i);
            g(j, i) = colors[1](j, i) / I(j, i);
            b(j, i) = colors[0](j, i) / I(j, i);
        }
    }
    
    // RGBY map( if _ < 0.0f => 0.0f )
    cv::Mat R = cv::max(0.0f, r - (g + b) / 2.0f);
    cv::Mat G = cv::max(0.0f, g - (b + r) / 2.0f);
    cv::Mat B = cv::max(0.0f, b - (r + g) / 2.0f);
    cv::Mat Y = cv::max(0.0f, (r + g) / 2.0f - abs(r - g) / 2.0f - b);
    cv::Mat RG = cv::abs(R - G);
    cv::Mat BY = cv::abs(B - Y);

    // integrate sp_data
	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			int index = y * width + x;
			int label = klabels[index];

			sp_data[label].orilabel = label;
			sp_data[label].size++;
			sp_data[label].center += cv::Point(x, y);
			sp_data[label].colors += I.at<cv::Vec3b>(y, x);
            sp_data[label].intensity += I.at<float>(y, x);
            sp_data[label].rg += RG.at<float>(y, x);
            sp_data[label].by += BY.at<float>(y, x);
			sp_data[label].flow += flowMat.at<cv::Point2f>(y, x);

			if(flowMat.at<cv::Point2f>(y, x) != cv::Point2f(0.0, 0.0)) {
				sp_data[label].raw_flow.push_back(flowMat.at<cv::Point2f>(y, x));
				m_size[label]++;
			}
		}
	}

    // average sp_data
	for(int i = 0; i < (int)sp_data.size(); i++) {
		sp_data[i].center.x /= sp_data[i].size;
		sp_data[i].center.y /= sp_data[i].size;
		sp_data[i].colors[0] /= float(sp_data[i].size);
		sp_data[i].colors[1] /= float(sp_data[i].size);
		sp_data[i].colors[2] /= float(sp_data[i].size);
        sp_data[i].intensity /= float(sp_data[i].size);
        sp_data[i].rg /= float(sp_data[i].size);
		sp_data[i].by /= float(sp_data[i].size);
        sp_data[i].flow.x /= float(m_size[i]);
		sp_data[i].flow.y /= float(m_size[i]);

		if(calcVariance(sp_data[i].raw_flow) < 0.7 && (double)m_size[i] / (double)sp_data[i].size >= 0.7)
			sp_data[i].status = true;

		else
			sp_data[i].status = false;
	}
}

void SPD::reLabeling(cv::Mat src,
					 cv::Mat& dst,
					 int* klabels,
					 std::vector<sp>& sp_data)
{
	int width = src.cols;
	int height = src.rows;

	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			int index = y * width + x;
			int label = klabels[index];

			if(sp_data[label].status == true)
				sp_data[label].relabel = sp_data[label].orilabel;

			else if(sp_data[label].status == false) {
				sp_data[label].relabel = -1;
				dst.at<cv::Vec3b>(y, x) = src.at<cv::Vec3b>(y, x);
			}
		}
	}
}

void SPD::SP_Clustering(cv::Mat& dst,
						std::vector<sp> sp_data,
						std::vector<sp>& cclst,
						int* klabels,
						int numlabels)
{
	//cv::Mat LabelMap(dst.size(), CV_8U);
	int num_clst = 1;
	int min_num = 0;
	double distance = 0.0, min_distance = 0.0;

	int i = 0;
	while(i < numlabels) {
		if(sp_data[i].status == true) {
			cclst.push_back(sp_data[i]);
			i++;
			break;
		}
		i++;
	}

	while(i < numlabels) {
		min_distance = DBL_MAX;

		for(int j = 0; j < (int)cclst.size(); j++) {
			while(1) {
				if(sp_data[i].status == true || i == numlabels - 1)
					break;

				else
					i++;
			}

			distance = CalcDistance(sp_data[i], cclst[j]);
			if(distance < min_distance) {
				min_distance = distance;
				min_num = j;
			}
		}

		if(min_distance < 120.0) {
			sp_data[i].relabel = min_num;
			cclst[min_num].center.x = (cclst[min_num].center.x + sp_data[i].center.x) / 2;
			cclst[min_num].center.y = (cclst[min_num].center.y + sp_data[i].center.y) / 2;
			cclst[min_num].colors[0] = (cclst[min_num].colors[0] + sp_data[i].colors[0]) / 2;
			cclst[min_num].colors[1] = (cclst[min_num].colors[1] + sp_data[i].colors[1]) / 2;
			cclst[min_num].colors[2] = (cclst[min_num].colors[2] + sp_data[i].colors[2]) / 2;
		}

		else {
			cclst.push_back(sp_data[i]);
			num_clst++;
		}
		i++;
	}

	int width = dst.cols;
	int height = dst.rows;
	std::vector<cv::Vec3b> color(numlabels);
	
	for(i = 0; i < numlabels; i++)
		color[i] = cv::Vec3b(rand()%181, 255, 255);

	cv::cvtColor(dst, dst, CV_BGR2HSV);
	for(int y = 0; y < height ; y++) {
		for(int x = 0; x < width; x++) {
			int index = y * width + x;
			int label = klabels[index];

			if(sp_data[label].status == true)
				dst.at<cv::Vec3b>(y, x) = color[sp_data[label].relabel];
		}
	}
	cv::cvtColor(dst, dst, CV_HSV2BGR);
}

void SPD::SaveSuperpixelData(std::vector<sp> sp_data, int frame)
{
	char buff[256];
	sprintf(buff, "Output/CSV/sp_data%d.csv", frame);
	std::ofstream csv(buff);

	csv << "label" << "," << "size" << "," << "center" << "," << "," << "color" << "," << "," << "," << "flow" << std::endl;; // header
	for(int i = 0; i < (int)sp_data.size(); i++)
		csv << i << "," << sp_data[i].size << "," << sp_data[i].center << "," << sp_data[i].colors << "," << sp_data[i].flow << std::endl;
}

// private
double SPD::calcVariance(std::vector<cv::Point2f> raw_flow)
{
	float variance = 0.0;
	cv::Point2f sum_flow = cv::Point2f(0.0, 0.0);
	cv::Point2f ave_flow = cv::Point2f(0.0, 0.0);


	for(int i = 0; i < (int)raw_flow.size(); i++)
		sum_flow += raw_flow[i];

	ave_flow.x = sum_flow.x / (float)raw_flow.size();
	ave_flow.y = sum_flow.y / (float)raw_flow.size();

	for(int i = 0; i < (int)raw_flow.size(); i++)
		variance += (ave_flow.x - raw_flow[i].x) * (ave_flow.y - raw_flow[i].y);

	variance /= (float)raw_flow.size();

	return std::abs(variance);
}

double SPD::CalcDistance(sp data1, sp data2)
{
	double dx = (double)(data1.center.x - data2.center.x);
	double dy = (double)(data1.center.y - data2.center.y);
	double euclid = sqrt(4 * dx * dx + dy * dy);

	double dc0 = (double)(data1.colors[0] - data2.colors[0]);
	double dc1 = (double)(data1.colors[1] - data2.colors[1]);
	double dc2 = (double)(data1.colors[2] - data2.colors[2]);
	double color = sqrt(dc0 * dc0 + dc1 * dc1 + dc2 * dc2);

	double weight = 2.0;
	return (euclid + weight * color);
}