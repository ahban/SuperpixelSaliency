//
//  main.cpp
//  SLIC_Superpixels
//
//  Created by Takayama Shota on 6/8/15.
//  Copyright (c) 2015 Takayama Shota. All rights reserved.
//

#include "Define.h"



int main() {
    
    for(int frame = 0; frame < 300; frame++) {
        
        char DATA_SET[] = "UCF";
        cv::Mat src1, src2;
        if(ReadImage(src1, src2, frame, DATA_SET) == false) {
            std::cout << "No image" << std::endl;
            return -1;
        }
//        src1 = cv::imread("/Users/takayama/Research/Data/test.png", 1);
//        src2 = cv::imread("/Users/takayama/Research/Data/test.png", 1);
        cv::Mat flowMat(src1.size(), CV_32FC2);
        std::vector<cv::Point2f> prevPt, nextPt;
        flow.calcOpticalFlow(src1, src2, flowMat, prevPt, nextPt, 1);
        
        cv::Mat slic = src1.clone();
        int numlabels = 0;
        int* klabels = NULL;
        if(k > 0) {
            segment.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(slic, slic.cols, slic.rows, klabels, numlabels, k, m);
            segment.DrawContoursAroundSegments(slic, klabels, slic.cols, slic.rows);
        }
        else
            return -1;
        
        std::vector<sp> sp_data(numlabels);
        spd.SP_Feature(sp_data, src1, flowMat, klabels);
        
        std::vector<std::vector<l2_id>> l2;
        l2.resize(numlabels);
        for(int i = 0; i < l2.size(); i++) {
            l2[i].resize(numlabels);
        }
        for(int i = 0; i < l2.size(); i++) {
            for(int j = 0; j < l2.size(); j++) {
                float dx = float(sp_data[i].center.x - sp_data[j].center.x);
                float dy = float(sp_data[i].center.y - sp_data[j].center.y);
                l2[i][j].delta = sqrtf(dx * dx + dy * dy);
                l2[i][j].id = sp_data[j].orilabel;
            }
        }
        for(int i = 0; i < l2.size(); i++) {
            std::sort(l2[i].begin(), l2[i].end());
        }
        std::vector<float> diff_rg(numlabels), diff_by(numlabels), diff_intensity(numlabels);
        for(int i = 0; i < l2.size(); i++) {
            for(int j = 0; j < 10; j++) {
                diff_rg[i] += std::fabs(float(sp_data[i].rg - sp_data[l2[i][j].id].rg));
                diff_by[i] += std::fabs(float(sp_data[i].by - sp_data[l2[i][j].id].by));
                diff_intensity[i] += std::fabs(float(sp_data[i].intensity - sp_data[l2[i][j].id].intensity));
            }
        }
//        for(int y = 0; y < height; y++) {
//            for(int x = 0; x < width; x++) {
//                
//            }
//        }
        
        cv::Mat conspI(src1.size(), CV_32F, cv::Scalar(0.0f));
        cv::Mat conspC(src1.size(), CV_32F, cv::Scalar(0.0f));
        
        for(int y = 0; y < src1.rows; y++) {
            for(int x = 0; x < src1.cols; x++ ) {
                int index = y * src1.cols + x;
                int label = klabels[index];
                conspI.at<float>(y, x) = diff_intensity[label];
                conspC.at<float>(y, x) = (diff_rg[label] + diff_by[label]) / 2.0f;
            }
        }
        normalizeRange(conspI); cv::imwrite("/Users/takayama/Desktop/conspI.png", conspI * 255.0f);
        normalizeRange(conspC); cv::imwrite("/Users/takayama/Desktop/conspC.png", conspC * 255.0f);
        cv::Mat saliency = (conspI + conspC) / 2.0f;

        cv::imwrite("/Users/takayama/Research/Data/Result/sp_saliency.png", saliency * 255.0f);
        cv::imshow("src", src1); cv::imshow("result", slic); cv::imshow("conspI", conspI); cv::imshow("conspC", conspC); cv::imshow("saliency", saliency);
        cv::waitKey(0);
    }
    
    return 0;
}
