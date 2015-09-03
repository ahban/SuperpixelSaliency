//
//  Define.h
//  SLIC_Superpixels
//
//  Created by Takayama Shota on 6/8/15.
//  Copyright (c) 2015 Takayama Shota. All rights reserved.
//

#ifndef SLIC_Superpixels_Define_h
#define SLIC_Superpixels_Define_h

#include <opencv2/opencv.hpp>
#include "OpticalFlow.h"
#include "SLIC.h"
#include "SuperpixelHandling.h"
#include <ctype.h>
#include <iostream>
#include <vector>
#include <cmath>

#define k 3000
#define m 10.0

FLOW flow;
SLIC segment;
SPD spd;

bool ReadImage(cv::Mat& img1,
               cv::Mat& img2,
               int frame,
               char DATA_SET[])
{
    char i_file[256];
    
    if(strcmp(DATA_SET, "UCF") == 0) {
        sprintf(i_file, "/Users/takayama/Research/Data/879-38/879-38_%05d.png", frame);
        img1 = cv::imread(i_file, 1);
        sprintf(i_file, "/Users/takayama/Research/Data/879-38/879-38_%05d.png", frame + 1);
        img2 = cv::imread(i_file, 1);
    }
    
    else if(strcmp(DATA_SET, "PETS2009") == 0) {
        sprintf(i_file, "/Users/takayama/Research/Data/S2_L1_Time_12-34_View_001/frame_%04d.jpg", frame);
        img1 = cv::imread(i_file, 1);
        sprintf(i_file, "/Users/takayama/Research/Data/S2_L1_Time_12-34_View_001/frame_%04d.jpg", frame + 1);
        img2 = cv::imread(i_file, 1);
    }
    
    
    if(img1.empty() || img2.empty())
        return false;
    
    else
        return true;
}

struct l2_id {
    float delta;
    int id;
};

bool operator<(const l2_id& left, const l2_id& right)
{
    return left.delta < right.delta;
}

bool operator>(const l2_id& left, const l2_id& right)
{
    return left.delta > right.delta;
}

// normalize [0,1]
void normalizeRange(cv::Mat& image) {
    double minval = 0.0, maxval = 0.0;
    cv::minMaxLoc(image, &minval, &maxval);
    
    image -= minval;
    if(minval < maxval)
        image /= maxval - minval;
}


#endif
