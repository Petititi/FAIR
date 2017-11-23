#ifndef _BINARIZATION_H_
#define _BINARIZATION_H_

#include "OpImg.h"
#include "connected.h"
//#include "MixingImage.h"
#include <sstream>

#include <opencv2/core/core.hpp>


double findCannyTWithOtsu( IplImage * image );
PileCanny * seuilCanny( const IplImage* srcarr, IplImage* dstarr,
  int demiWindow, double k=1, double p=0.4 );
IplImage* computeImg( IplImage *frame, double k=1, int window=12 );
void initGlobalVars( int x, int y );

IplImage * Threshold_Otsu( IplImage * image );

void binarizeOtsu( IplImage *frame );
void binarizeSauvola( IplImage *frame );
cv::Mat binarizePerso( IplImage *frame, float factor1, float factor2, bool postFilter=true, bool rescale=true );

void binariseGrabCut( cv::Mat img );

void findText( cv::Mat img );

#endif