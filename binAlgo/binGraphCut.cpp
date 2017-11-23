#include <opencv2/imgproc/imgproc.hpp>

#include "binarization.h"

using namespace cv;

void getBinMask( const Mat& comMask, Mat& binMask )
{
  if( comMask.empty() || comMask.type()!=CV_8UC1 )
    CV_Error( CV_StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)" );
  if( binMask.empty() || binMask.rows!=comMask.rows || binMask.cols!=comMask.cols )
    binMask.create( comMask.size(), CV_8UC1 );
  binMask = comMask & 1;
  binMask = 255-(binMask*255);
}

Mat equalizeColor( const Mat& comMask )
{
  double min, max;
  minMaxLoc(comMask, &min, &max);
  return (comMask-min)*(255/(max-min));
}

void binariseGrabCut( Mat img )
{
  //first binarize using classical model:
  Mat mask = img.clone();
  cvtColor( img, mask, CV_RGB2GRAY );
  IplImage copyOfImage = mask;
  cv::Mat canny = 255-binarizePerso( &copyOfImage, 0.7, true, false );

  imshow("binarized", mask);

  cv::Mat dist, labels;
  distanceTransform( canny, dist, labels, CV_DIST_L2, CV_DIST_MASK_PRECISE );
  //reprocess the mask to set correct values:
  int error=0;
  for( int i = 0; i < mask.rows; i++ )
  {
    uchar* _dx = mask.ptr<uchar>(i);
    float* _dist = dist.ptr<float>(i);
    for( int j = 0; j < mask.cols; j++ )
    {
      if(_dx[j]<128)
      {
        error++;
        if( error%4 == 0 )
          _dx[j] = GC_PR_FGD;
        else
          _dx[j] = GC_FGD;
      }
      else
      {
        if( _dist[j]<25 )
        {
          error++;
          if( error%2 == 0 )
            _dx[j] = GC_PR_FGD;
          else
            _dx[j] = GC_PR_BGD;
        }
        else
          _dx[j] = GC_BGD;
      }
    }
  }

  waitKey(50);

  Mat bgdModel, fgdModel;

  //now refine using grabcut:
  // /// Convert to grayscale
  Mat eqImg, eqImg1;
  cvtColor( img, eqImg, CV_RGB2GRAY );
  /// Apply Histogram Equalization
  eqImg1 = equalizeColor( eqImg );
  cvtColor( eqImg1, eqImg, CV_GRAY2RGB );
  cv::imshow("eqImg", eqImg1);
  grabCut( eqImg, mask, Rect(), bgdModel, fgdModel, 5, GC_INIT_WITH_MASK );
  grabCut( eqImg, mask, Rect(), bgdModel, fgdModel, 15 );

  Mat binFinal;
  getBinMask(mask, binFinal);
  cvtColor(binFinal, img, CV_GRAY2RGB );
}