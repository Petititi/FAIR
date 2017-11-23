#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include "binarization.h"

using namespace cv;

Scalar randomColor( RNG& rng )
{
  int icolor = (unsigned) rng;
  return Scalar( icolor&255, (icolor>>8)&255, (icolor>>16)&255 );
}

vector<Scalar> computeCC_Color(cv::Mat img, cv::Mat binaryMask, cv::Mat& labels)
{
  //now get connected componnents:
  Mat stats, centroids;
  connectedComponentsWithStats( binaryMask, labels, stats, centroids, 4 );

  //now remove unwanted CC :
  vector<int> newLabels;
  int newLabel = 0;
  for(int i=0; i<stats.rows; i++ )
  {
    int area = stats.at<int>( i, CC_STAT_AREA );
    if( area>400 || area<10 )//remove this label:
      newLabels.push_back(0);
    else
      newLabels.push_back(newLabel++);
  }
  if( newLabel<=2 )//problem... keep every CC:
  {
    newLabel = stats.rows;
    for(int i=0; i<stats.rows; i++ )
      newLabels[i] = i;
  }

  //compute color of each componnents:
  vector<Scalar> ccColors;
  vector<int> nbIndex;
  //create colors:
  for(int i=0; i<newLabel; i++ )
  {
    ccColors.push_back( Scalar(0) );
    nbIndex.push_back(0);
  }

  for( int i = 0; i < binaryMask.rows; i++ )
  {
    uchar* input = img.ptr<uchar>( i );
    int* labelID=labels.ptr<int>( i );
    for( int j = 0; j < binaryMask.cols; j++ )
    {
      int idLab = labelID[j] = newLabels[labelID[j]];
      Scalar &meanColor = ccColors[idLab];

      nbIndex[idLab]++;
      meanColor[0] += input[j*3];
      meanColor[1] += input[j*3+1];
      meanColor[2] += input[j*3+2];
    }
  }
  for(int i=0; i<newLabel; i++ )
  {
    Scalar &meanColor = ccColors[i];
    double nbVal = nbIndex[i];
    meanColor[0] = meanColor[0]/nbVal;
    meanColor[1] = meanColor[1]/nbVal;
    meanColor[2] = meanColor[2]/nbVal;
  }
  return ccColors;
}

struct Stat_Cluster
{
  Scalar mean;
  double var;
  unsigned int nbCC;
  Stat_Cluster(){mean = Scalar(0); var=0; nbCC=0;};
};

vector<Stat_Cluster> computeStat( vector<Scalar>& colors, vector<int>& labels, int nbCC )
{
  vector<Stat_Cluster> outStats;
  for(int i=0; i<nbCC; i++)
    outStats.push_back( Stat_Cluster() );

  for(int i=0; i<colors.size(); i++)
  {
    outStats[labels[i]].nbCC++;
    outStats[labels[i]].mean[0] += colors[i][0];
    outStats[labels[i]].mean[1] += colors[i][1];
    outStats[labels[i]].mean[2] += colors[i][2];
  }
  for(int i=0; i<nbCC; i++)
  {
    outStats[i].mean[0] /= outStats[i].nbCC;
    outStats[i].mean[1] /= outStats[i].nbCC;
    outStats[i].mean[2] /= outStats[i].nbCC;
  }
  //variances:
  for(int i=0; i<colors.size(); i++)
  {
    double tmpVar = 0;
    tmpVar += pow( outStats[labels[i]].mean[0] - colors[i][0], 2);
    tmpVar += pow( outStats[labels[i]].mean[1] - colors[i][1], 2);
    tmpVar += pow( outStats[labels[i]].mean[2] - colors[i][2], 2);
    outStats[labels[i]].var+=sqrt(tmpVar);
  }
  for(int i=0; i<nbCC; i++)
    outStats[i].var/=outStats[i].nbCC;

  return outStats;
}
double findBiggestVariance( vector<Scalar>& colors, vector<int>& labels, int nbCC )
{
  vector<Stat_Cluster> stats = computeStat(colors,labels,nbCC);
  //find max:
  double maxVal = 0;
  for(int i=0; i<nbCC; i++)
  {
    if(stats[i].var>maxVal)
      maxVal = stats[i].var;
  }
  return maxVal;
}

//utilise un mélange de gaussiennes pour trouver le bon nombre de CC:
vector<int> computeCorrespondingColor(vector<Scalar>& colors, float maxDist)
{
  int nbGaussiennes = 2;//commence avec 2 couleurs différentes
  vector<int> labels;
  Mat colorsMat = Mat::zeros(Size( 3, colors.size() ), CV_64F);
  for(int i=0; i<colors.size(); i++)
  {
    colorsMat.at<double>(i, 0) = colors[i][0];
    colorsMat.at<double>(i, 1) = colors[i][1];
    colorsMat.at<double>(i, 2) = colors[i][2];
  }

  double modelError = 150;
  while ( modelError>40 && nbGaussiennes<20 )
  {
    Mat logLikelihoods;
    EM cluster(nbGaussiennes);
    cluster.train( colorsMat, logLikelihoods, labels );
    double* likelihoods = logLikelihoods.ptr<double>();
    modelError = findBiggestVariance( colors, labels, nbGaussiennes);
    nbGaussiennes++;
  }
  return labels;
}

void findText( cv::Mat img )
{
  //first binarize the input:
  Mat mask = img.clone(), labels;
  if( img.channels()>1 )
    cvtColor( img, mask, CV_RGB2GRAY );
  cv::imshow("output", img);

  IplImage copyOfImage = mask;
  binarizePerso( &copyOfImage, 0.3, true, true );

  vector<Scalar> ccColors = computeCC_Color(img, mask, labels);
  vector<int> correspondingClass = computeCorrespondingColor(ccColors, 32);

  //find nbClass:
  int nbClass = 0;
  for ( int idClass : correspondingClass )
  {
    if( nbClass<idClass )
      nbClass = idClass;
  }
  ++nbClass;

  vector<Stat_Cluster> stats = computeStat(ccColors,correspondingClass,nbClass);
  //find text CC:
  int idCC_max = 0;
  int nbMax_CC = 0;
  for ( int i=0; i<nbClass; i++ )
  {
    Stat_Cluster stat = stats[i];
    if( nbMax_CC<stat.nbCC )
    {
      idCC_max = i;
      nbMax_CC = stat.nbCC;
    }
  }

  vector<Scalar> colors;
  RNG rng( 0xFFFFFFFF );
  //create colors:
  for(int i=0; i<nbClass; i++ )
    colors.push_back( randomColor(rng) );

  Mat colorImgWithCC = Mat::zeros(mask.size(), CV_8UC3);
  //now draw an image with these connected componnents:
  for( int i = 0; i < mask.rows; i++ )
  {
    uchar* _dx = colorImgWithCC.ptr<uchar>( i );
    int* labelID=labels.ptr<int>( i );
    for( int j = 0; j < mask.cols; j++ )
    {
      if( labelID[j] == 0 )
      {
        _dx[j*3] = 0;
        _dx[j*3+1] = 0;
        _dx[j*3+2] = 0;
      }
      else
      {
        int idLab = correspondingClass[labelID[j]];
        Scalar &meanColor = colors[idLab];
        _dx[j*3] = colors[idLab][0];
        _dx[j*3+1] = colors[idLab][1];
        _dx[j*3+2] = colors[idLab][2];
      }

    }
  }
  
  Mat greyDiff = Mat::zeros(mask.size(), CV_8UC1);
  Scalar& meanColor1 = stats[idCC_max].mean;
  for( int i = 0; i < mask.rows; i++ )
  {
    uchar* input = img.ptr<uchar>( i );
    uchar* _dx = greyDiff.ptr<uchar>( i );
    int* labelID=labels.ptr<int>( i );
    for( int j = 0; j < mask.cols; j++ )
    {
      int idLab = labelID[j];
      Scalar meanColor(input[j*3], input[j*3+1], input[j*3+2]);

      float dist = 0;
      dist+=abs(meanColor[0] - meanColor1[0]);
      dist+=abs(meanColor[1] - meanColor1[1]);
      dist+=abs(meanColor[2] - meanColor1[2]);
      if( dist>255 )
        dist=255;
      _dx[j] = dist;
    }
  }
  

  imshow("Blobs", colorImgWithCC);
  
  Mat greyClone;
  resize(greyDiff, greyClone, Size(), 2, 2, INTER_LANCZOS4 );
  greyDiff = greyClone.clone();
  imshow("greyDiff", greyDiff);

  IplImage copyOfImage1 = greyClone;
  binarizePerso( &copyOfImage1, 0.3, true, false );

  imshow("BinaryFinal", greyClone);
}