#define _CRTDBG_MAP_ALLOC

#include <stdlib.h>
#include <crtdbg.h>

#include "binarization.h"
#include <sstream>
#include <opencv2/highgui/highgui.hpp>


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>

#include <boost/filesystem.hpp>
using namespace boost::filesystem;

using cv::Mat;
using cv::Size;
using cv::Vec;
using cv::Ptr;
using cv::VideoCapture;
using std::string;
using cv::imread;
using std::ostringstream;
using std::vector;


using namespace std;
using namespace cv;

char idAlgo = -1;//always start with our algo

int handleInputs( int inputState)
{
  int key =0;
  if( inputState & 0x01 )
    key = waitKey(0);
  else
    key = waitKey(25);

  if( key==27 )
  {
    inputState = -1;
  }

  if( key==' ' )
  {//switch between freezed and continus mode
    if( (inputState & 0x01) == 0 )
    {
      cout<<"entering freeze mode"<<endl;
      inputState |= 0x01;
    }
    else
    {
      cout<<"leaving freeze mode"<<endl;
      inputState &= 0xFE;
    }
  }

  if( key=='c' )
  {
    idAlgo = (idAlgo+1)%3;//3 algos différents
  }
  if( key=='v' )
  {
    idAlgo = 0;//3 algos différents
  }
  if( key=='b' )
  {
    idAlgo = 1;//3 algos différents
  }
  if( key=='n' )
  {
    idAlgo = 2;//3 algos différents
  }
  return inputState;
}

void usingWecam( int idWebcam)
{
  cout<<"using webcam id = "<<idWebcam<<endl;
  //first try to load images from webcam :
  VideoCapture *capture_ = new VideoCapture();
  if( capture_->open( idWebcam ))
  {
    capture_->set( CV_CAP_PROP_FRAME_HEIGHT,640 );
    capture_->set( CV_CAP_PROP_FRAME_WIDTH,480 );
  };
  cv::Mat myImage;
  capture_->retrieve( myImage );
  bool stay = true;
  char myState = 0;
  while( !myImage.empty() && stay )
  {
    myState = handleInputs(myState);
    Mat grayImg;
    cvtColor( myImage,grayImg,CV_RGB2GRAY );
    IplImage copyOfImage = grayImg;
    if( myState>=0 )
    {
      if( myState&0x01 )//freezed...
      {
        cvDestroyWindow( "Converted flux" );
        Mat workingImg;
        workingImg = grayImg.clone(); copyOfImage = workingImg;
        binarizeOtsu( &copyOfImage );
        imshow("Otsu", workingImg);
        workingImg = grayImg.clone(); copyOfImage = workingImg;
        binarizeSauvola( &copyOfImage );
        imshow("Sauvola", workingImg);
        workingImg = grayImg.clone(); copyOfImage = workingImg;
        binarizePerso( &copyOfImage, 0.25, true, true );
        imshow("Notre solution", workingImg);
        while( myState&0x01 )//wait for a defreeze...
          myState = handleInputs(myState);
        cvDestroyWindow( "Otsu" );
        cvDestroyWindow( "Sauvola" );
        cvDestroyWindow( "Notre solution" );
      }
      else
      {
        switch(idAlgo)
        {
        case 0:
          binarizeOtsu( &copyOfImage );
          break;
        case 1:
          binarizeSauvola( &copyOfImage );
          break;
        default:
          binarizePerso( &copyOfImage, 0.2, false, false );
        }
        imshow("Converted flux", grayImg);
      }

      imshow("Original flux", myImage);
      if( (myState&0x01)==0 )
        capture_->retrieve( myImage );
    }
    else
      stay = false;
  }

  capture_->release();
  delete capture_;
}

cv::Mat inversColor(cv::Mat input)
{
  Mat tmp;
  cv::cvtColor( input,tmp,CV_RGB2Lab );
  // split the image into separate color planes
  vector<Mat> planes;
  split(tmp, planes);
  planes[0] = 255-planes[0];
  merge(planes, tmp);
  cv::cvtColor( tmp,tmp,CV_Lab2RGB );
  return tmp;
}

cv::Mat usingImage( string fileName, bool rescale = false, bool fullQuality = true, bool debugResult = false )
{
  cv::Mat img = imread(fileName);
  if( rescale )
    cv::resize( img, img, Size( 1536, 1024 ) );

  IplImage copyOfImage = img;
  binarizePerso( &copyOfImage, 0.4, true, fullQuality );
  /*
  findText(img);
  cv::waitKey();
  Mat inv_mask = inversColor(img);
  findText(inv_mask);
  */
  //binariseGrabCut( img );

  if( debugResult )
  {
    imshow("Binarized", img);
    int key = cv::waitKey();
    if( key == 'i' ||  key == 'I' )
    {
      img = imread(fileName, 0);
      if( rescale )
        cv::resize( img, img, Size( 1536, 1024 ) );
      img = 255-img;
      IplImage copyOfImage = img;
      binarizePerso( &copyOfImage, 0.5, true, fullQuality );
      imshow("Binarized", img);
      cv::waitKey();
    }
  }
  return img;
}

void usingFolder( string folderName, string outputFolder, bool debugResult = false )
{
  path dirTmp( folderName.c_str( ) );
  if ( !boost::filesystem::exists( dirTmp ) || !boost::filesystem::is_directory( dirTmp ) )
    return;


  cv::Mat img;
  boost::filesystem::directory_iterator iter= boost::filesystem::directory_iterator( dirTmp );
  while( iter != boost::filesystem::directory_iterator( ) )
  {
    string name_of_file = iter->path( ).string( );
    if( !boost::filesystem::is_directory( path(name_of_file) ) )
    {
      string just_the_name = iter->path().filename().string();
      if( just_the_name.find(".bmp")!=string::npos || just_the_name.find(".png")!=string::npos || just_the_name.find(".jpg")!=string::npos )
      {
        cout<<just_the_name<<endl;
        img = usingImage( name_of_file, false, true, false );
        if( debugResult )
        {
          cv::waitKey();
        }
        else
          cv::imwrite( outputFolder+just_the_name, img );
      }
    }
    iter++;
  }
}

int main(int argc, char **argv)
{
  //cout<<"DIBCO'11, Team: Lelore-Bouchara. Computation in progress...\n";
  IplImage *finalImg;
  IplImage *finalImgScaled;
  if(argc<=2){
    /*
    int idWebcam = 0;
    if( argc>1 )
    idWebcam = atoi( argv[1] );
    usingWecam(idWebcam);
    */
    if( true )
    {
      string nameOfFolder ="D:\\Travail\\Binarization\\data\\";
      //string nameOfFolder ="D:\\Travail\\documents\\concours\\robustReading\\challenge1\\Images\\";

      string outputFolder =nameOfFolder + "binPAMI\\";
      usingFolder( nameOfFolder, outputFolder, false );
    }
    else
    {
      string nameOfImg ="D:\\Travail\\Binarization\\data\\_H01.bmp";
      imshow( "tt", usingImage(nameOfImg ) );
    }
    waitKey(0);
    return 0;
  }

  //double param1 = 0.63, param2 = 0.85;
  double param1 = 3, param2 = 1.2;
  if( argc==4 )
  {
    stringstream param1Str;
    param1Str<<argv[3];
    param1Str>>param1;
  }
  if( argc==5 )
  {
    stringstream param1Str, param2Str;
    param1Str<<argv[3];
    param1Str>>param1;
    param2Str<<argv[4];
    param2Str>>param2;
  }
  string imgIn = argv[1];
  string imgOut = argv[2];

  cv::Mat img = imread(imgIn, 0);
  IplImage copyOfImage = img;
  binarizePerso( &copyOfImage, param1, param2, true, true );

#ifdef _DEBUG
  cvShowImage("final", &copyOfImage);
  waitKey(0);
#endif // _DEBUG
  imwrite(imgOut, img);

}
