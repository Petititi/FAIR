#include <iostream>
#include <sstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//#include <boost/filesystem.hpp>   // includes all needed Boost.Filesystem declarations

#include <Windows.h>
#include <vector>

//using boost::filesystem::path;

using namespace std;
using namespace cv;

namespace cv{
  CVAPI( int ) cvHaveImageReader( const char* filename );
}

//defined in main.cpp
double compareFM(const string nameGT,const string out);

CRITICAL_SECTION myMutex;
HANDLE mySemaphore;
double nbThreadRunning=0;

typedef struct params_per_image{
  int nbParams;
  vector<double *>bestParams;
  double *lowParams;
  int curImage;
  double *stepParams;
  double *highParams;
  vector<double> bestFMesure;
  vector<string> myFiles;
  string nomAlgo;
  string inputDir;
  string gtDir;
  bool isDone;
	params_per_image( string nomAlgo, string inputDir, string gtDir, int nbParams){
    isDone = false;
    curImage = 0;
    this->nomAlgo = nomAlgo;
    this->inputDir = inputDir;
    this->gtDir = gtDir;
    this->nbParams = nbParams;
    lowParams = new double[nbParams];
    stepParams = new double[nbParams];
    highParams = new double[nbParams];
  };
} params_thread;

DWORD WINAPI thread_one_image(LPVOID param)
{

  int i=0;

  params_per_image* par = (params_per_image*) param;
  double* myParams = new double[par->nbParams];
  for( i=0; i<par->nbParams; i++ )
    myParams[i] = par->lowParams[i];

  EnterCriticalSection( &myMutex );
  ++nbThreadRunning;
  int myImg = par->curImage++;
  if( myImg>=par->myFiles.size() )
  {
    par->isDone = true;
    nbThreadRunning--;
    LeaveCriticalSection( &myMutex );
    ReleaseSemaphore(mySemaphore, 1, NULL);
    return 0;
  }
  par->bestParams.push_back(new double[par->nbParams]);
  par->bestFMesure.push_back( 0 );
  LeaveCriticalSection( &myMutex );

  STARTUPINFO si;
  PROCESS_INFORMATION pi;

  string nomImgCurrent = par->inputDir+par->myFiles[myImg];
  string nomGTCurrent = par->gtDir+par->myFiles[myImg];

  double fMesureTMP = 0;
  while( i>=0 )// || par->curParams[0]>par->curParams[1] )
  {

    //create the command line:
    ostringstream nomTemp;
    nomTemp<<par->inputDir<<"img"<<myImg<<".jpg";

    ZeroMemory( &si, sizeof( si ) );
    si.cb = sizeof( si );
    ZeroMemory( &pi, sizeof( pi ) );

    ostringstream line;
    line<<par->nomAlgo<<" \""<<nomImgCurrent<<"\"";
    line<<" \""<<nomTemp.str()<<"\"";
    for( i=0; i<par->nbParams; i++ )
      line << " " << myParams[i];

    //lancement de la binarisation
    //cout<<line.str()<<endl;
    if( !CreateProcess( NULL, (LPSTR) line.str().c_str(), NULL, NULL, NULL, NULL, NULL, NULL, &si, &pi ) )
      cout<<"createProcess failed : "<<GetLastError() <<endl;

    WaitForSingleObject( pi.hProcess, INFINITE );
    CloseHandle( pi.hProcess );
    CloseHandle( pi.hThread );

    //calcul de la F-Mesure :
    fMesureTMP = compareFM( nomGTCurrent, nomTemp.str() );

    if( fMesureTMP>par->bestFMesure[myImg] )
    {
      par->bestFMesure[myImg] = fMesureTMP;
      for( i=0; i<par->nbParams; i++ )
      {
        par->bestParams[myImg][i] = myParams[i];
      }
    }

    DeleteFile( nomTemp.str().c_str() );//remove the temporary file!


    //update the parameters:
    for( i=par->nbParams-1; i>=0; i-- )
    {
     myParams[i] += par->stepParams[i];
      if( myParams[i]>=par->highParams[i] )
        myParams[i] = par->lowParams[i];
      else
        break;//we are done!
    }
  }
  delete [] myParams;

  EnterCriticalSection( &myMutex );
  cout<<par->myFiles[myImg]<<"\t Best FM: "<<par->bestFMesure[myImg];
  for( i=0; i<par->nbParams; i++ )
    cout<<"\t "<<par->bestParams[myImg][i];
  cout<<endl;

  nbThreadRunning--;
  LeaveCriticalSection( &myMutex );
  ReleaseSemaphore(mySemaphore, 1, NULL);
  return 0;

}

void runThreads(params_thread *vals)
{
  InitializeCriticalSection( &myMutex );

  mySemaphore = CreateSemaphore(
    NULL, //no security attributes
    0L, //initial count
    LONG_MAX, //maximum count (defined in C++ as at least 2147483647)
    NULL); //unnamed semaphore

  for(int i=0; i<4; i++)//suppose 6 processeurs...
    CreateThread( NULL, NULL, thread_one_image, vals, NULL, NULL );

  bool stay = true;
  while(stay){
    WaitForSingleObject(mySemaphore,INFINITE);
    CreateThread( NULL, NULL, thread_one_image, vals, NULL, NULL );
    EnterCriticalSection( &myMutex );
    stay = !vals->isDone;
    LeaveCriticalSection( &myMutex );
  }

  EnterCriticalSection( &myMutex );
  stay = nbThreadRunning>0;
  LeaveCriticalSection( &myMutex );
  while(stay)
  {
    WaitForSingleObject(mySemaphore,INFINITE);
    EnterCriticalSection( &myMutex );
    stay = nbThreadRunning>0;
    LeaveCriticalSection( &myMutex );
  }

}

double findBestParameters( string nomAlgo, string inDir, string gtDir){
  params_per_image *vals=new params_per_image( nomAlgo, inDir, gtDir, 1 );
  int i;
  //19 154
  vals->lowParams[0] = 0.68;//30
  vals->highParams[0] = 1.31;
  vals->stepParams[0] = 0.01;

  vals->myFiles.push_back("H01.bmp");
  vals->myFiles.push_back("H02.bmp");
  vals->myFiles.push_back("H03.bmp");
  vals->myFiles.push_back("H04.bmp");
  vals->myFiles.push_back("H05.bmp");
  vals->myFiles.push_back("h06.bmp");
  vals->myFiles.push_back("H07.bmp");
  vals->myFiles.push_back("P01.bmp");
  vals->myFiles.push_back("P02.bmp");
  vals->myFiles.push_back("P03.bmp");
  vals->myFiles.push_back("P04.bmp");
  vals->myFiles.push_back("P05.bmp");
  vals->myFiles.push_back("P06.bmp");
  vals->myFiles.push_back("P07.bmp");
  /*
  //load each images:
  path dirTmp( inDir.c_str( ) );
  if ( !boost::filesystem::exists( dirTmp ) || !boost::filesystem::is_directory( dirTmp ) ) {
    return 0.;
  }


  boost::filesystem::directory_iterator iter= boost::filesystem::directory_iterator( dirTmp );
  while( iter != boost::filesystem::directory_iterator( ) )
  {
    string name_of_file = iter->path( ).string( );
    string just_the_name = iter->path().filename().string( );
    //if( cv::cvHaveImageReader( (const char* )name_of_file.c_str( ) ) )
    if( just_the_name.find( ".bmp" ) != string::npos )
      vals->myFiles.push_back( just_the_name );
    iter++;
  }*/

  runThreads( vals );

  delete vals;
  return 0;
}




