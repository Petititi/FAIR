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

//defined in perImage.cpp:
double findBestParameters( string nomAlgo, string inDir, string gtDir);

namespace cv{
  CVAPI( int ) cvHaveImageReader( const char* filename );
}

double compareFM(const string nameGT,const string out){
  Mat GT = imread( nameGT, 0 );
  Mat res = imread( out, 0 );

  if( GT.empty() )
  {
    cout<<"Can't load "<< nameGT <<endl;
    system("pause");
    return 0;
  }
  if( res.empty() )
  {
    cout<<"Can't load "<< out <<endl;
    system("pause");
    return 0;
  }
  if( GT.cols!=res.cols || GT.rows!=res.rows )
  {
    cout<<"error : nbrow||nbcols not equal !"<<endl;
    system("pause");
    return 0;
  }

  double tp1=0.0,fn1=0.0,fp1=0.0;

  int w = GT.cols, h = GT.rows;
  size_t npixels = w * h;

  for (int row =0; row < h; row++){
    unsigned char *testGT=GT.ptr(row);
    unsigned char *testBin=res.ptr(row);
    for(int i=0;i<w;i++){
      if(testGT[i]<=100){
        if(testBin[i]<=100)
          tp1++;//les deux pixels sont bien du texte
        else
          fn1++;//C'est du fond alors que ça devrait être du texte
      }else{
        if(testBin[i]<=100)
          fp1++;//c'est du texte alors que ça devrait etre du fond
      }

    }
  }
  double rc1=(double)tp1/(fn1+tp1);
  double pr1=((double)tp1)/(fp1+tp1);

  double out_dbl = 100*(2.0*rc1*pr1)/(rc1+pr1);
  return MIN( MAX(out_dbl,0), 255);
}

CRITICAL_SECTION mutex;
HANDLE hSemaphore;
double nbThread=0;
double *fmesures;
int nbParams1;
int nbParams2;

vector< string > myFiles;

Mat outMatrix;

typedef struct params_thread{
  int nbParams;
  double *bestParams;
  double *lowParams;
  double *curParams;
  double *stepParams;
  double *highParams;
  double bestFMesure;
  string nomAlgo;
  string inputDir;
  string gtDir;
  bool isDone;
	params_thread( string nomAlgo, string inputDir, string gtDir, int nbParams){
    isDone = false;
    bestFMesure=0;
    this->nomAlgo = nomAlgo;
    this->inputDir = inputDir;
    this->gtDir = gtDir;
    this->nbParams = nbParams;
    bestParams = new double[nbParams];
    lowParams = new double[nbParams];
    curParams = new double[nbParams];
    stepParams = new double[nbParams];
    highParams = new double[nbParams];
  };
} params_thread;

void printMatrix( params_thread* params )
{
  system("cls");
  cout<<"Best value : "<<params->bestFMesure<<" ->";
  for( int i=0; i<params->nbParams; i++ )
  {
    cout<<" "<<params->bestParams[i];
  }
  cout<<endl;

  cout<<"0.000\t";
  for( int j=0; j<=nbParams1; j++ )
    cout<<params->lowParams[0]+j*params->stepParams[0]<<"\t";
  cout<<endl;

  for( int i=1; i<=nbParams2; i++ )
  {
    cout<<params->lowParams[1]+i*params->stepParams[1]<<"\t";
    for( int j=0; j<nbParams1; j++ )
    {
      cout<<((double)fmesures[j*nbParams2 + i])<<"\t";
    }
    cout<<endl;
  }
}

DWORD WINAPI thread(LPVOID param)
{
  params_thread* par = (params_thread*) param;
  double* myParams = new double[par->nbParams];
  int i=0;
  EnterCriticalSection( &mutex );
  ++nbThread;
  //update the parameters:
  for( i=par->nbParams-1; i>=0; i-- )
  {
    par->curParams[i] += par->stepParams[i];
    if( par->curParams[i]>=par->highParams[i] )
      par->curParams[i] = par->lowParams[i];
    else
      break;//we are done!
  }
  if( i<0 )// || par->curParams[0]>par->curParams[1] )
  {
    if( i<0 )
      par->isDone = true;
    nbThread--;
    LeaveCriticalSection( &mutex );
    ReleaseSemaphore(hSemaphore, 1, NULL);
    return 0;
  }

  //create the command line:

  ostringstream nomTemp;
  nomTemp<<par->inputDir<<"img";
  for( i=0; i<par->nbParams; i++ )
    nomTemp<<"_"<<par->curParams[i];
  nomTemp<<".jpg";
  for( i=0; i<par->nbParams; i++ )
  {
    myParams[i] = par->curParams[i];
  }
  LeaveCriticalSection( &mutex );

  STARTUPINFO si;
  PROCESS_INFORMATION pi;

  int idImg = 0;

  double fMesureMean = 0;

  for( idImg = 0; idImg<myFiles.size(); idImg++ )
  {
    ZeroMemory( &si, sizeof( si ) );
    si.cb = sizeof( si );
    ZeroMemory( &pi, sizeof( pi ) );

    string nomImgCurrent = par->inputDir+myFiles[idImg];
    string nomGTCurrent = par->gtDir+myFiles[idImg];

    EnterCriticalSection( &mutex );

    ostringstream line;
    line<<"\""<<par->nomAlgo<<"\" \""<<nomImgCurrent<<"\"";
    line<<" \""<<nomTemp.str()<<"\"";
    for( i=0; i<par->nbParams; i++ )
    {
      line << " " << myParams[i];
    }
    LeaveCriticalSection( &mutex );
    //lancement de la binarisation
    //cout<<line.str()<<endl;
    if( !CreateProcess( NULL, (LPSTR) line.str().c_str(), NULL, NULL, NULL, NULL, NULL, NULL, &si, &pi ) )
    {
      cout<<"createProcess failed : "<<GetLastError() <<endl;
      cout<< line.str() <<endl;
    }

    WaitForSingleObject( pi.hProcess, INFINITE );
    CloseHandle( pi.hProcess );
    CloseHandle( pi.hThread );
    //calcul de la F-Mesure :
    fMesureMean += compareFM( nomGTCurrent, nomTemp.str() );
  }
  fMesureMean /= myFiles.size();

  EnterCriticalSection( &mutex );

  DeleteFile( nomTemp.str().c_str() );//remove the temporary file!


  if( fMesureMean>par->bestFMesure )
  {
    par->bestFMesure = fMesureMean;
    for( i=0; i<par->nbParams; i++ )
    {
      par->bestParams[i] = myParams[i];
    }
  }
  int idx1 = ((myParams[0]-par->lowParams[0])/par->stepParams[0])+0.5;
  int idx2 = ((myParams[1]-par->lowParams[1])/par->stepParams[1])+0.5;
  fmesures[idx1*nbParams2 + idx2] = fMesureMean;

  printMatrix( par );
  if( idx2%nbParams2==1 )
  {
    imwrite( par->inputDir+"Final.jpg", outMatrix );
  }

  nbThread--;

  LeaveCriticalSection( &mutex );
  ReleaseSemaphore(hSemaphore, 1, NULL);

  delete [] myParams;
}

void computeThreads(params_thread *vals)
{
  InitializeCriticalSection( &mutex );

  hSemaphore = CreateSemaphore(
    NULL, //no security attributes
    0L, //initial count
    LONG_MAX, //maximum count (defined in C++ as at least 2147483647)
    NULL); //unnamed semaphore

  for(int i=0; i<4; i++)//suppose 6 processeurs...
    CreateThread( NULL, NULL, thread, vals, NULL, NULL );

  bool stay = true;
  while(stay){
    WaitForSingleObject(hSemaphore,INFINITE);
    CreateThread( NULL, NULL, thread, vals, NULL, NULL );
    EnterCriticalSection( &mutex );
    stay = !vals->isDone;
    LeaveCriticalSection( &mutex );
  }

  EnterCriticalSection( &mutex );
  stay = nbThread>0;
  LeaveCriticalSection( &mutex );
  while(stay)
  {
    WaitForSingleObject(hSemaphore,INFINITE);
    EnterCriticalSection( &mutex );
    stay = nbThread>0;
    LeaveCriticalSection( &mutex );
  }

}

double findBestParameter( string nomAlgo, string inDir, string gtDir){
  params_thread *vals=new params_thread( nomAlgo, inDir, gtDir, 2 );
  int i;
  //19 154
  vals->lowParams[0] = vals->curParams[0] = 2;//30
  vals->lowParams[1] = vals->curParams[1] = 0.6;//15
  vals->highParams[0] = 4;
  vals->highParams[1] = 2;
  vals->stepParams[0] = 0.25;
  vals->stepParams[1] = 0.2;
  double diff = (vals->highParams[0]-vals->lowParams[0]);
  nbParams1 = ( diff/vals->stepParams[0] ) + 0.5;
  diff = (vals->highParams[1]-vals->lowParams[1]);
  nbParams2 = ( diff/vals->stepParams[1] ) + 0.5;

  outMatrix = Mat( nbParams1+1, nbParams2+1, CV_64F );
  fmesures = outMatrix.ptr<double>(0);


  for( i=0; i<nbParams1*nbParams2; i++ )
  {
    fmesures[i]=0;
  }
  //load each images:
  myFiles.push_back("H01.bmp");
  myFiles.push_back("H02.bmp");
  myFiles.push_back("H03.bmp");
  myFiles.push_back("H04.bmp");
  myFiles.push_back("H05.bmp");
  myFiles.push_back("h06.bmp");
  myFiles.push_back("H07.bmp");
  myFiles.push_back("HW1.bmp");
  myFiles.push_back("HW2.bmp");
  myFiles.push_back("HW3.bmp");
  myFiles.push_back("HW4.bmp"); 
  myFiles.push_back("HW5.bmp");
  myFiles.push_back("HW6.bmp");
  myFiles.push_back("HW7.bmp");
  myFiles.push_back("HW8.bmp");
  myFiles.push_back("H_DIBCO_1.bmp");
  myFiles.push_back("H_DIBCO_10.bmp");
  myFiles.push_back("H_DIBCO_11.bmp");
  myFiles.push_back("H_DIBCO_12.bmp");
  myFiles.push_back("H_DIBCO_13.bmp");
  myFiles.push_back("H_DIBCO_14.bmp");
  myFiles.push_back("H_DIBCO_2.bmp");
  myFiles.push_back("H_DIBCO_3.bmp");
  myFiles.push_back("H_DIBCO_4.bmp");
  myFiles.push_back("H_DIBCO_5.bmp");
  myFiles.push_back("H_DIBCO_6.bmp");
  myFiles.push_back("H_DIBCO_7.bmp");
  myFiles.push_back("H_DIBCO_8.bmp");
  myFiles.push_back("H_DIBCO_9.bmp");
  myFiles.push_back("P01.bmp");
  myFiles.push_back("P02.bmp");
  myFiles.push_back("P03.bmp");
  myFiles.push_back("P04.bmp");
  myFiles.push_back("P05.bmp");
  myFiles.push_back("P06.bmp");
  myFiles.push_back("P07.bmp");
  myFiles.push_back("PR1.bmp");
  myFiles.push_back("PR2.bmp");
  myFiles.push_back("PR3.bmp");
  myFiles.push_back("PR4.bmp");
  myFiles.push_back("PR5.bmp");
  myFiles.push_back("PR6.bmp");
  myFiles.push_back("PR7.bmp");
  myFiles.push_back("PR8.bmp");
  myFiles.push_back("_H01.bmp");
  myFiles.push_back("_H02.bmp");
  myFiles.push_back("_H03.bmp");
  myFiles.push_back("_H04.bmp");
  myFiles.push_back("_H05.bmp");
  myFiles.push_back("_H06.bmp");
  myFiles.push_back("_H07.bmp");
  myFiles.push_back("_H08.bmp");
  myFiles.push_back("_H09.bmp");
  myFiles.push_back("_H10.bmp");
  /*
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
      myFiles.push_back( just_the_name );
    iter++;
  }*/

  computeThreads( vals );
  double bestF = vals->bestFMesure;


  cout<<bestF<<" ->";
  for( i=0; i<vals->nbParams; i++ )
  {
    cout<<" "<<vals->bestParams[i];
  }
  cout<<endl;

  imwrite( vals->inputDir+"Final.jpg", outMatrix );

  delete vals;
  return bestF;
}

void getNBP( string nomAlgo, string inDir, string gtDir)
{
  myFiles.push_back("H01.bmp");
  myFiles.push_back("H02.bmp");
  myFiles.push_back("H03.bmp");
  myFiles.push_back("H04.bmp");
  myFiles.push_back("H05.bmp");
  myFiles.push_back("h06.bmp");
  myFiles.push_back("H07.bmp");
  myFiles.push_back("HW1.bmp");
  myFiles.push_back("HW2.bmp");
  myFiles.push_back("HW3.bmp");
  myFiles.push_back("HW4.bmp"); 
  myFiles.push_back("HW5.bmp");
  myFiles.push_back("HW6.bmp");
  myFiles.push_back("HW7.bmp");
  myFiles.push_back("HW8.bmp");
  myFiles.push_back("H_DIBCO_1.bmp");
  myFiles.push_back("H_DIBCO_10.bmp");
  myFiles.push_back("H_DIBCO_11.bmp");
  myFiles.push_back("H_DIBCO_12.bmp");
  myFiles.push_back("H_DIBCO_13.bmp");
  myFiles.push_back("H_DIBCO_14.bmp");
  myFiles.push_back("H_DIBCO_2.bmp");
  myFiles.push_back("H_DIBCO_3.bmp");
  myFiles.push_back("H_DIBCO_4.bmp");
  myFiles.push_back("H_DIBCO_5.bmp");
  myFiles.push_back("H_DIBCO_6.bmp");
  myFiles.push_back("H_DIBCO_7.bmp");
  myFiles.push_back("H_DIBCO_8.bmp");
  myFiles.push_back("H_DIBCO_9.bmp");
  myFiles.push_back("P01.bmp");
  myFiles.push_back("P02.bmp");
  myFiles.push_back("P03.bmp");
  myFiles.push_back("P04.bmp");
  myFiles.push_back("P05.bmp");
  myFiles.push_back("P06.bmp");
  myFiles.push_back("P07.bmp");
  myFiles.push_back("PR1.bmp");
  myFiles.push_back("PR2.bmp");
  myFiles.push_back("PR3.bmp");
  myFiles.push_back("PR4.bmp");
  myFiles.push_back("PR5.bmp");
  myFiles.push_back("PR6.bmp");
  myFiles.push_back("PR7.bmp");
  myFiles.push_back("PR8.bmp");
  myFiles.push_back("_H01.bmp");
  myFiles.push_back("_H02.bmp");
  myFiles.push_back("_H03.bmp");
  myFiles.push_back("_H04.bmp");
  myFiles.push_back("_H05.bmp");
  myFiles.push_back("_H06.bmp");
  myFiles.push_back("_H07.bmp");
  myFiles.push_back("_H08.bmp");
  myFiles.push_back("_H09.bmp");
  myFiles.push_back("_H10.bmp");

  double totalW = 0, totalH = 0;
  for( int i=0; i<myFiles.size(); i++)
  {
    Mat img = imread(inDir+myFiles[i]);
    totalW += (img.rows*img.cols)/1000000.0;
  }
  cout<<totalW<<endl;
}

//---------------------------------------------------------------------------------------------
int main( int argc, char **argv )
{
  /*
	if(argc<5){
		cout<<"number of args < 4... Abording"<<endl;
		return 0;
  }*/

  findBestParameter("D:\\Travail\\Binarization\\PAMI\\VisualStudio\\bin\\Release\\evaluation0.0.1.exe",
    "D:\\Travail\\Binarization\\data\\",
    "D:\\Travail\\Binarization\\data\\Comparatif\\");
  //Release
  /*getNBP( "D:\\Travail\\These\\Binarisation rapide\\PAMI\\VisualStudio\\bin\\Release\\binarize0.0.1.exe",
    ((string)"D:\\Travail\\Binarization\\data\\"),
    ((string)"D:\\Travail\\Binarization\\Comparatif\\") );*/
  return 0;
}





