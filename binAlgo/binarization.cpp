#include "binarization.h"

#pragma warning(disable:4251)
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;


IplImage *greyLevels = NULL;  
IplImage *canny = NULL;
CvMat *labelimg = NULL;
IplImage *gradImg = NULL;
void initGlobalVars(int x,int y){
  if( canny!=NULL )
    cvReleaseImage(&canny);
  canny = cvCreateImage(cvSize(x,y),8,1);
  if( greyLevels!=NULL )
    cvReleaseImage(&greyLevels);
  greyLevels = cvCreateImage(cvSize(x,y),8,1);
  if( labelimg!=NULL )
    cvReleaseMat(&labelimg);
  labelimg = cvCreateMat(y,x,CV_32SC1);
  if( gradImg!=NULL )
    cvReleaseImage(&gradImg);
  gradImg=cvCreateImage(cvSize(x,y),8,1);
}

double findCannyTWithOtsu( IplImage * image)
{


	int hist_size[] = {255};
	CvHistogram* hist;

	hist = cvCreateHist( 1, hist_size, CV_HIST_ARRAY);
	cvCalcHist( &image, hist, 0, 0 );
	double max_val = 0;

	CV_FUNCNAME( "icvGetThreshVal_Otsu" );

	__BEGIN__;

	int i, count;
	const float* h;
	double sum = 0, mu = 0;
	bool uniform = false;
	double low = 0, high = 0, delta = 0;
	float* nu_thresh = 0;
	double mu1 = 0, q1 = 0;
	double max_sigma = 0;

	if( !CV_IS_HIST(hist) || CV_IS_SPARSE_HIST(hist) || hist->mat.dims != 1 )
		CV_ERROR( CV_StsBadArg,
		"The histogram in Otsu method must be a valid dense 1D histogram" );

	count = hist->mat.dim[0].size;
	h = (float*)cvPtr1D( hist->bins, 0 );

	if( !CV_HIST_HAS_RANGES(hist) || CV_IS_UNIFORM_HIST(hist) )
	{
		if( CV_HIST_HAS_RANGES(hist) )
		{
			low = hist->thresh[0][0];
			high = hist->thresh[0][1];
		}
		else
		{
			low = 0;
			high = count;
		}

		delta = (high-low)/count;
		low += delta*0.5;
		uniform = true;
	}
	else
		nu_thresh = hist->thresh2[0];

	for( i = 0; i < count; i++ )
	{
		sum += h[i];
		if( uniform )
			mu += (i*delta + low)*h[i];
		else
			mu += (nu_thresh[i*2] + nu_thresh[i*2+1])*0.5*h[i];
	}

	sum = fabs(sum) > FLT_EPSILON ? 1./sum : 0;
	mu *= sum;

	mu1 = 0;
	q1 = 0;

	for( i = 0; i < count; i++ )
	{
		double p_i, q2, mu2, val_i, sigma;

		p_i = h[i]*sum;
		mu1 *= q1;
		q1 += p_i;
		q2 = 1. - q1;

		if( MIN(q1,q2) < FLT_EPSILON || MAX(q1,q2) > 1. - FLT_EPSILON )
			continue;

		if( uniform )
			val_i = i*delta + low;
		else
			val_i = (nu_thresh[i*2] + nu_thresh[i*2+1])*0.5;

		mu1 = (mu1 + val_i*p_i)/q1;
		mu2 = (mu - q1*mu1)/q2;
		sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
		if( sigma > max_sigma )
		{
			max_sigma = sigma;
			max_val = val_i;
		}
	}
	cvReleaseHist(&hist);
	__END__;
	return max_val;
}

CvMat *dx = NULL;
CvMat *dy = NULL;

PileCanny * seuilCanny( const IplImage* srcarr,IplImage* dstarr,
         int demiWindow,double k,double percent)
{
  int aperture_size= 5;
  double moy=0,vari=0;
  if (dx==NULL)
  {
    dx = cvCreateMat( srcarr->height, srcarr->width, CV_16SC1 );
    dy = cvCreateMat( srcarr->height, srcarr->width, CV_16SC1 );
  }
  if( dx->rows!=srcarr->height )
  {
    cvReleaseMat(&dx);
    cvReleaseMat(&dy);
    dx = cvCreateMat( srcarr->height, srcarr->width, CV_16SC1 );
    dy = cvCreateMat( srcarr->height, srcarr->width, CV_16SC1 );
  }
  int i, j;
  cvSobel( srcarr, dx, 1, 0, aperture_size );
  cvSobel( srcarr, dy, 0, 1, aperture_size );

  //----------------------------------------------------------------------------------------------------------------------------
  //------------------------------------Première passe pour estimer la valeur des params :--------------------------------------
  //----------------------------------------------------------------------------------------------------------------------------

  double maxi=0,nbMini=0;
  double mean=0;
  unsigned int cpt=0;
  //First find the maximum value:
  for( i = 0; i < srcarr->height; i++ )
  {
    const short* _dx = (short*)(dx->data.ptr + dx->step*i);
    const short* _dy = (short*)(dy->data.ptr + dy->step*i);
    int x, y;
    for( j = 0; j < srcarr->width; j++ )
    {
      int s = abs(_dx[j]) + abs(_dy[j]);
      if(maxi<s)
        maxi=s;
      cpt++;
    }
  }

  cpt=0;
  for( i = 0; i < srcarr->height; i++ )
  {
    const short* _dx = (short*)(dx->data.ptr + dx->step*i);
    const short* _dy = (short*)(dy->data.ptr + dy->step*i);
    int x, y;
    for( j = 0; j < srcarr->width; j++ )
    {
      int s = abs(_dx[j]) + abs(_dy[j]);
      double rescaledValue = (double)s/(double)maxi;
      CV_SetReal2D(gradImg,i,j,(uchar)(rescaledValue*255));
      cpt++;
      mean+=s;
    }
  }

  //----------------------------------------------------------------------------------------------------------------------------
  //--------------------------------------------Calcul des paramètres de canny :------------------------------------------------
  //----------------------------------------------------------------------------------------------------------------------------


  double seuil=findCannyTWithOtsu(gradImg);
  seuil=seuil*maxi/255;
  seuil=2*seuil;

  double high_thresh=seuil*k;//maxi*seuil*k;
  double low_thresh=high_thresh*percent;
  //	cout<<"ht : "<<high_thresh<<" lt : "<<low_thresh<<endl;
  
  //cout<<"maxi : "<<maxi<<" ; high_thresh : "<<high_thresh<<endl;

	PileCanny *p=new PileCanny;
	PILE_CANNY_INIT(MAX( 1 << 10, srcarr->width*srcarr->height/10 ));
	//if(maxi>600)
  {
		//----------------------------------------------------------------------------------------------------------------------------
		//--------------------------------------------Détection des contours avec canny :---------------------------------------------
		//----------------------------------------------------------------------------------------------------------------------------

		void *buffer = 0;
		uchar **stack_top, **stack_bottom = 0;

		CvMat srcstub, *src = (CvMat*)srcarr;
		CvMat dststub, *dst = (CvMat*)dstarr;
		CvSize size;
		int flags = aperture_size;
		int low, high;
		int* mag_buf[3];
		uchar* map;
		int mapstep, maxsize;
		CvMat mag_row;

		src = cvGetMat( src, &srcstub );
		dst = cvGetMat( dst, &dststub );
	
		if( low_thresh > high_thresh )
		{
			double t;
			CV_SWAP( low_thresh, high_thresh, t );
		}

		aperture_size &= INT_MAX;
		if( (aperture_size & 1) == 0 || aperture_size < 3 || aperture_size > 7 )
			return 0;

		size = cvGetMatSize( src );


		if( flags & CV_CANNY_L2_GRADIENT )
		{
			Cv32suf ul, uh;
			ul.f = (float)low_thresh;
			uh.f = (float)high_thresh;

			low = ul.i;
			high = uh.i;
		}
		else
		{
			low = cvFloor( low_thresh );
			high = cvFloor( high_thresh );
		}

		buffer = cvAlloc( (size.width+2)*(size.height+2) + (size.width+2)*3*sizeof(int));

		mag_buf[0] = (int*)buffer;
		mag_buf[1] = mag_buf[0] + size.width + 2;
		mag_buf[2] = mag_buf[1] + size.width + 2;
		map = (uchar*)(mag_buf[2] + size.width + 2);
		mapstep = size.width + 2;

		maxsize = MAX( 1 << 10, size.width*size.height/10 );
		stack_top = stack_bottom = (uchar**)cvAlloc( maxsize*sizeof(stack_top[0]));

		memset( mag_buf[0], 0, (size.width+2)*sizeof(int) );
		memset( map, 1, mapstep );
		memset( map + mapstep*(size.height + 1), 1, mapstep );

		 //sector numbers 
		 //  (Top-Left Origin)
	//
	//       1   2   3
	//         *  *  * 
	//          * * *  
	//        0*******0
	//          * * *  
	//         *  *  * 
	//        3   2   1
    

		#define CANNY_PUSH(d)    *(d) = (uchar)2, *stack_top++ = (d)
		#define CANNY_POP(d)     (d) = *--stack_top

		mag_row = cvMat( 1, size.width, CV_32F );
			short* _dx,*_dy;

		// calculate magnitude and angle of gradient, perform non-maxima supression.
		// fill the map with one of the following values:
		//   0 - the pixel might belong to an edge
		//   1 - the pixel can not belong to an edge
		//   2 - the pixel does belong to an edge
		for( i = 0; i <= size.height; i++ )
		{
			int* _mag = mag_buf[(i > 0) + 1] + 1;
			float* _magf = (float*)_mag;
			_dx = (short*)(dx->data.ptr + dx->step*i);
			_dy = (short*)(dy->data.ptr + dy->step*i);
			uchar* _map;
			int x, y;
			int magstep1, magstep2;
			int prev_flag = 0;

			if( i < size.height )
			{
				_mag[-1] = _mag[size.width] = 0;

				if( !(flags & CV_CANNY_L2_GRADIENT) )
					for( j = 0; j < size.width; j++ )
						_mag[j] = abs(_dx[j]) + abs(_dy[j]);
            
				else
				{
					for( j = 0; j < size.width; j++ )
					{
						x = _dx[j]; y = _dy[j];
						_magf[j] = (float)std::sqrt((double)x*x + (double)y*y);
					}
				}
			}
			else
				memset( _mag-1, 0, (size.width + 2)*sizeof(int) );

			// at the very beginning we do not have a complete ring
			// buffer of 3 magnitude rows for non-maxima suppression
			if( i == 0 )
				continue;

			_map = map + mapstep*i + 1;
			_map[-1] = _map[size.width] = 1;
        
			_mag = mag_buf[1] + 1; // take the central row
			_dx = (short*)(dx->data.ptr + dx->step*(i-1));
			_dy = (short*)(dy->data.ptr + dy->step*(i-1));
    
			magstep1 = (int)(mag_buf[2] - mag_buf[1]);
			magstep2 = (int)(mag_buf[0] - mag_buf[1]);

			if( (stack_top - stack_bottom) + size.width > maxsize )
			{
				uchar** new_stack_bottom;
				maxsize = MAX( maxsize * 3/2, maxsize + size.width );
				new_stack_bottom = (uchar**)cvAlloc( maxsize * sizeof(stack_top[0]));
				memcpy( new_stack_bottom, stack_bottom, (stack_top - stack_bottom)*sizeof(stack_top[0]) );
				stack_top = new_stack_bottom + (stack_top - stack_bottom);
				cvFree( &stack_bottom );
				stack_bottom = new_stack_bottom;
			}

			for( j = 0; j < size.width; j++ )
			{
				#define CANNY_SHIFT 15
				#define TG22  (int)(0.4142135623730950488016887242097*(1<<CANNY_SHIFT) + 0.5)

				x = _dx[j];
				y = _dy[j];
				int s = x ^ y;
				int m = _mag[j];

				x = abs(x);
				y = abs(y);
				if( m > low )
				{
					int tg22x = x * TG22;
					int tg67x = tg22x + ((x + x) << CANNY_SHIFT);

					y <<= CANNY_SHIFT;

					if( y < tg22x )
					{
						if( m > _mag[j-1] && m >= _mag[j+1] )
						{
							if( m > high && !prev_flag && _map[j-mapstep] != 2 )
							{
								CANNY_PUSH( _map + j );
								prev_flag = 1;
							}
							else
								_map[j] = (uchar)0;
							continue;
						}
					}
					else if( y > tg67x )
					{
						if( m > _mag[j+magstep2] && m >= _mag[j+magstep1] )
						{
							if( m > high && !prev_flag && _map[j-mapstep] != 2 )
							{
								CANNY_PUSH( _map + j );
								prev_flag = 1;
							}
							else
								_map[j] = (uchar)0;
							continue;
						}
					}
					else
					{
						s = s < 0 ? -1 : 1;
						if( m > _mag[j+magstep2-s] && m > _mag[j+magstep1+s] )
						{
							if( m > high && !prev_flag && _map[j-mapstep] != 2 )
							{
								CANNY_PUSH( _map + j );
								prev_flag = 1;
							}
							else
								_map[j] = (uchar)0;
							continue;
						}
					}
				}
				prev_flag = 0;
				_map[j] = (uchar)1;
			}

			// scroll the ring buffer
			_mag = mag_buf[0];
			mag_buf[0] = mag_buf[1];
			mag_buf[1] = mag_buf[2];
			mag_buf[2] = _mag;
		}

		// now track the edges (hysteresis thresholding)
		while( stack_top > stack_bottom )
		{
			uchar* m;
			if( (stack_top - stack_bottom) + 8 > maxsize )
			{
				uchar** new_stack_bottom;
				maxsize = MAX( maxsize * 3/2, maxsize + 8 );
				new_stack_bottom = (uchar**)cvAlloc( maxsize * sizeof(stack_top[0]));
				memcpy( new_stack_bottom, stack_bottom, (stack_top - stack_bottom)*sizeof(stack_top[0]) );
				stack_top = new_stack_bottom + (stack_top - stack_bottom);
				cvFree( &stack_bottom );
				stack_bottom = new_stack_bottom;
			}

			CANNY_POP(m);
    
			if( !m[-1] )
				CANNY_PUSH( m - 1 );
			if( !m[1] )
				CANNY_PUSH( m + 1 );
			if( !m[-mapstep-1] )
				CANNY_PUSH( m - mapstep - 1 );
			if( !m[-mapstep] )
				CANNY_PUSH( m - mapstep );
			if( !m[-mapstep+1] )
				CANNY_PUSH( m - mapstep + 1 );
			if( !m[mapstep-1] )
				CANNY_PUSH( m + mapstep - 1 );
			if( !m[mapstep] )
				CANNY_PUSH( m + mapstep );
			if( !m[mapstep+1] )
				CANNY_PUSH( m + mapstep + 1 );
		}

		//PILE_CANNY_INIT(500);

		//    src : (CvMat*)srcarr;
		//    dst : (CvMat*)dstarr;
	
		// the final pass, form the final image

		uchar* _dst,*_src;
		for( i = demiWindow; i < size.height-demiWindow-1; i++ )
		{
			const uchar* _map = map + mapstep*(i+1) + 1;
			_dst = dst->data.ptr + dst->step*i;
			//_dst = (uchar*)dstarr->imageData + dstarr->widthStep*i;
			_src = src->data.ptr + src->step*i;

	//        _dx = (short*)(dx->data.ptr + dx->step*(i));
	//        _dy = (short*)(dy->data.ptr + dy->step*(i));

			//pc->shouldResize(size.width);
			PILE_CANNY_RESIZE(size.width);
		
        
			for( j = demiWindow; j < size.width-demiWindow-1; j++ ){
				if((_map[j] >> 1)==0)
					_dst[j]=0;
				else{
          uchar *position = (uchar*)(&_src[j] - src->data.ptr);
					PILE_CANNY_PUSH(position);
					_dst[j]=254;
				}
				//PILE_CANNY_PUSH(&_src[j]);
			}
			//_dst[j] = (uchar)-(_map[j] >> 1);
		}
		cvFree( &buffer );
		cvFree( &stack_bottom );
	}
   // cvReleaseMat( &dx );
//	cvReleaseMat( &dy );
	return p;
	//return pc;
}

void testCannyValues(PileCanny* p, IplImage* canny, const IplImage* srcarr, bool twoThreshold, float thresh1, float thresh2 )
{
  IplImage *testImg = cvCreateImage( cvSize(srcarr->width, srcarr->height), 8, 1);
  if (dx==NULL)
  {
    dx = cvCreateMat( srcarr->height, srcarr->width, CV_16SC1 );
    dy = cvCreateMat( srcarr->height, srcarr->width, CV_16SC1 );
  }
  if( dx->rows!=srcarr->height )
  {
    cvReleaseMat(&dx);
    cvReleaseMat(&dy);
    dx = cvCreateMat( srcarr->height, srcarr->width, CV_16SC1 );
    dy = cvCreateMat( srcarr->height, srcarr->width, CV_16SC1 );
  }
  int i, j;
  cvSobel( srcarr, dx, 1, 0, 5 );
  cvSobel( srcarr, dy, 0, 1, 5);

  double maxi=0,nbMini=0;
  //double moy=0,vari=0;
  unsigned int cpt=0;

  //First find the maximum value:
  for( i = 0; i < srcarr->height; i++ )
  {
    const short* _dx = (short*)(dx->data.ptr + dx->step*i);
    const short* _dy = (short*)(dy->data.ptr + dy->step*i);
    int x, y;
    for( j = 0; j < srcarr->width; j++ )
    {
      int s = abs(_dx[j]) + abs(_dy[j]);
      if(maxi<s)
        maxi=s;
      //moy+=s;
      cpt++;
    }
  }

  cvSet( testImg, cvScalar(0) );

  //Now scan image

  uchar *addrPixel;
  PILE_CANNY_RELOAD();

  PILE_CANNY_POP(addrPixel);
  while(!PILE_CANNY_IS_END()){
    int indexPixel=(int)(addrPixel);

    int xLocal=indexPixel%srcarr->widthStep;
    int yLocal=(int)(indexPixel/srcarr->widthStep);

    short _dx = ((short*)(dx->data.ptr + dx->step*yLocal))[xLocal];
    short _dy = ((short*)(dy->data.ptr + dy->step*yLocal))[xLocal];

    double s = ((double)(abs(_dx) + abs(_dy))) / maxi;
    s = s*254.0+1;
    CV_SetReal2D(testImg, yLocal, xLocal, (uchar)s );

    PILE_CANNY_POP(addrPixel);
  }

  PILE_CANNY_DELETE();

  PILE_CANNY_INIT(MAX( 1 << 10, testImg->width*testImg->height/10 ));

  ConnectedComponents cc(300);
  CvMat *labels;
  labels = cvCreateMat(testImg->height,testImg->width,CV_32SC1);
  cc.param1 = thresh1;
  cc.param2 = thresh2;
  cc.label_image_diff( testImg, labels, 0 );

  cc.computeMeansCC(testImg, labels, 0);
  
  //cvSaveImage( "cannyTest.png", testImg );

  CvMat *src = (CvMat*)srcarr;
  CvSize size;
  size = cvGetSize( src );
  for( i = 2; i < size.height-3; i++ )
  {
    PILE_CANNY_RESIZE(size.width);

    unsigned char* ligneImg = (unsigned char*)(testImg->imageData + testImg->widthStep*i);
    unsigned char* ligneCanny = (unsigned char*)(canny->imageData + canny->widthStep*i);
    unsigned char* srcImg = (unsigned char*)(srcarr->imageData + srcarr->widthStep*i);
    for( j = 2; j < size.width-3; j++ ){
      if(ligneImg[j]!=0)
      {
        uchar *position = (uchar *)(srcarr->widthStep*i+j);
        PILE_CANNY_PUSH(position);
        if( twoThreshold )
          ligneCanny[j] = ligneImg[j];
        else
          ligneCanny[j] = 255;
      }
      else
      {
        ligneCanny[j] = 0;
      }
    }
    //_dst[j] = (uchar)-(_map[j] >> 1);
  }

  cvReleaseImage( &testImg );
  cvReleaseMat(&labels);
}

IplImage* computeImg(IplImage *frame,double k,int window){
  if(frame==NULL){
    return NULL;
  }
  //
  //int taille=3+(frame->width+frame->height)/500;//estime la taille de la fenetre d'analyse à partir de la taille de l'image
  //if(taille<5)taille=5;
  int voisinage=2;
  if(frame->width+frame->height>10000) voisinage++;
  if(frame->width+frame->height>5000) voisinage++;
  if(frame->width+frame->height>1000) voisinage++;

  IplImage *dest;

  double bas=0,haut=0;

  OpImg tests;
  ConnectedComponents cc(500);

  IplImage *finalImg;
  //utilise le modèle pour détecter les classes autour des contours :
  //dest=tests.expandEdge(frame,canny,5,2);
  //dest=tests.expandEdge(frame,pc,canny,voisinage+1);
  //finalImg=tests.SeuilSauvola(frame,10,0.2);
  //finalImg=tests.SeuilSauvolaOptimiz(frame,window,k);
  finalImg=Threshold_Otsu(frame);

  //"colorie" l'interieur des lettres...
  //finalImg = cc.connected(dest,labelimg);
  //tests.binarize(dest);

  //cout<<"filterCC..."<<endl;

  //filtrage final pour enlever les composantes connexes jugées comme étant du bruit :
  //cc.filterCC(finalImg,labelimg);


  //cvShowImage("scanned",frame);
  //cvShowImage("binSauvola",binSauvola);

  //cvReleaseMat(&labelimg);
  //cvReleaseImage(&imgBis);
  //cvReleaseImage(&canny);
  //cvReleaseImage(&dest);
  //cvReleaseImage(&frame);
  return finalImg;
}

void binarizeSauvola( IplImage *frame )
{
  OpImg tests;
  IplImage *finalImg;
  finalImg=tests.SeuilSauvola(frame,5,0.08);
  cvCopy(finalImg, frame);

  cvReleaseImage( &finalImg );
}

void binarizeOtsu( IplImage *frame )
{
  OpImg tests;
  ConnectedComponents cc(500);

  IplImage *finalImg;
  finalImg=Threshold_Otsu(frame);
  cvCopy(finalImg, frame);

  cvReleaseImage( &finalImg );
}

cv::Mat binarizePerso(IplImage *grey, float threshold1, float threshold2, bool postFilter, bool rescale)
{
  PileCanny *pc;
  OpImg tests;
  ConnectedComponents cc(500);

  double ratio=1.0;
  if( rescale )
    ratio = 2;
  if( greyLevels==NULL || greyLevels->width!=grey->width*ratio
     || greyLevels->height!=grey->height*ratio)
    initGlobalVars(grey->width*ratio,grey->height*ratio);

  int nbPic=0;
  bool stay=true;
  bool save=false;
  int key;

  if( ratio!=1.0 )
  {
    cvResize(grey,greyLevels,CV_INTER_LINEAR );//CV_INTER_CUBIC, CV_INTER_AREA, CV_INTER_NN
  }
  else
    cvCopy( grey, greyLevels );
  cv::Mat copyOfImg(greyLevels), smoothImg;
  cvSmooth(greyLevels,greyLevels,CV_GAUSSIAN, 3, 3);
  
  pc=seuilCanny(greyLevels,canny,2, 0.6, 0.5);//determine le seuil et trouve les contours seuil bas (0.63)
  double meanValue = cv::mean(Mat(canny))[0];
  float param1 = 0.6;
  while ( meanValue>16 )
  {
    PILE_CANNY_DELETE_PTR(pc);
    param1+=0.1;
    pc=seuilCanny(greyLevels,canny,2, param1, 0.5);//determine le seuil et trouve les contours seuil bas (0.63)
    double meanValue1 = cv::mean(Mat(canny))[0];
    double diff = meanValue-meanValue1;
    if( diff<5 )//no significant changes...
      meanValue = 0;
    else
      meanValue = meanValue1;
  }
#ifdef _DEBUG
  cvShowImage("cannyBefore", canny);
  cvSaveImage("cannyBefore.png", canny);
#endif // _DEBUG
  testCannyValues( pc, canny, greyLevels, postFilter, threshold1, threshold2 );
  Mat copyOfCanny(canny);
#ifdef _DEBUG
  cvShowImage("canny", canny);
  cvSaveImage("cannyAfter.png", canny);
  waitKey(25);
#endif // _DEBUG

  IplImage *dest=tests.expandEdgeEM(greyLevels,pc,canny,3);

#ifdef _DEBUG
  imshow("debug_2", (Mat)dest);
#endif // _DEBUG

  IplImage *finalImgScaled = cc.connected(dest,OpImg::labelimg,greyLevels, postFilter);

  if( ratio!=1.0 )
  {
#ifdef _DEBUG
    cvSaveImage("finalBefore.png", finalImgScaled);
#endif // _DEBUG
    cvSmooth( finalImgScaled, finalImgScaled, CV_BLUR, 5 );
    cvResize( finalImgScaled, grey, CV_INTER_CUBIC );//CV_INTER_CUBIC, CV_INTER_AREA, CV_INTER_NN

    //64 if ratio==2, 128 if ratio==1
    cvThreshold( grey, grey, 146, 255, CV_THRESH_BINARY );//150
  }
  else
  {
    cvCopy( finalImgScaled, grey );
  }
  //cc.filterSmallDots( finalImg, 7 );
  
  cvReleaseImage( &finalImgScaled );
  cvReleaseImage( &dest );

  return copyOfCanny.clone();
}


IplImage * Threshold_Otsu( IplImage * image )
{

  IplImage *out= cvCreateImage(cvSize(image->width,image->height),8,1);
  //cout<<"Attention, penser  a diminuer de 30% le temps de calcul (optimisation non prise en compte)"<<endl;

  int hist_size[] = {255};
  CvHistogram* hist;

  hist = cvCreateHist( 1, hist_size, CV_HIST_ARRAY);
  cvCalcHist( &image, hist, 0, 0 );
  double max_val = 0;

  int i, count;
  const float* h;
  double sum = 0, mu = 0;
  bool uniform = false;
  double low = 0, high = 0, delta = 0;
  float* nu_thresh = 0;
  double mu1 = 0, q1 = 0;
  double max_sigma = 0;

  count = hist->mat.dim[0].size;
  h = (float*)cvPtr1D( hist->bins, 0 );

  if( !CV_HIST_HAS_RANGES(hist) || CV_IS_UNIFORM_HIST(hist) )
  {
    if( CV_HIST_HAS_RANGES(hist) )
    {
      low = hist->thresh[0][0];
      high = hist->thresh[0][1];
    }
    else
    {
      low = 0;
      high = count;
    }

    delta = (high-low)/count;
    low += delta*0.5;
    uniform = true;
  }
  else
    nu_thresh = hist->thresh2[0];

  for( i = 0; i < count; i++ )
  {
    sum += h[i];
    if( uniform )
      mu += (i*delta + low)*h[i];
    else
      mu += (nu_thresh[i*2] + nu_thresh[i*2+1])*0.5*h[i];
  }

  sum = fabs(sum) > FLT_EPSILON ? 1./sum : 0;
  mu *= sum;

  mu1 = 0;
  q1 = 0;

  for( i = 0; i < count; i++ )
  {
    double p_i, q2, mu2, val_i, sigma;

    p_i = h[i]*sum;
    mu1 *= q1;
    q1 += p_i;
    q2 = 1. - q1;

    if( MIN(q1,q2) < FLT_EPSILON || MAX(q1,q2) > 1. - FLT_EPSILON )
      continue;

    if( uniform )
      val_i = i*delta + low;
    else
      val_i = (nu_thresh[i*2] + nu_thresh[i*2+1])*0.5;

    mu1 = (mu1 + val_i*p_i)/q1;
    mu2 = (mu - q1*mu1)/q2;
    sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
    if( sigma > max_sigma )
    {
      max_sigma = sigma;
      max_val = val_i;
    }
  }

  unsigned char* ligneImg;
  unsigned char* ligneEdge;
  unsigned char* ligneOut;
  double echange;
  double pixel;

  //first pass : only on edges :
  for(int i=0;i<image->height;i++){
    ligneImg = (unsigned char*)(image->imageData+image->widthStep*i);
    ligneOut = (unsigned char*)(out->imageData+out->widthStep*i);
    for(int j=0;j<image->width;j++){
      if(ligneImg[j]>max_val)
        ligneOut[j]=254;
      else
        ligneOut[j]=0;
    }
  }

  cvReleaseHist(&hist);
  return out;
}
