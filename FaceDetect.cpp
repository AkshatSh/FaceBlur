// Akshat Shrivastava

#include "opencv2/objdetect.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"

#include "opencv2/videoio/videoio_c.h"
#include "opencv2/highgui/highgui_c.h"

#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <math.h>

using namespace std;
using namespace cv;

String face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
string window_name = "Blur - Face detection";
RNG rng(12345);

void faceDetect(Mat frame);

int main( int argc, char** argv )
{
    VideoCapture cap("TestVid.mov");


    VideoWriter writer("output3.mov", 
               cap.get(CV_CAP_PROP_FOURCC),
               cap.get(CV_CAP_PROP_FPS),
               cv::Size(cap.get(CV_CAP_PROP_FRAME_WIDTH),
               cap.get(CV_CAP_PROP_FRAME_HEIGHT)));;

    if (!writer.isOpened()){
      printf("--(!)Error Opening Writer\n");
      return -1; 
    }

    if( !face_cascade.load( face_cascade_name ) ){ 
    	printf("--(!)Error loading\n"); 
    	return -1; 
    }

    while( cap.isOpened() )
    {
        Mat frame;
        Mat newFrame;
        
        if ( ! cap.read(frame) ) {
            break;
        }

        if (!frame.empty()){
        	faceDetect(frame);
          writer.write(frame);
          imshow(window_name, frame);
        }

        int k = waitKey(10);

        if ( k==27 ) {
            break;
        }
    }
    cap.release();
    writer.release();
    return 0;
}

// Detect the face present in the frame and blur it out with a GuassianBlur with Kernel size of 369 
void faceDetect( Mat frame )
{
    // Create a vector to hold all the frames with faces
    std::vector<Rect> faces;
    Mat frame_gray;
    cvtColor( frame, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    // Detect the faces in the faces vector
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

    // Itereate through the face
    for( size_t i = 0; i < faces.size(); i++ )
    {
        Point corner(faces[i].x, faces[i].y);
        Point oppCorner(faces[i].x + faces[i].width, faces[i].y + faces[i].height);

        Mat faceROI = frame( faces[i] );

        Rect faceLocation = Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height);

        Mat newFace;
        GaussianBlur( faceROI, newFace, Size( 369, 369 ), 0, 0 );
        newFace.copyTo(frame(faces[i]));
    }
 }

