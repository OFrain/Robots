
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat image, gray;
Mat output, output_norm, output_norm_scaled;

void harrisCornerDetector() {
    //Opens orginal image
    //cv::imshow("Original image", image);

    // Converting the color image into grayscale
    cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Detecting corners using the cornerHarris built in function
    output = Mat::zeros(image.size(), CV_32FC1);
    cornerHarris(gray, output,
        3,              // Neighborhood size
        3,              // Aperture parameter for the Sobel operator
        0.04);          // Harris detector free parameter

    // Normalizing - Convert corner values to integer so they can be drawn
    normalize(output, output_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(output_norm, output_norm_scaled);

    // Drawing a circle around corners
    for (int j = 0; j < output_norm.rows; j++) {
        for (int i = 0; i < output_norm.cols; i++) {
            if ((int)output_norm.at<float>(j, i) > 128) {
                circle(image, Point(i, j), 4, Scalar(0, 255, 0), 2, 8, 0); //points i and j are where the croner is. 
            }
        }
    }
    // Displaying the result
    cv::imshow("Output Harris", image);
    cv::waitKey();
}


int main()
{
    // Loading the actual image
    //image = imread("C:\\Users\\Orain\\Downloads\\first_200_right\\first_200_right\\000000.png", IMREAD_COLOR);
    image = imread("C:\\Users\\orain\\Downloads\\first_200_right\\first_200_right\\000000.png", IMREAD_COLOR);

    // Edge cases
    if (image.empty()) {
        cout << "Error loading image" << endl;
    }
    harrisCornerDetector();

    return 0;
}
/*


Mat src, src_gray, dst; //src is the colored image src_gray is just a B/W image with the corners 
int thresh = 128; //Intensity of the corners detections
int max_thresh = 255;
const char* source_window = "Source image";
const char* corners_window = "Corners detected";

void cornerHarris_demo(int, void*);

int main(int argc, char** argv)
{
    src = imread("C:\\Users\\orain\\Downloads\\first_200_right\\first_200_right\\000000.png");
    if (src.empty())
    {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
    cvtColor(src, src_gray, COLOR_BGR2GRAY); //Converts the arg 1 by a filter arg 3 and saves it in arg 2
    namedWindow(source_window);              //Opens new window
    //createTrackbar("Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo); //creates a slider to adjust the thresh value
    imshow(source_window, src);             //Fills arg 1 window with arg 2 image                    
    cornerHarris_demo(0, 0);
    waitKey();
    return 0;
}
void cornerHarris_demo(int, void*)
{
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    Mat dst = Mat::zeros(src.size(), CV_32FC1);
    cornerHarris(src_gray, dst, blockSize, apertureSize, k);
    Mat dst_norm, dst_norm_scaled;
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);
    for (int i = 0; i < dst_norm.rows; i++)
    {
        for (int j = 0; j < dst_norm.cols; j++)
        {
            if ((int)dst_norm.at<float>(i, j) > thresh)
            {
                circle(dst_norm_scaled, Point(j, i), 5, Scalar(0), 2, 8, 0);
            }
        }
    }
    namedWindow(corners_window);
    imshow(corners_window, dst_norm_scaled);
}*/