
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <vector>
#include <chrono>
#include <thread>

using namespace cv;
using namespace std;

Mat image, gray, gray1;
Mat output, output_norm, output1;

void harrisCornerDetector() {

    // Loading the actual image
    //image = imread("C:\\Users\\Orain\\Downloads\\first_200_right\\first_200_right\\000000.png", IMREAD_COLOR);
    image = imread("C:\\Users\\orain\\Downloads\\first_200_right\\first_200_right\\000000.png", IMREAD_COLOR);

    VideoWriter video = VideoWriter("robotVideo.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30., image.size());

    // Edge cases
    if (image.empty()) {
        cout << "Error loading image" << endl;
    }

    // Converting the color image into grayscale
    cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Detecting corners using the cornerHarris built in function
    output = Mat::zeros(image.size(), CV_32FC1);
    cornerHarris(gray, output,
        2,              // blockSize
        3,              // Aperture parameter for the Sobel operator
        0.04);          // Harris detector free parameter

    // Normalizing - Convert corner values to integer so they can be drawn
    normalize(output, output_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

    // Drawing a circle around corners
    for (int j = 0; j < output_norm.rows; j++) {
        for (int i = 0; i < output_norm.cols; i++) {
            if ((int)output_norm.at<float>(j, i) > 128) {
                circle(image, Point(i, j), 4, Scalar(0, 255, 0), 2, 8, 0); //points i and j are where the corner is.  
                //add points to vector or Mat
            }
        }
    }
    cv::imshow("Output Harris", image);
    cv::waitKey();
    video.write(image);

    for (int i = 1; i <= 200; i++)
    {
        // read next image and compute features
        String filename = format("%06d.png", i); 
        Mat img1 = imread("C:\\Users\\orain\\Downloads\\first_200_right\\first_200_right\\" + filename);
        cvtColor(img1, gray, cv::COLOR_BGR2GRAY);
        Mat  output_norm1; // Rematch every loop so that its empty
        cornerHarris(gray, output1, 2, 3, 0.04);
        normalize(output1, output_norm1, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
        // Drawing a circle around corners
        for (int j = 0; j < output_norm1.rows; j++) {
            for (int i = 0; i < output_norm1.cols; i++) {
                if ((int)output_norm1.at<float>(j, i) > 128) {
                    circle(img1, Point(i, j), 4, Scalar(0, 255, 0), 2, 8, 0); //points i and j are where the corner is.  
                    //add point to array
                }
            }
        }
        //maybe put matching/knn here btw output(priorImage) and output1(nextImage)
        //draw lines from old points

        output = output1; //needed for next loop 
        //cv::imshow("Test loop Out", img1);
        //cv::waitKey();
        video.write(img1);

    }
}

int main()
{
    harrisCornerDetector();

    return 0;
}
