#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cvdef.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <math.h>


using namespace cv;
using namespace std;

Mat image;
Mat output, output1;

void harrisCornerDetector() {

    // Loading the actual image
    image = imread("C:\\Users\\Orain\\Downloads\\first_200_right\\first_200_right\\000000.png", IMREAD_COLOR);
    //image = imread("C:\\Users\\orain\\Downloads\\first_200_right\\first_200_right\\000000.png", IMREAD_COLOR); //IMREAD_GRAYSCALE for grey scale IMREAD_COLOR for color
    VideoWriter video = VideoWriter("TestingKNN.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30., image.size());

    // Edge cases
    if (image.empty()) {
        cout << "Error loading image" << endl;
        return;
    }

    Ptr<SIFT> detector = cv::SIFT::create(500, 3, 0.09, 10, 1.6);
    /*
    nFeatures	        The number of best features to retain.
    nOctaveLayers	    The number of layers in each octave.
    contrastThreshold	The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions.
                        The larger the threshold, the less features are produced by the detector.
    edgeThreshold	    The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold,
                        the less features are filtered out (more features are retained).
    sigma	            The sigma of the Gaussian applied to the input image at the octave #0.
    */
    std::vector<cv::KeyPoint> keypoints;
    Mat descriptors;
    detector->detectAndCompute(image, Mat(), keypoints, descriptors);
    cv::drawKeypoints(image, keypoints, output, Scalar(0, 255, 0));

    //cv::imshow("First Image", output);
    //cv::waitKey();
    //video.write(output);
    for (int j = 1; j <= 200; j++)
    {
        // read next image and compute features 
        String filename = format("%06d.png", j);
        Mat image1 = imread("C:\\Users\\orain\\Downloads\\first_200_right\\first_200_right\\" + filename);

        vector<KeyPoint> keypoints1;
        Mat descriptors1;
        detector->detectAndCompute(image1, Mat(), keypoints1, descriptors1);
        cv::drawKeypoints(image1, keypoints1, output1, Scalar(0, 255, 0));

        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descriptors, descriptors1, knn_matches, 2);


        // Filter matches based on distance ratio test
        std::vector<cv::DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i][0].distance < 0.65 * knn_matches[i][1].distance)   
            // 0.65 doesnt control how long the line is it changes whats the next nearest 
            // neighbor and then sees if the original neighbor is the best neighbor.
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }
        for (size_t i = 0; i < good_matches.size(); i++) 
        {
            cv::KeyPoint kp1 = keypoints[good_matches[i].queryIdx];
            cv::KeyPoint kp2 = keypoints1[good_matches[i].trainIdx];
            cv::line(output1, kp1.pt, kp2.pt, cv::Scalar(0, 255, 0));
        }

        //imshow("Matches", output1);
        //cv::waitKey();
        //Needed for the next loop 
        image = image1;
        keypoints = keypoints1;
        descriptors = descriptors1;
        video.write(output1);
    }

}

int main()
{

    harrisCornerDetector();

    return 0;
}
/*
*   // Add results to image and save.
    cv::Mat output;
    cv::drawKeypoints(input, keypoints, output);
    cv::imwrite("sift_result.jpg", output);
    // Converting the color image into grayscale
    cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    // Detecting corners using the cornerHarris built in function
    output = Mat::zeros(image.size(), CV_32FC1);
    cornerHarris(gray, output,
        2,              // blockSize
        3,              // Aperture parameter for the Sobel operator
        0.04);          // Harris detector free parameter
    // Normalizing - Convert corner values to integer so they can be drawn
    normalize(output, output_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat()); //output_norm holds the points

    // Drawing a circle around corners
    for (int j = 0; j < output_norm.rows; j++) {
        for (int i = 0; i < output_norm.cols; i++) {
            if ((int)output_norm.at<float>(j, i) > 128) {
                circle(image, Point(i, j), 4, Scalar(0, 255, 0), 2, 8, 0); //points i and j are where the corner is.

            }
        }
    }
    //cv::imshow("Output Harris", image);
    //cv::waitKey();
    video.write(image);

    for (int i = 1; i <= 2; i++)
    {
        // read next image and compute features
        String filename = format("%06d.png", i);
        Mat image1 = imread("C:\\Users\\orain\\Downloads\\first_200_right\\first_200_right\\" + filename);
        cvtColor(image1, gray, cv::COLOR_BGR2GRAY);
        // Remake every loop so that its empty
        Mat  output_norm1, output1;

        cornerHarris(gray, output1, 2, 3, 0.04);
        normalize(output1, output_norm1, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
        // Drawing a circle around corners
        for (int j = 0; j < output_norm1.rows; j++) {
            for (int i = 0; i < output_norm1.cols; i++) {
                if ((int)output_norm1.at<float>(j, i) > 128) {
                    circle(image1, Point(i, j), 2, Scalar(0, 255, 0), 1, 8, 0);
                }
            }
        }

        vector<DMatch> matches;
        //matcher.match(output_norm, output_norm1, matches, Mat());
        //maybe put matching/knn here btw output(priorImage) and output1(nextImage)
        cv::imshow("Test loop Out", image1);
        cv::waitKey();
        //video.write(image);

    }
*/
