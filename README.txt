Project: 
	Homework 1 UGA Robotics ELEE-4280

Project Description: 
	This project takes in 200 files of single frames from a video, conducts edge/corner tracking on those them to then and then draws a line between them representing how these points move as the frames progress. 
	The end project is a video showing how the point move between frames. 

 	First it computes and detects all the key points and descriptors of the first image.It then does this again for the next image storing the values in a different set of variables. 
	K-Nearest Neighbor is then taken on these data sets using brute force. How it is set up is using the first images points it will find the nearest point in the second images points. 
	Once two nearest neighbors are found between the images the they are store in a new matches variable. A filter is run to compare the first KNN match to the next closest 
	match to see if it is truly the best match. This is done by comparing distances between the two to see if which set of points has the smallest distance between the two. 
	From here it takes the good matches and draws a line between them and writes it to a video. 

Needed Parameters:78
	OpenCV library
	C++ compiler 
	200 single frames files
	C++ code editor

Dev Environment:
	Visual Studio 2022 

Running: 
	Because I'm in Visual Studio there is no need for a make file, I installed a compiler and then use this to run my code. To fully compile and run the code with all 200 frame files it takes about 
	2-3 minutes to make the video.

