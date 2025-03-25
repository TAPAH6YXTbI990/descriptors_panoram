#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    string image_path1 = "C:/Users/User/.vscode/projects/panoram1.png";
    string image_path2 = "C:/Users/User/.vscode/projects/panoram2.png";
   
    Mat img1 = imread(image_path1, IMREAD_COLOR);
    Mat img2 = imread(image_path2, IMREAD_COLOR);

    if (img1.empty() || img2.empty())
    {
        cerr << "Error while uploading image!" << endl;
        return -1; 
    }

    int width = 800;
    int height = 580;

    resize(img1, img1, Size(width, height), INTER_LINEAR);
    resize(img2, img2, Size(width, height), INTER_LINEAR);

    imshow("img1", img1);
    imshow("img2", img2);

    Ptr<ORB> orb = ORB::create();
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    orb->detectAndCompute(img1, Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, Mat(), keypoints2, descriptors2); 

    BFMatcher matcher(NORM_HAMMING);
    vector<vector<DMatch>> matches;
    matcher.knnMatch(descriptors1, descriptors2, matches, 2);

    const float ratio_thresh = 0.75f;
    vector<DMatch> good_matches;
    for (size_t i = 0; i < matches.size(); i++)
    {
        if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)
        {
            good_matches.push_back(matches[i][0]);
        }
    }

    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches,
        Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("Matches", img_matches);

    vector<cv::Point2f> points1, points2;
    for (int i = 0; i < good_matches.size(); i++)
    {
        points1.push_back(keypoints1[good_matches[i].queryIdx].pt);
        points2.push_back(keypoints2[good_matches[i].trainIdx].pt);
    }

    Mat H = findHomography(points2, points1, RANSAC);

    Mat result;
    warpPerspective(img2, result, H, Size(img1.cols + img2.cols, img1.rows));
    img1.copyTo(result(Rect(0, 0, img1.cols, img1.rows)));

    imshow("Result", result);

    waitKey(0);

    return 0;
}