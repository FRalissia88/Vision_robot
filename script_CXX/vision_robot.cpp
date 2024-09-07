/*************************
 * Ce script calcule les paramètres d'étalonnage d'une caméra à partir des photos de dammiers
 * disponibles dans le répertoire "repertoire_image/*.jpg"
 * 
 * Ensuite, il calcule la matrice permettant la mise à plat du damier "vue_au_sol.jpg" (warping).
 * 
 * Les différentes variables nécessaires au warping (remise à plat pour la vue oiseau) sont disponibles dans le fichier 
 * "parametres_calibration_robot.py".
*************************/

// PARAMETRES DAMIER (nombre de coins intérieurs verticaux et horizontaux)
//#define COINS_X 8
//#define COINS_Y 6

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

int main() {
    const int COINS_X = 8;
    const int COINS_Y = 6;
    Size patternSize(COINS_X, COINS_Y);
    
    TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001);
    vector<Point3f> objp;
    for (int i = 0; i < COINS_Y; ++i) {
        for (int j = 0; j < COINS_X; ++j) {
            objp.push_back(Point3f((float)j, (float)i, 0));
        }
    }

    vector<vector<Point3f>> objpoints;
    vector<vector<Point2f>> imgpoints;
    vector<String> images;
    glob("repertoire_image/*.jpg", images);

    for (auto const& fname : images) {
        Mat img = imread(fname);
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        
        vector<Point2f> corners;
        bool ret = findChessboardCorners(gray, patternSize, corners);
        if (ret) {
            objpoints.push_back(objp);
            cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), criteria);
            imgpoints.push_back(corners);
            drawChessboardCorners(img, patternSize, corners, ret);
            imshow("img", img);
            waitKey(0);
        }
    }
    destroyAllWindows();

    Mat cameraMatrix, distCoeffs;
    vector<Mat> rvecs, tvecs;
    calibrateCamera(objpoints, imgpoints, Size(images[0].size()), cameraMatrix, distCoeffs, rvecs, tvecs);

    Mat img = imread("vue_au_sol.jpg");
    
    auto corners_unwarp = [](Mat& img, int nx, int ny, Mat& mtx, Mat& dist) {
        Mat undistorted;
        undistort(img, undistorted, mtx, dist);
        
        Mat gray;
        cvtColor(undistorted, gray, COLOR_BGR2GRAY);
        
        vector<Point2f> corners;
        bool ret = findChessboardCorners(gray, Size(nx, ny), corners);
        Mat warped;
        Mat M;

        if (ret) {
            int offset = 100;
            Size imgSize = undistorted.size();
            drawChessboardCorners(undistorted, Size(nx, ny), corners, ret);
            vector<Point2f> src = { corners[0], corners[nx - 1], corners.back(), corners[nx * (ny - 1)] };
            vector<Point2f> dst = { Point2f(offset, imgSize.height / 2), Point2f(imgSize.width - offset, imgSize.height / 2),
                                    Point2f(imgSize.width - offset, imgSize.height - offset), Point2f(offset, imgSize.height - offset) };
            M = getPerspectiveTransform(src, dst);
            warpPerspective(undistorted, warped, M, imgSize, INTER_LINEAR);
            cout << "etalonnage OK" << endl;
        }
        return make_pair(warped, M);
    };

    Mat top_down;
    Mat perspective_M;
    tie(top_down, perspective_M) = corners_unwarp(img, COINS_X, COINS_Y, cameraMatrix, distCoeffs);
    
    imshow("img", top_down);
    waitKey(0);

    FileStorage fs("parametres_calibration_robot.yml", FileStorage::WRITE);
    fs << "perspective_M" << perspective_M;
    fs << "cameraMatrix" << cameraMatrix;
    fs << "distCoeffs" << distCoeffs;
    fs.release();

    return 0;
}


