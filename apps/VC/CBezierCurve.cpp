// CBezierCurve.cpp
// Chao Du 2015 April

// Ignore this warning because it is very noisy for this file
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-conversion"

#include "CBezierCurve.hpp"

#include <opencv2/imgproc.hpp>

//#define _DEBUG

using namespace ChaoVis;

// Bezier point
inline float GetPt(int n1, int n2, float percent)
{
    int diff = n2 - n1;
    return n1 + (diff * percent);
}

// Constructor
CBezierCurve::CBezierCurve(int nSampleInterval)
    : fNumControlPoints(4), fSampleInterval(nSampleInterval)
{
    for (int i = 0; i < fNumControlPoints; ++i) {
        fControlPoints.push_back(Vec2<double>(0.0, 0.0));
    }
}

// Constructor
CBezierCurve::CBezierCurve(
    const std::vector<Vec2<double>>& nControlPoints, int nSampleInterval)
    : fNumControlPoints(4), fSampleInterval(nSampleInterval)
{
    assert(static_cast<int>(nControlPoints.size()) == fNumControlPoints);  // 4

    for (int i = 0; i < fNumControlPoints; ++i) {
        fControlPoints.push_back(nControlPoints[i]);
    }
}

// Set control points
void CBezierCurve::SetControlPoints(
    const std::vector<Vec2<double>>& nControlPoints)
{
    assert(static_cast<int>(nControlPoints.size()) == fNumControlPoints);  // 4

    for (int i = 0; i < fNumControlPoints; ++i) {
        fControlPoints[i] = nControlPoints[i];
    }
}

// Set control points
void CBezierCurve::SetControlPoints(
    const std::vector<cv::Vec2f>& nControlPoints)
{
    assert(static_cast<int>(nControlPoints.size()) == fNumControlPoints);  // 4

    for (int i = 0; i < fNumControlPoints; ++i) {
        fControlPoints[i] =
            Vec2<double>(nControlPoints[i][0], nControlPoints[i][1]);
    }
}

// Get sample points
void CBezierCurve::GetSamplePoints(std::vector<Vec2<double>>& nSamplePoints)
{
    double aTotalLength = pythag<double>(
                              fControlPoints[0][0] - fControlPoints[1][0],
                              fControlPoints[0][1] - fControlPoints[1][1]) +
                          pythag<double>(
                              fControlPoints[1][0] - fControlPoints[2][0],
                              fControlPoints[1][1] - fControlPoints[2][1]) +
                          pythag<double>(
                              fControlPoints[2][0] - fControlPoints[3][0],
                              fControlPoints[2][1] - fControlPoints[3][1]);

    int aNumOfPts = aTotalLength / fSampleInterval;
    float aInterval = 1.0f / aNumOfPts;

    for (int i = 0; i < aNumOfPts; i++) {
        float xa =
            GetPt(fControlPoints[0][0], fControlPoints[1][0], i * aInterval);
        float ya =
            GetPt(fControlPoints[0][1], fControlPoints[1][1], i * aInterval);
        float xb =
            GetPt(fControlPoints[1][0], fControlPoints[2][0], i * aInterval);
        float yb =
            GetPt(fControlPoints[1][1], fControlPoints[2][1], i * aInterval);
        float xc =
            GetPt(fControlPoints[2][0], fControlPoints[3][0], i * aInterval);
        float yc =
            GetPt(fControlPoints[2][1], fControlPoints[3][1], i * aInterval);

        float xxa = GetPt(xa, xb, i * aInterval);
        float yya = GetPt(ya, yb, i * aInterval);
        float xxb = GetPt(xb, xc, i * aInterval);
        float yyb = GetPt(yb, yc, i * aInterval);

        float xxx = GetPt(xxa, xxb, i * aInterval);
        float yyy = GetPt(yya, yyb, i * aInterval);

        nSamplePoints.push_back(Vec2<double>(xxx, yyy));
    }
}

// Get sample points
void CBezierCurve::GetSamplePoints(std::vector<cv::Vec2f>& nSamplePoints)
{
    double aTotalLength = pythag<double>(
                              fControlPoints[0][0] - fControlPoints[1][0],
                              fControlPoints[0][1] - fControlPoints[1][1]) +
                          pythag<double>(
                              fControlPoints[1][0] - fControlPoints[2][0],
                              fControlPoints[1][1] - fControlPoints[2][1]) +
                          pythag<double>(
                              fControlPoints[2][0] - fControlPoints[3][0],
                              fControlPoints[2][1] - fControlPoints[3][1]);

    int aNumOfPts = aTotalLength / fSampleInterval;
    float aInterval = 1.0f / aNumOfPts;

    for (int i = 0; i < aNumOfPts; i++) {
        float xa =
            GetPt(fControlPoints[0][0], fControlPoints[1][0], i * aInterval);
        float ya =
            GetPt(fControlPoints[0][1], fControlPoints[1][1], i * aInterval);
        float xb =
            GetPt(fControlPoints[1][0], fControlPoints[2][0], i * aInterval);
        float yb =
            GetPt(fControlPoints[1][1], fControlPoints[2][1], i * aInterval);
        float xc =
            GetPt(fControlPoints[2][0], fControlPoints[3][0], i * aInterval);
        float yc =
            GetPt(fControlPoints[2][1], fControlPoints[3][1], i * aInterval);

        float xxa = GetPt(xa, xb, i * aInterval);
        float yya = GetPt(ya, yb, i * aInterval);
        float xxb = GetPt(xb, xc, i * aInterval);
        float yyb = GetPt(yb, yc, i * aInterval);

        float xxx = GetPt(xxa, xxb, i * aInterval);
        float yyy = GetPt(yya, yyb, i * aInterval);

        nSamplePoints.push_back(cv::Vec2f(xxx, yyy));
    }
}

// // Draw curve on image
// void CBezierCurve::DrawOnImage(cv::Mat& nImg, const cv::Scalar& nColor)
// {
// //    circle( nImg, cv::Point2f( 1000, 500 ), 55, cv::Scalar( 0, 0, 255 ) );
// // REVISIT - FILL ME HERE
// #ifdef _DEBUG
//     cv::line(
//         nImg, cv::Point2f(fControlPoints[0][0], fControlPoints[0][1]),
//         cv::Point2f(fControlPoints[1][0], fControlPoints[1][1]),
//         cv::Scalar(0, 0, 255), 3);
//     cv::line(
//         nImg, cv::Point2f(fControlPoints[1][0], fControlPoints[1][1]),
//         cv::Point2f(fControlPoints[2][0], fControlPoints[2][1]),
//         cv::Scalar(0, 255, 0), 2);
//     cv::line(
//         nImg, cv::Point2f(fControlPoints[2][0], fControlPoints[2][1]),
//         cv::Point2f(fControlPoints[3][0], fControlPoints[3][1]),
//         cv::Scalar(255, 0, 0), 1);
// #endif  // _DEBUG

//     double aTotalLen = pythag<double>(
//                            fControlPoints[0][0] - fControlPoints[1][0],
//                            fControlPoints[0][1] - fControlPoints[1][1]) +
//                        pythag<double>(
//                            fControlPoints[1][0] - fControlPoints[2][0],
//                            fControlPoints[1][1] - fControlPoints[2][1]) +
//                        pythag<double>(
//                            fControlPoints[2][0] - fControlPoints[3][0],
//                            fControlPoints[2][1] - fControlPoints[3][1]);

//     int aNumOfPts = aTotalLen / fSampleInterval;
//     float aInterval = 1.0f / aNumOfPts;

//     float prev_x{}, prev_y{};
//     for (int i = 0; i < aNumOfPts; i++) {
//         float xa =
//             GetPt(fControlPoints[0][0], fControlPoints[1][0], i * aInterval);
//         float ya =
//             GetPt(fControlPoints[0][1], fControlPoints[1][1], i * aInterval);
//         float xb =
//             GetPt(fControlPoints[1][0], fControlPoints[2][0], i * aInterval);
//         float yb =
//             GetPt(fControlPoints[1][1], fControlPoints[2][1], i * aInterval);
//         float xc =
//             GetPt(fControlPoints[2][0], fControlPoints[3][0], i * aInterval);
//         float yc =
//             GetPt(fControlPoints[2][1], fControlPoints[3][1], i * aInterval);

//         float xxa = GetPt(xa, xb, i * aInterval);
//         float yya = GetPt(ya, yb, i * aInterval);
//         float xxb = GetPt(xb, xc, i * aInterval);
//         float yyb = GetPt(yb, yc, i * aInterval);

//         float xxx = GetPt(xxa, xxb, i * aInterval);
//         float yyy = GetPt(yya, yyb, i * aInterval);

//         if (i == 0) {
//             prev_x = xxx;
//             prev_y = yyy;
//         } else {
//             cv::line(
//                 nImg, cv::Point2f(prev_x, prev_y), cv::Point2f(xxx, yyy),
//                 nColor);

//             prev_x = xxx;
//             prev_y = yyy;
//         }

// #ifdef _DEBUG
//         circle(
//             nImg, cv::Point2f(fControlPoints[0][0], fControlPoints[0][1]), 7,
//             cv::Scalar(255, 0, 0));
//         circle(
//             nImg, cv::Point2f(fControlPoints[1][0], fControlPoints[1][1]), 7,
//             cv::Scalar(255, 0, 0));
//         circle(
//             nImg, cv::Point2f(fControlPoints[2][0], fControlPoints[2][1]), 7,
//             cv::Scalar(255, 0, 0));
//         circle(
//             nImg, cv::Point2f(fControlPoints[3][0], fControlPoints[3][1]), 7,
//             cv::Scalar(255, 0, 0));
// #endif  // _DEBUG
//     }

// Draw curve on the QGraphicsScene
void CBezierCurve::DrawOnImage(QGraphicsScene* scene, const QColor& color)
{
    QPen pen(color);
    pen.setStyle(Qt::DashLine);

    // Debug colors
    QColor debugColor1(0, 0, 255);  // Blue
    QColor debugColor2(0, 255, 0);  // Green
    QColor debugColor3(255, 0, 0);  // Red

#ifdef _DEBUG
    // Debug control lines
    scene->addLine(QLineF(fControlPoints[0][0], fControlPoints[0][1],
                          fControlPoints[1][0], fControlPoints[1][1]), debugColor1);

    scene->addLine(QLineF(fControlPoints[1][0], fControlPoints[1][1],
                          fControlPoints[2][0], fControlPoints[2][1]), debugColor2);

    scene->addLine(QLineF(fControlPoints[2][0], fControlPoints[2][1],
                          fControlPoints[3][0], fControlPoints[3][1]), debugColor3);

    // Debug control points
    scene->addEllipse(fControlPoints[0][0]-3.5, fControlPoints[0][1]-3.5, 7, 7, QPen(), QBrush(debugColor3));
    scene->addEllipse(fControlPoints[1][0]-3.5, fControlPoints[1][1]-3.5, 7, 7, QPen(), QBrush(debugColor3));
    scene->addEllipse(fControlPoints[2][0]-3.5, fControlPoints[2][1]-3.5, 7, 7, QPen(), QBrush(debugColor3));
    scene->addEllipse(fControlPoints[3][0]-3.5, fControlPoints[3][1]-3.5, 7, 7, QPen(), QBrush(debugColor3));
#endif

    double aTotalLen = pythag<double>(
                           fControlPoints[0][0] - fControlPoints[1][0],
                           fControlPoints[0][1] - fControlPoints[1][1]) +
                       pythag<double>(
                           fControlPoints[1][0] - fControlPoints[2][0],
                           fControlPoints[1][1] - fControlPoints[2][1]) +
                       pythag<double>(
                           fControlPoints[2][0] - fControlPoints[3][0],
                           fControlPoints[2][1] - fControlPoints[3][1]);

    int aNumOfPts = aTotalLen / fSampleInterval;
    float aInterval = 1.0f / aNumOfPts;

    qDebug() << "aNumOfPts: " << aNumOfPts;

    float prev_x{}, prev_y{};
    for (int i = 0; i < aNumOfPts; i++) {
        float xa = GetPt(fControlPoints[0][0], fControlPoints[1][0], i * aInterval);
        float ya = GetPt(fControlPoints[0][1], fControlPoints[1][1], i * aInterval);
        float xb = GetPt(fControlPoints[1][0], fControlPoints[2][0], i * aInterval);
        float yb = GetPt(fControlPoints[1][1], fControlPoints[2][1], i * aInterval);
        float xc = GetPt(fControlPoints[2][0], fControlPoints[3][0], i * aInterval);
        float yc = GetPt(fControlPoints[2][1], fControlPoints[3][1], i * aInterval);

        float xxa = GetPt(xa, xb, i * aInterval);
        float yya = GetPt(ya, yb, i * aInterval);
        float xxb = GetPt(xb, xc, i * aInterval);
        float yyb = GetPt(yb, yc, i * aInterval);

        float xxx = GetPt(xxa, xxb, i * aInterval);
        float yyy = GetPt(yya, yyb, i * aInterval);

        if (i == 0) {
            prev_x = xxx;
            prev_y = yyy;
        } else {
            qDebug() << "prev_x: " << prev_x << " prev_y: " << prev_y << " xxx: " << xxx << " yyy: " << yyy;
            scene->addLine(prev_x, prev_y, xxx, yyy, pen);

            prev_x = xxx;
            prev_y = yyy;
        }

#ifdef _DEBUG
    // Draw debug circles on each point of the curve
    for (int i = 0; i < aNumOfPts; i++) {
        float xa = GetPt(fControlPoints[0][0], fControlPoints[1][0], i * aInterval);
        float ya = GetPt(fControlPoints[0][1], fControlPoints[1][1], i * aInterval);
        float xb = GetPt(fControlPoints[1][0], fControlPoints[2][0], i * aInterval);
        float yb = GetPt(fControlPoints[1][1], fControlPoints[2][1], i * aInterval);
        float xc = GetPt(fControlPoints[2][0], fControlPoints[3][0], i * aInterval);
        float yc = GetPt(fControlPoints[2][1], fControlPoints[3][1], i * aInterval);

        float xxa = GetPt(xa, xb, i * aInterval);
        float yya = GetPt(ya, yb, i * aInterval);
        float xxb = GetPt(xb, xc, i * aInterval);
        float yyb = GetPt(yb, yc, i * aInterval);

        float xxx = GetPt(xxa, xxb, i * aInterval);
        float yyy = GetPt(yya, yyb, i * aInterval);

        // Add a circle for this point
        scene->addEllipse(xxx - 1.5, yyy - 1.5, 3, 3, QPen(), QBrush(Qt::yellow));  // yellow for visibility
    }
#endif  // _DEBUG
    }
}


#pragma clang diagnostic pop
