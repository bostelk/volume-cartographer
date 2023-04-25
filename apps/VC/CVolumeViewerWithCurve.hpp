// CVolumeViewerWithCurve.h
// Chao Du 2015 April
#pragma once

#include <QCheckBox>
#include <vector>
#include <opencv2/core.hpp>
#include "CBSpline.hpp"
#include "CVolumeViewer.hpp"
#include "CXCurve.hpp"
#include "ColorFrame.hpp"

namespace ChaoVis
{

// REVISIT - NOTE - since there are two modes, edit and draw, for the
// application,
//           we need to add corresponding modes to the widget, too.

class CVolumeViewerWithCurve : public CVolumeViewer
{

    Q_OBJECT

public:
    enum EViewState {
        ViewStateEdit,  // edit mode
        ViewStateDraw,  // draw mode
        ViewStateIdle
    };  // idle mode

public:
    CVolumeViewerWithCurve();
    ~CVolumeViewerWithCurve() = default;

    virtual void SetImage(const QImage& nSrc);

    // for drawing mode
    void SetSplineCurve(CBSpline& nCurve);
    void UpdateSplineCurve(void);
    void ResetSplineCurve(void) { fControlPoints.clear(); }
    // for editing mode
    void SetIntersectionCurve(CXCurve& nCurve);
    void SetImpactRange(int nImpactRange);

    void UpdateView(void);
    void SetShowCurve(bool b) { showCurve = b; }

    void SetViewState(EViewState nViewState) { fViewState = nViewState; }
    EViewState GetViewState(void) { return fViewState; }

    void setButtonsEnabled(bool state);

protected:
    void mousePressEvent(QMouseEvent* e);
    void mouseMoveEvent(QMouseEvent* event);
    void mouseReleaseEvent(QMouseEvent* event);
    void paintEvent(QPaintEvent* event);
    void UpdateButtons(void);

private slots:
    void OnShowCurveStateChanged(int state);
    void OnHistEqStateChanged(int state);

private:
    void WidgetLoc2ImgLoc(const cv::Vec2f& nWidgetLoc, cv::Vec2f& nImgLoc);

    int SelectPointOnCurve(const CXCurve* nCurve, const cv::Vec2f& nPt);

    void DrawIntersectionCurve(void);

private slots:

signals:
    void SendSignalPathChanged(void);

private:
    // for drawing
    ColorFrame* colorSelector{nullptr};
    QCheckBox* fShowCurveBox;
    QCheckBox* fHistEqBox;
    bool showCurve;
    bool histEq;
    CBSpline* fSplineCurveRef;
    std::vector<cv::Vec2f> fControlPoints;

    // for editing
    CXCurve* fIntersectionCurveRef;
    int fSelectedPointIndex;
    bool fVertexIsChanged;

    QPointF fLastPos;  // last mouse position on the image
    int fImpactRange;  // how many points a control point movement can affect

    // image drawn
    cv::Mat fImgMat;
    cv::Mat fImgMatCache;

    EViewState fViewState;

};  // class CVolumeViewerWithCurve

}  // namespace ChaoVis
