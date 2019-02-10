//
// Created by prostoichelovek on 29.01.19.
//

#include "Filter.h"

Filter::Filter(int stateSize_, int measSize_,
               Scalar processNoiseCov, Scalar measurementNoiseCov, Scalar errorCovPost, int contrSize) {
    stateSize = stateSize_;
    measSize = measSize_;
    KF = KalmanFilter(stateSize, measSize, contrSize, F_type);
    meas = Mat(measSize, 1, F_type);

    setIdentity(KF.transitionMatrix);
    KF.measurementMatrix = Mat::zeros(measSize, stateSize, F_type);
    KF.measurementMatrix.at<float>(0) = 1.0f;
    KF.measurementMatrix.at<float>(5) = 1.0f;
    KF.processNoiseCov.at<float>(0) = 1e-2;
    KF.processNoiseCov.at<float>(5) = 1e-2;
    KF.processNoiseCov.at<float>(10) = 5.0f;
    KF.processNoiseCov.at<float>(15) = 5.0f;
    Mat_<float> measurement(2, 1);
    measurement.setTo(Scalar(0));
    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, processNoiseCov);
    setIdentity(KF.measurementNoiseCov, measurementNoiseCov);
    setIdentity(KF.errorCovPost, errorCovPost);
}

vector<float> Filter::predict() {
    double precTick = ticks;
    ticks = (double) getTickCount();
    double dT = (ticks - precTick) / getTickFrequency();

    KF.transitionMatrix.at<float>(2) = dT;
    KF.transitionMatrix.at<float>(7) = dT;
    KF.predict();
    Mat estimated = KF.correct(meas);
    vector<float> res;
    for (int i = 0; i < stateSize; i++) {
        res.emplace_back(estimated.at<float>(i));
    }
    return res;
}

void Filter::update(const vector<float> newData) {
    for (int i = 0; i < measSize; i++) {
        if (newData.size() == i)
            return;
        meas.at<float>(i) = newData[i];
    }
}
