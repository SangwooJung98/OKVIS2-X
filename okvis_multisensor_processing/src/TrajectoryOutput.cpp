/**
 * OKVIS2-X - Open Keyframe-based Visual-Inertial SLAM Configurable with Dense 
 * Depth or LiDAR, and GNSS
 *
 * Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 * Copyright (c) 2020, Smart Robotics Lab / Imperial College London
 * Copyright (c) 2025, Mobile Robotics Lab / Technical University of Munich 
 * and ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause, see LICENESE file for details
 */

/**
 * @file TrajectoryOutput.cpp
 * @brief Source file for the TrajectoryOutput class.
 * @author Stefan Leutenegger
 */

#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <glog/logging.h>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <okvis/TrajectoryOutput.hpp>
#include <okvis/RealtimePublisher.hpp>

namespace {

cv::Mat makeStreamReady(const cv::Mat& image) {
  if (image.empty()) {
    return cv::Mat();
  }
  if (image.type() == CV_8UC3) {
    return image.clone();
  }
  cv::Mat converted;
  if (image.channels() == 1) {
    cv::cvtColor(image, converted, cv::COLOR_GRAY2BGR);
    return converted;
  }
  if (image.channels() == 4) {
    cv::cvtColor(image, converted, cv::COLOR_BGRA2BGR);
    return converted;
  }
  image.convertTo(converted, CV_8UC3, 255.0);
  return converted;
}

}  // namespace


okvis::TrajectoryOutput::TrajectoryOutput(bool draw) : draw_(draw) {
  if(draw_) {
    _image.create(_imageSize, _imageSize, CV_8UC3);
    _image.setTo(cv::Scalar(0,0,0));
  }
}

okvis::TrajectoryOutput::TrajectoryOutput(const std::string &filename, bool rpg, bool draw)
    : draw_(draw) {
  setCsvFile(filename, rpg);
  if(draw_) {
    _image.create(_imageSize, _imageSize, CV_8UC3);
    _image.setTo(cv::Scalar(0,0,0));
  }
}

okvis::TrajectoryOutput::~TrajectoryOutput() {
  if(jsonEnabled_ && jsonFile_.is_open()) {
    jsonFile_ << "\n  ]\n}\n";
    jsonFile_.close();
  }
}

void okvis::TrajectoryOutput::setCsvFile(const std::string &filename, bool rpg) {
  rpg_ = rpg;
  createCsvFile(filename, csvFile_, rpg);
}

void okvis::TrajectoryOutput::setRGBCsvFile(const std::string & filename) {
  createCsvFile(filename, rgbCsvFile_, rpg_);
}

bool okvis::TrajectoryOutput::setJsonCamera(const okvis::cameras::NCameraSystem& ncamera,
                                            size_t camIdx,
                                            const std::string& imagePrefix,
                                            const std::string& imageExt,
                                            int zeroPad) {
  if(camIdx >= ncamera.numCameras()) {
    LOG(WARNING) << "JSON camera index " << camIdx << " out of range.";
    return false;
  }

  auto camGeom = ncamera.cameraGeometry(camIdx);
  if(!camGeom) {
    LOG(WARNING) << "Camera geometry not available for index " << camIdx;
    return false;
  }

  auto pinhole = std::dynamic_pointer_cast<const okvis::cameras::PinholeCameraBase>(camGeom);
  if(!pinhole) {
    LOG(WARNING) << "JSON export currently supports pinhole cameras only.";
    return false;
  }

  jsonIntrinsics_.fx = pinhole->focalLengthU();
  jsonIntrinsics_.fy = pinhole->focalLengthV();
  jsonIntrinsics_.cx = pinhole->imageCenterU();
  jsonIntrinsics_.cy = pinhole->imageCenterV();
  jsonIntrinsics_.width = pinhole->imageWidth();
  jsonIntrinsics_.height = pinhole->imageHeight();
  jsonIntrinsics_.focal = jsonIntrinsics_.fx;
  jsonIntrinsics_.valid = true;

  auto T_SC_ptr = ncamera.T_SC(camIdx);
  if(!T_SC_ptr) {
    LOG(WARNING) << "Extrinsics (T_SC) not available for camera " << camIdx;
    return false;
  }
  jsonT_SC_ = *T_SC_ptr;
  jsonCameraSet_ = true;
  jsonCameraIdx_ = camIdx;
  depthCameraIdx_ = camIdx;
  jsonImagePrefix_ = imagePrefix;
  jsonImageExt_ = imageExt;
  jsonImageZeroPad_ = zeroPad;
  jsonFrameCounter_ = 0;
  tryAttachRealtimePublisher();
  return true;
}

void okvis::TrajectoryOutput::setDepthExportConfig(double minRangeMeters,
                                                   double maxRangeMeters,
                                                   double depthScaleMetersPerUnit,
                                                   int pixelStride) {
  depthMinRange_ = std::max(0.0, minRangeMeters);
  depthMaxRange_ = std::max(depthMinRange_ + 0.1, maxRangeMeters);
  depthScaleMetersPerUnit_ = depthScaleMetersPerUnit;
  depthPixelStride_ = std::max(1, pixelStride);
  depthCameraIdx_ = jsonCameraIdx_;

  if(!jsonCameraSet_) {
    LOG(WARNING) << "[DepthPC] setDepthExportConfig() called before setJsonCamera(). Depth export disabled.";
    depthPointCloudEnabled_ = false;
    return;
  }

  depthPointCloudEnabled_ = depthPointCloudDirReady_;
  if(!depthPointCloudEnabled_) {
    LOG(WARNING) << "[DepthPC] point_cloud_depth directory unavailable. Depth export disabled.";
  } else {
    LOG(INFO) << "[DepthPC] enabled on camIdx=" << depthCameraIdx_
              << " range=[" << depthMinRange_ << "," << depthMaxRange_
              << "] stride=" << depthPixelStride_;
  }
  tryAttachRealtimePublisher();
}

void okvis::TrajectoryOutput::setJsonFile(const std::string & filename) {
  configureRealtimePublisherFromEnv();
  jsonFile_.open(filename.c_str(), std::ios_base::out);
  OKVIS_ASSERT_TRUE(Exception, jsonFile_.good(),
                    "couldn't create trajectory JSON file at " << filename)
  jsonFile_ << "{\n  \"cameras\": [\n";
  jsonDir_ = filename;
  auto pos = jsonDir_.find_last_of("/\\");
  if(pos != std::string::npos) {
    jsonDir_ = jsonDir_.substr(0, pos);
  } else {
    jsonDir_ = ".";
  }
  imagesDir_ = jsonDir_ + "/images";
  try {
    std::filesystem::create_directories(imagesDir_);
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to create images directory at " << imagesDir_ << ": " << e.what();
  }
  pointCloudDir_ = jsonDir_ + "/point_cloud";
  try {
    std::filesystem::create_directories(pointCloudDir_);
    pointCloudEnabled_ = true;
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to create point_cloud directory at " << pointCloudDir_ << ": " << e.what();
    pointCloudEnabled_ = false;
  }
  pointCloudDepthDir_ = jsonDir_ + "/point_cloud_depth";
  try {
    std::filesystem::create_directories(pointCloudDepthDir_);
    depthPointCloudDirReady_ = true;
  } catch (const std::exception& e) {
    LOG(WARNING) << "Failed to create point_cloud_depth directory at " << pointCloudDepthDir_ << ": " << e.what();
    depthPointCloudDirReady_ = false;
  }
  // pointCloudPlyDir_ = jsonDir_ + "/point_cloud_ply";
  // try {
  //   std::filesystem::create_directories(pointCloudPlyDir_);
  //   pointCloudPlyEnabled_ = true;
  // } catch (const std::exception& e) {
  //   LOG(WARNING) << "Failed to create point_cloud_ply directory at " << pointCloudPlyDir_ << ": " << e.what();
  //   pointCloudPlyEnabled_ = false;
  // }
  imageListPath_ = jsonDir_ + "/image_list.txt";
  imageListFile_.open(imageListPath_, std::ios::out | std::ios::trunc);
  if(!imageListFile_.good()) {
    LOG(WARNING) << "Failed to open image_list.txt at " << imageListPath_;
    imageListEnabled_ = false;
  } else {
    imageListEnabled_ = true;
  }
  jsonEnabled_ = true;
  firstJsonEntry_ = true;
}

void okvis::TrajectoryOutput::processState(
    const State& state, const TrackingState & trackingState,
    std::shared_ptr<const AlignedMap<StateId, State>> updatedStates,
    std::shared_ptr<const MapPointVector> landmarks) {
  // resolve image name for this timestamp; if not found, try oldest pending to keep counts aligned
  std::string imageNameResolved;
  {
    std::lock_guard<std::mutex> lock(imageMutex_);
    auto it = imageNameByStamp_.find(toKey(state.timestamp));
    if(it != imageNameByStamp_.end()) {
      imageNameResolved = it->second;
      imageNameByStamp_.erase(it);
    } else if(!imageNameByStamp_.empty()) {
      // fallback: take oldest pending image to keep 1:1 count
      auto itOldest = imageNameByStamp_.begin();
      imageNameResolved = itOldest->second;
      imageNameByStamp_.erase(itOldest);
    }
  }

  writeStateToCsv(csvFile_, state, rpg_);

  cv::Mat depthImageResolved;
  if(depthPointCloudEnabled_) {
    std::lock_guard<std::mutex> lock(depthMutex_);
    const uint64_t key = toKey(state.timestamp);
    auto itDepth = depthImagesByStamp_.find(key);
    if(itDepth != depthImagesByStamp_.end()) {
      depthImageResolved = itDepth->second;
      depthImagesByStamp_.erase(itDepth);
    } else if(!depthImagesByStamp_.empty()) {
      const uint64_t tolerance = 5000000; // 5ms tolerance in nanoseconds
      auto lower = depthImagesByStamp_.lower_bound(key);
      auto pick = depthImagesByStamp_.end();
      if(lower != depthImagesByStamp_.end()) {
        if(lower->first - key <= tolerance) {
          pick = lower;
        }
      }
      if(pick == depthImagesByStamp_.end() && lower != depthImagesByStamp_.begin()) {
        auto prev = std::prev(lower);
        if(key >= prev->first && key - prev->first <= tolerance) {
          pick = prev;
        }
      }
      if(pick == depthImagesByStamp_.end()) {
        pick = depthImagesByStamp_.begin();
      }
      depthImageResolved = pick->second;
      depthImagesByStamp_.erase(pick);
    }
  }

  const uint64_t timestampKey = toKey(state.timestamp);
  // if image index < 2, skip JSON/PC to align counts
  int imgIdx = imageIndexFromName(imageNameResolved);
  const bool validImageForOutputs = !imageNameResolved.empty() && imgIdx >= 2;
  if(jsonEnabled_) {
    if(!validImageForOutputs) {
      LOG_EVERY_N(WARNING, 200) << "Skip JSON/PC for ts=" << timestampKey
                                << " (no saved image matched)";
    } else if(jsonCameraSet_ && jsonIntrinsics_.valid) {
      writeStateToJson(state, imageNameResolved);
      jsonFile_.flush(); // keep file consistent for realtime consumers
    } else {
      LOG_EVERY_N(WARNING, 200) << "JSON output enabled but camera intrinsics not set.";
    }
  }

  const bool wantsRealtimeStream =
      realtimePublisher_ && realtimePublisher_->isReady();
  std::vector<Eigen::Vector3d,
              Eigen::aligned_allocator<Eigen::Vector3d>>
      sparsePoints;
  std::vector<Eigen::Vector3d,
              Eigen::aligned_allocator<Eigen::Vector3d>>
      depthPointsWorld;
  if ((pointCloudEnabled_ || wantsRealtimeStream) && validImageForOutputs) {
    collectLandmarkPoints(state, landmarks, sparsePoints);
  }

  if(depthPointCloudEnabled_ && validImageForOutputs && !depthImageResolved.empty()) {
    if(!computeDepthPointCloudWorld(state, depthImageResolved, depthPointsWorld)) {
      LOG_EVERY_N(WARNING, 200) << "Failed to build depth-based point cloud for frame "
                                << state.id.value();
    }
  }

  // per-frame sparse point cloud saving
  if(pointCloudEnabled_ && validImageForOutputs) {
    if(!writePointCloud(state, sparsePoints, imageNameResolved)) {
      LOG_EVERY_N(WARNING, 200) << "Failed to write point cloud for frame " << state.id.value();
    }
  }

  if (wantsRealtimeStream && validImageForOutputs) {
    cv::Mat streamImage;
    if(popStreamImage(timestampKey, streamImage) && !streamImage.empty()) {
      submitRealtimeFrame(state, imgIdx, imageNameResolved, streamImage,
                          sparsePoints, depthPointsWorld);
    } else {
      LOG_EVERY_N(WARNING, 200)
          << "[RealtimePublisher] Missing RGB buffer for timestamp "
          << timestampKey;
    }
  }

  if(depthPointCloudEnabled_ && validImageForOutputs && !depthImageResolved.empty()) {
    if(!writeDepthPointCloud(state, depthPointsWorld, imageNameResolved)) {
      LOG_EVERY_N(WARNING, 200) << "Failed to write depth-based point cloud for frame "
                                << state.id.value();
    }
  } else if(depthPointCloudEnabled_ && validImageForOutputs && depthImageResolved.empty()) {
    LOG_EVERY_N(INFO, 100) << "[DepthPC] No depth match for ts=" << toKey(state.timestamp)
                           << " buffer_size=" << depthImagesByStamp_.size();
  }

  if(draw_ && !updatedStates->empty()) {
    std::shared_ptr<GraphStates> statePtr(
          new GraphStates{state, trackingState, updatedStates, landmarks});
    states_.PushBlockingIfFull(statePtr, 1000);
  }
}

bool okvis::TrajectoryOutput::addImuMeasurement(
  const okvis::Time &stamp, const Eigen::Vector3d &alpha, const Eigen::Vector3d &omega)
{
  if(draw_) {
    return !imuMeasurements_.PushNonBlockingDroppingIfFull(
      ImuMeasurement(stamp, ImuSensorReadings(omega,alpha)), 1000);
  }
  return false;
}

bool okvis::TrajectoryOutput::processRGBImage(const okvis::Time& timestamp, const cv::Mat& image) {
  // Save image to disk for COLMAP-style export
  if(!imagesDir_.empty() && !image.empty()) {
    const std::string imageName = nextImageName();
    const std::string fullPath = imagesDir_ + "/" + imageName;
    try {
      cv::imwrite(fullPath, image);
      std::lock_guard<std::mutex> lock(imageMutex_);
      imageNameByStamp_[toKey(timestamp)] = imageName;
      // LOG(INFO) << "Saved image " << fullPath << " ts=" << toKey(timestamp);
      // append to image_list if index >=2
      int idx = imageIndexFromName(imageName);
      if(imageListEnabled_ && idx >= 2) {
        imageListFile_ << imageName << " " << idx << "\n";
        imageListFile_.flush();
      }
      if(realtimePublisher_) {
        const uint64_t key = toKey(timestamp);
        cv::Mat streamImage = makeStreamReady(image);
        if(!streamImage.empty()) {
          pushStreamImage(key, streamImage);
        }
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to write image " << fullPath << ": " << e.what();
    }
  }

  // Optional: if an RGB CSV is set, also log pose at RGB timestamps
  if (rgbCsvFile_) {
    rgbTrajectory_.enqueue(image, timestamp);
    auto state_pairs = rgbTrajectory_.getStates(trajectory_);
    for (auto state_pair : state_pairs) {
      writeStateToCsv(rgbCsvFile_, state_pair.second, rpg_);
    }
  }
  return true;
}

bool okvis::TrajectoryOutput::processDepthImage(const okvis::Time& timestamp,
                                                size_t camIdx,
                                                const cv::Mat& depthImage) {
  if(!depthPointCloudEnabled_) {
    return false;
  }
  if(camIdx != depthCameraIdx_) {
    return false;
  }
  if(depthImage.empty()) {
    return false;
  }

  cv::Mat depthFloat;
  if(depthImage.type() == CV_32F) {
    depthFloat = depthImage.clone();
  } else {
    depthImage.convertTo(depthFloat, CV_32F);
  }

  {
    std::lock_guard<std::mutex> lock(depthMutex_);
    depthImagesByStamp_[toKey(timestamp)] = depthFloat;
    while(depthImagesByStamp_.size() > depthImageBufferSize_) {
      depthImagesByStamp_.erase(depthImagesByStamp_.begin());
    }
  }
  // LOG(INFO) << "[DepthPC] buffered depth ts=" << toKey(timestamp)
  //           << " camIdx=" << camIdx
  //           << " rows=" << depthFloat.rows
  //           << " cols=" << depthFloat.cols;
  return true;
}

void okvis::TrajectoryOutput::drawTopView(cv::Mat& outImg) {
  std::shared_ptr<GraphStates> graphStates;
  while(states_.PopNonBlocking(&graphStates)) {

    // do the update here to avoid race conditions
    std::set<okvis::StateId> affectedStateIds;
    trajectory_.update(std::get<1>(*graphStates), std::get<2>(*graphStates), affectedStateIds);
    State state = std::get<0>(*graphStates);

    // also propagate IMU, if requested
    if(states_.Empty()) {
      ImuMeasurement imuMeasurement;
      while(imuMeasurements_.PopNonBlocking(&imuMeasurement)) {
        State propagatedState;
        if(trajectory_.addImuMeasurement(imuMeasurement.timeStamp,
                                      imuMeasurement.measurement.accelerometers,
                                      imuMeasurement.measurement.gyroscopes,
                                      propagatedState)) {
          state = propagatedState;
        }
      }
    }

    // append the path
    Eigen::Vector3d r = state.T_WS.r();
    Eigen::Matrix3d C = state.T_WS.C();
    _path.push_back(cv::Point2d(r[0], r[1]));
    _heights.push_back(r[2]);

    // maintain scaling
    if (r[0] - _frameScale < _min_x)
      _min_x = r[0] - _frameScale;
    if (r[1] - _frameScale < _min_y)
      _min_y = r[1] - _frameScale;
    if (r[2] < _min_z)
      _min_z = r[2];
    if (r[0] + _frameScale > _max_x)
      _max_x = r[0] + _frameScale;
    if (r[1] + _frameScale > _max_y)
      _max_y = r[1] + _frameScale;
    if (r[2] > _max_z)
      _max_z = r[2];
    _scale = std::min(_imageSize / (_max_x - _min_x), _imageSize / (_max_y - _min_y));

    // reset image
    _image.setTo(cv::Scalar(10, 10, 10));

    // First the landmarks
    const MapPointVector& landmarks = *std::get<3>(*graphStates);
    for(const auto & lm : landmarks) {
      if(fabs(lm.point[3])<1e-12) continue;
      if(lm.quality<0.0001) continue;
      Eigen::Vector3d pt3d = lm.point.head<3>()/lm.point[3];
      cv::Point2d pt = convertToImageCoordinates(cv::Point2d(pt3d[0], pt3d[1]));
      if (pt.x<0 || pt.y<0 || pt.x > _imageSize-1 || pt.y > _imageSize-1) continue;
      const double colourScale = std::min(1.0,lm.quality/0.1);
      cv::circle(_image, pt, 1.0, colourScale*cv::Scalar(0,255,0), cv::FILLED, cv::LINE_AA);
    }

    // Get GPS Measurements
    const AlignedMap<StateId, State> updated_states =  *std::get<2>(*graphStates);
    for(auto updated_state : updated_states){
      kinematics::Transformation T_WG = updated_state.second.T_GW.inverse();
      for(auto& measurement : updated_state.second.gpsPoints){
        Eigen::Vector3d point_in_W = (T_WG.T() * measurement.homogeneous()).head<3>();
        _gps.push_back(cv::Point2d(point_in_W.x(), point_in_W.y()));
      }
    }

    // draw the path
    drawPath();

    // Draw GPS
    if(_gps.size() > 0){
      drawGps();
    }

    // draw non-causal trajectory
    drawPathNoncausal();

    // draw IMU frame axes
    Eigen::Vector3d e_x = C.col(0);
    Eigen::Vector3d e_y = C.col(1);
    Eigen::Vector3d e_z = C.col(2);
    cv::line(
          _image,
          convertToImageCoordinates(_path.back()),
          convertToImageCoordinates(
            _path.back() + cv::Point2d(e_x[0], e_x[1]) * _frameScale),
        cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    cv::line(
          _image,
          convertToImageCoordinates(_path.back()),
          convertToImageCoordinates(
            _path.back() + cv::Point2d(e_y[0], e_y[1]) * _frameScale),
        cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    cv::line(
          _image,
          convertToImageCoordinates(_path.back()),
          convertToImageCoordinates(
            _path.back() + cv::Point2d(e_z[0], e_z[1]) * _frameScale),
        cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

    // some text:
    std::stringstream postext;
    postext << "position = [" << r[0] << ", " << r[1] << ", " << r[2] << "]";
    cv::putText(_image, postext.str(), cv::Point(15,15),
                cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
    std::stringstream veltext;
    veltext << "velocity = [" << state.v_W[0] << ", " << state.v_W[1] << ", " << state.v_W[2]
            << "]";
    cv::putText(_image, veltext.str(), cv::Point(15,35),
                cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
    std::stringstream gyroBiasText;
    gyroBiasText << "gyro bias = [" << state.b_g[0] << ", " << state.b_g[1] << ", " << state.b_g[2]
                 << "]";
    cv::putText(_image, gyroBiasText.str(), cv::Point(15,55),
                cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);
    std::stringstream accBiasText;
    accBiasText << "acc bias = [" << state.b_a[0] << ", " << state.b_a[1] << ", " << state.b_a[2]
                << "]";
    cv::putText(_image, accBiasText.str(), cv::Point(15,75),
                  cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,255), 1, cv::LINE_AA);

    // Quality
    const TrackingState& tracking = std::get<1>(*graphStates);
    cv::Scalar trackingColour;
    if(tracking.trackingQuality == TrackingQuality::Lost) {
      trackingColour = cv::Scalar(0,0,255);
      cv::putText(_image, "TRACKING LOST", cv::Point2f(5,_imageSize-10), cv::FONT_HERSHEY_COMPLEX,
                  0.3, trackingColour, 1, cv::LINE_AA);
    } else if (tracking.trackingQuality == TrackingQuality::Marginal) {
      trackingColour = cv::Scalar(0,255,255);
      cv::putText(_image, "Tracking marginal", cv::Point2f(5,_imageSize-10),
                  cv::FONT_HERSHEY_COMPLEX, 0.3, trackingColour, 1, cv::LINE_AA);
    } else {
      trackingColour = cv::Scalar(0,255,0);
      cv::putText(_image, "Tracking good", cv::Point2f(5,_imageSize-10), cv::FONT_HERSHEY_COMPLEX,
                  0.3, trackingColour, 1, cv::LINE_AA);
    }
    if(tracking.recognisedPlace) {
      cv::putText(_image, "Recognised place", cv::Point2f(160,_imageSize-10),
                  cv::FONT_HERSHEY_COMPLEX, 0.3, cv::Scalar(255,0,0), 1, cv::LINE_AA);
    }

    // hack: agent i pose / world frame axes
    for (const auto &T_AW : state.T_AiW) {
      _T_AW = T_AW.second;
    }
    e_x = _T_AW.inverse().C().col(0);
    e_y = _T_AW.inverse().C().col(1);
    e_z = _T_AW.inverse().C().col(2);
    cv::Point2d r_WA(_T_AW.inverse().r()[0], _T_AW.inverse().r()[1]);
    cv::line(
      _image,
      convertToImageCoordinates(r_WA),
      convertToImageCoordinates(r_WA + cv::Point2d(e_x[0], e_x[1]) * 2.0 * _frameScale),
      cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    cv::line(
      _image,
      convertToImageCoordinates(r_WA),
      convertToImageCoordinates(r_WA + cv::Point2d(e_y[0], e_y[1]) * 2.0 *_frameScale),
      cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    cv::line(
      _image,
      convertToImageCoordinates(r_WA),
      convertToImageCoordinates(r_WA + cv::Point2d(e_z[0], e_z[1]) * 2.0 *_frameScale),
      cv::Scalar(255, 0, 0), 1, cv::LINE_AA);


    // output
    outImg = _image;
  }
}

cv::Point2d okvis::TrajectoryOutput::convertToImageCoordinates(
  const cv::Point2d &pointInMeters) const
{
  cv::Point2d pt = (pointInMeters - cv::Point2d(_min_x, _min_y)) * _scale;
  return cv::Point2d(pt.x, _imageSize - pt.y); // reverse y for more intuitive top-down plot
}

void okvis::TrajectoryOutput::drawPath()
{
  for (size_t i = 0; i + 1 < _path.size(); ) {
    cv::Point2d p0 = convertToImageCoordinates(_path[i]);
    cv::Point2d p1 = convertToImageCoordinates(_path[i + 1]);
    cv::Point2d diff = p1-p0;
    if(diff.dot(diff)<2.0){
      _path.erase(_path.begin() + i + 1);  // clean short segment
      _heights.erase(_heights.begin() + i + 1);
      continue;
    }
    cv::line(_image, p0, p1, cv::Scalar(80, 80, 80), 1, cv::LINE_AA);
    i++;
  }
}

void okvis::TrajectoryOutput::drawGps()
{
  for (size_t i = 0; i < _gps.size(); i++) {
    cv::Point2d p0 = convertToImageCoordinates(_gps[i]);
    if (p0.x<0 || p0.y<0 || p0.x > _imageSize-1 || p0.y > _imageSize-1) 
    {
      continue;
    }
    cv::drawMarker(_image, p0, cv::Scalar(255, 0, 255), cv::MARKER_DIAMOND, 12);
  }
  _gps.clear();
}

void okvis::TrajectoryOutput::drawPathNoncausal()
{
  auto stateIds = trajectory_.stateIds();
  std::vector<cv::Point2d> path; // Path in 2d.
  std::vector<double> heights; // Heights on the path.
  std::vector<bool> isKeyframe; // Keyframe status on the path.
  path.reserve(stateIds.size());
  heights.reserve(stateIds.size());
  isKeyframe.reserve(stateIds.size());
  for(const auto & id : stateIds) {
    State state;
    trajectory_.getState(id, state);
    Eigen::Vector3d r = state.T_WS.r();
    path.push_back(cv::Point2d(r[0],r[1]));
    heights.push_back(r[2]);
    isKeyframe.push_back(state.isKeyframe);

    // draw covisibilities, too

    if (state.isKeyframe) {
      cv::Point2d p0 = convertToImageCoordinates(cv::Point2d(r[0],r[1]));
      for (const auto &id : state.covisibleFrameIds) {
        State state1;
        trajectory_.getState(id, state1);
        Eigen::Vector3d r1 = state1.T_WS.r();
        cv::Point2d p1 = convertToImageCoordinates(cv::Point2d(r1[0],r1[1]));
        cv::line(_image, p0, p1, cv::Scalar(0, 127, 127), 1, cv::LINE_AA);
      }
    }
  }
 
  for (size_t i = 0, j = 1; j < path.size(); ) {
    cv::Point2d p0 = convertToImageCoordinates(path[i]);
    cv::Point2d p1 = convertToImageCoordinates(path[j]);
    if(isKeyframe[j]) {
      cv::circle(_image, p1, 1, cv::Scalar(255, 255, 0), cv::FILLED, cv::LINE_AA);
    }
    if(i==0) {
      cv::circle(_image, p0, 1, cv::Scalar(255, 255, 0), cv::FILLED, cv::LINE_AA);
    }
    cv::Point2d diff = p1-p0;
    if(diff.dot(diff)<2.0){
      ++j;
      continue;
    }
    double rel_height = (heights[i] - _min_z + heights[j] - _min_z)
                        * 0.5 / (_max_z - _min_z);
    cv::line(_image, p0, p1, rel_height * cv::Scalar(255, 0, 0)
            + (1.0 - rel_height) * cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    i = j;
    j++;
  }
}

bool okvis::TrajectoryOutput::createCsvFile(
  const std::string &filename, std::fstream& stream, const bool rpg) {
  stream.open(filename.c_str(), std::ios_base::out);
  OKVIS_ASSERT_TRUE(Exception, stream.good(), "couldn't create trajectory file at " << filename)
  if(rpg) {
    stream << "# timestamp tx ty tz qx qy qz qw" << std::endl;
  } else {
    stream << "timestamp" << ", " << "p_WS_W_x" << ", " << "p_WS_W_y" << ", "
           << "p_WS_W_z" << ", " << "q_WS_x" << ", " << "q_WS_y" << ", "
           << "q_WS_z" << ", " << "q_WS_w" << ", " << "v_WS_W_x" << ", "
           << "v_WS_W_y" << ", " << "v_WS_W_z" << ", " << "b_g_x" << ", "
           << "b_g_y" << ", " << "b_g_z" << ", " << "b_a_x" << ", " << "b_a_y"
           << ", " << "b_a_z" << std::endl;
  }
  return true;
}


bool okvis::TrajectoryOutput::writeStateToCsv(
  std::fstream& csvFile, const okvis::State& state, const bool rpg) {
  if (!csvFile.good()) {
    return false;
  }
  Eigen::Vector3d p_WS_W = state.T_WS.r();
  Eigen::Quaterniond q_WS =state. T_WS.q();
  if(rpg) {
    csvFile << std::setprecision(19) << state.timestamp.toSec() << " "
             << p_WS_W[0] << " " << p_WS_W[1] << " " << p_WS_W[2] << " "
             << q_WS.x() << " " << q_WS.y() << " " << q_WS.z() << " " << q_WS.w() << std::endl;
  } else {
    std::stringstream time;
    time << state.timestamp.sec << std::setw(9)
         << std::setfill('0') <<  state.timestamp.nsec;
    csvFile << time.str() << ", " << std::scientific
            << std::setprecision(18)
            << p_WS_W[0] << ", " << p_WS_W[1] << ", " << p_WS_W[2] << ", "
            << q_WS.x() << ", " << q_WS.y() << ", " << q_WS.z() << ", " << q_WS.w() << ", "
            << state.v_W[0] << ", " << state.v_W[1] << ", " << state.v_W[2] << ", "
            << state.b_g[0] << ", " << state.b_g[1] << ", " << state.b_g[2] << ", "
            << state.b_a[0] << ", " << state.b_a[1] << ", " << state.b_a[2] << ", " << std::endl;
  }
  return true;
}

std::string okvis::TrajectoryOutput::makeJsonImageName() {
  std::stringstream ss;
  ss << jsonImagePrefix_ << std::setw(jsonImageZeroPad_) << std::setfill('0')
     << jsonFrameCounter_ << "." << jsonImageExt_;
  return ss.str();
}

std::string okvis::TrajectoryOutput::nextImageName() {
  std::lock_guard<std::mutex> lock(imageMutex_);
  std::string name = makeJsonImageName();
  ++jsonFrameCounter_;
  return name;
}

uint64_t okvis::TrajectoryOutput::toKey(const okvis::Time& t) const {
  return static_cast<uint64_t>(t.sec) * 1000000000ull + static_cast<uint64_t>(t.nsec);
}

int okvis::TrajectoryOutput::imageIndexFromName(const std::string& name) const {
  if(name.size() <= jsonImagePrefix_.size() + jsonImageExt_.size()+1) return -1;
  try {
    auto numStr = name.substr(jsonImagePrefix_.size(),
                              name.size() - jsonImagePrefix_.size() - 1 - jsonImageExt_.size());
    return std::stoi(numStr);
  } catch (...) {
    return -1;
  }
}

void okvis::TrajectoryOutput::appendImageList(const std::string& name, int idx) {
  if(imageListEnabled_ && idx >= 2) {
    imageListFile_ << name << " " << idx << "\n";
    imageListFile_.flush();
  }
}

bool okvis::TrajectoryOutput::writeStateToJson(const okvis::State& state,
  const std::string& imageName) {
  if (!jsonFile_.good()) {
    return false;
  }

  // world->sensor (T_WS) composed with sensor->camera (jsonT_SC_) gives world->camera.
  // COLMAP 스타일 camera->world 행렬이 필요하므로 전체를 역변환해 camera->world를 기록한다.
  okvis::kinematics::Transformation T_WC = state.T_WS * jsonT_SC_;
  okvis::kinematics::Transformation T_CW = T_WC.inverse();
  Eigen::Matrix4d T = T_CW.T();

  if(!firstJsonEntry_) {
    jsonFile_ << ",\n";
  }
  firstJsonEntry_ = false;

  jsonFile_ << "    {\n"
            << "      \"T_camera_world\": [\n";
  for(int r = 0; r < 4; ++r) {
    jsonFile_ << "        [";
    for(int c = 0; c < 4; ++c) {
      jsonFile_ << std::setprecision(18) << T(r,c);
      if(c < 3) jsonFile_ << ",\n          ";
    }
    jsonFile_ << "]";
    if(r < 3) jsonFile_ << ",\n";
    else jsonFile_ << "\n";
  }

  jsonFile_ << "      ],\n"
            << "      \"image\": \"" << imageName << "\",\n"
            << "      \"intrinsic\": {\n"
            << "        \"fx\": " << jsonIntrinsics_.fx << ",\n"
            << "        \"fy\": " << jsonIntrinsics_.fy << ",\n"
            << "        \"cx\": " << jsonIntrinsics_.cx << ",\n"
            << "        \"cy\": " << jsonIntrinsics_.cy << "\n"
            << "      },\n"
            << "      \"width\": " << jsonIntrinsics_.width << ",\n"
            << "      \"height\": " << jsonIntrinsics_.height << ",\n"
            << "      \"focal\": " << jsonIntrinsics_.focal << "\n"
            << "    }";

  // LOG(INFO) << "Appended trajectory JSON entry frame=" << state.id.value()
  //           << " ts=" << toKey(state.timestamp) << " image=" << imageName;

  return true;
}

void okvis::TrajectoryOutput::collectLandmarkPoints(
    const okvis::State& state,
    std::shared_ptr<const okvis::MapPointVector> landmarks,
    std::vector<Eigen::Vector3d,
                Eigen::aligned_allocator<Eigen::Vector3d>>& pointsWorld) const {
  pointsWorld.clear();
  if(!landmarks) {
    return;
  }
  pointsWorld.reserve(landmarks->size());
  for(const auto& lm : *landmarks) {
    if(lm.stateId != state.id.value()) {
      continue;
    }
    Eigen::Vector3d p_W = lm.point.head<3>();
    pointsWorld.emplace_back(p_W);
  }
}

bool okvis::TrajectoryOutput::computeDepthPointCloudWorld(
    const okvis::State& state,
    const cv::Mat& depthImage,
    std::vector<Eigen::Vector3d,
                Eigen::aligned_allocator<Eigen::Vector3d>>& pointsWorld) const {
  pointsWorld.clear();
  if(!jsonIntrinsics_.valid) {
    LOG_EVERY_N(WARNING, 50)
        << "[DepthPC] JSON intrinsics invalid. Cannot project depth.";
    return false;
  }
  if(depthImage.empty()) {
    LOG_EVERY_N(WARNING, 50) << "[DepthPC] depth image empty for frame "
                             << state.id.value();
    return false;
  }

  cv::Mat depthFloat;
  if(depthImage.type() == CV_32F) {
    depthFloat = depthImage.clone();
  } else {
    depthImage.convertTo(depthFloat, CV_32F);
  }

  const double fx = jsonIntrinsics_.fx;
  const double fy = jsonIntrinsics_.fy;
  const double cx = jsonIntrinsics_.cx;
  const double cy = jsonIntrinsics_.cy;
  okvis::kinematics::Transformation T_WC = state.T_WS * jsonT_SC_;
  Eigen::Matrix4d T = T_WC.T();

  const int stride = std::max(1, depthPixelStride_);
  pointsWorld.reserve(static_cast<size_t>(depthFloat.rows) *
                      static_cast<size_t>(depthFloat.cols) / stride);

  for(int v = 0; v < depthFloat.rows; v += stride) {
    const float* rowPtr = depthFloat.ptr<float>(v);
    for(int u = 0; u < depthFloat.cols; u += stride) {
      const float rawDepth = rowPtr[u];
      if(!std::isfinite(rawDepth) || rawDepth <= 0.0f) {
        continue;
      }
      const double depthMeters =
          static_cast<double>(rawDepth) * depthScaleMetersPerUnit_;
      if(depthMeters < depthMinRange_ || depthMeters > depthMaxRange_) {
        continue;
      }
      const double z = depthMeters;
      const double x = (static_cast<double>(u) - cx) / fx * z;
      const double y = (static_cast<double>(v) - cy) / fy * z;
      Eigen::Vector4d p_C;
      p_C << x, y, z, 1.0;
      Eigen::Vector4d p_W = T * p_C;
      pointsWorld.emplace_back(p_W[0], p_W[1], p_W[2]);
    }
  }

  if(pointsWorld.empty()) {
    LOG_EVERY_N(INFO, 100)
        << "[DepthPC] filtered depth produced zero points for frame "
        << state.id.value();
    return false;
  }
  return true;
}

bool okvis::TrajectoryOutput::writePointCloud(
    const okvis::State& state,
    const std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d>>& pointsWorld,
    const std::string& imageName) {
  (void) imageName;
  if(!pointCloudEnabled_ || imageName.empty()) {
    return false;
  }
  if(!jsonCameraSet_) {
    return false;
  }

  const bool writeTxt = pointCloudEnabled_ && !pointCloudDir_.empty();
  if(!writeTxt /* && !writePly */) {
    return false;
  }

  bool wroteAny = false;
  if(writeTxt) {
    std::stringstream ssTxt;
    ssTxt << pointCloudDir_ << "/point_cloud_" << state.id.value() << ".txt";
    std::ofstream ofsTxt(ssTxt.str());
    if(ofsTxt.good()) {
      for(const auto& pt : pointsWorld) {
        ofsTxt << pt[0] << " " << pt[1] << " " << pt[2] << "\n";
      }
      ofsTxt.close();
      wroteAny = true;
    } else {
      LOG(WARNING) << "Failed to open sparse TXT point cloud at " << ssTxt.str();
    }
  }
  return wroteAny;
}

bool okvis::TrajectoryOutput::writeDepthPointCloud(
    const okvis::State& state,
    const std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d>>& pointsWorld,
    const std::string& imageName) {
  (void) imageName;
  if(pointCloudDepthDir_.empty()) {
    LOG_EVERY_N(WARNING, 50) << "[DepthPC] point_cloud_depth directory invalid.";
    return false;
  }
  if(imageName.empty()) {
    LOG_EVERY_N(WARNING, 50) << "[DepthPC] image name empty for frame " << state.id.value();
    return false;
  }
  if(pointsWorld.empty()) {
    LOG_EVERY_N(INFO, 100) << "[DepthPC] filtered depth produced zero points for frame "
                           << state.id.value();
    return false;
  }

  std::stringstream ssTxt;
  ssTxt << pointCloudDepthDir_ << "/point_cloud_" << state.id.value() << ".txt";
  std::ofstream ofsTxt(ssTxt.str());
  if(!ofsTxt.good()) {
    LOG(WARNING) << "Failed to open depth TXT point cloud at " << ssTxt.str();
    return false;
  }

  for(const auto& pt : pointsWorld) {
    ofsTxt << pt[0] << " " << pt[1] << " " << pt[2] << "\n";
  }
  ofsTxt.close();

  // LOG(INFO) << "Saved depth-based sparse point cloud for frame " << state.id.value()
  //           << " with " << pointsWorld.size() << " points.";
  return true;
}

void okvis::TrajectoryOutput::configureRealtimePublisherFromEnv() {
  if(realtimeEnvChecked_) {
    LOG(INFO) << "[RealtimePublisher] Env already checked: host=" << realtimeHost_
              << " port=" << realtimePort_;
    return;
  }
  realtimeEnvChecked_ = true;
  const char* hostEnv = std::getenv("OKVIS_STREAM_HOST");
  const char* portEnv = std::getenv("OKVIS_STREAM_PORT");
  if(!hostEnv || !portEnv) {
    return;
  }
  int port = std::atoi(portEnv);
  if(port <= 0 || port >= 65535) {
    LOG(WARNING) << "[RealtimePublisher] Invalid port specified via "
                 << "OKVIS_STREAM_PORT=" << portEnv;
    return;
  }
  realtimeHost_ = hostEnv;
  realtimePort_ = static_cast<uint16_t>(port);
  realtimeStreamingConfigured_ = true;
  LOG(INFO) << "[RealtimePublisher] Configured stream target " << realtimeHost_
            << ":" << realtimePort_;
}

void okvis::TrajectoryOutput::tryAttachRealtimePublisher() {
  if(!realtimeStreamingConfigured_) {
    LOG(INFO) << "[RealtimePublisher] Not configured; skipping attach.";
    return;
  }
  if(!jsonIntrinsics_.valid || !jsonCameraSet_) {
    LOG(INFO) << "[RealtimePublisher] Intrinsics not ready; skipping attach.";
    return;
  }
  if(!realtimePublisher_) {
    realtimePublisher_ =
        std::make_shared<RealtimePublisher>(realtimeHost_, realtimePort_);
    LOG(INFO) << "[RealtimePublisher] Instance created.";
  }

  const uint32_t width = static_cast<uint32_t>(
      std::max(0, jsonIntrinsics_.width));
  const uint32_t height = static_cast<uint32_t>(
      std::max(0, jsonIntrinsics_.height));

  realtimePublisher_->setCameraIntrinsics(
      jsonIntrinsics_.fx, jsonIntrinsics_.fy, jsonIntrinsics_.cx,
      jsonIntrinsics_.cy, width, height, depthMinRange_, depthMaxRange_,
      static_cast<uint32_t>(jsonCameraIdx_));
  LOG(INFO) << "[RealtimePublisher] Intrinsics pushed (w=" << width
            << ", h=" << height << ", camIdx=" << jsonCameraIdx_ << ")";
}

void okvis::TrajectoryOutput::pushStreamImage(uint64_t key,
                                              const cv::Mat& image) {
  if(image.empty()) {
    LOG(WARNING) << "[RealtimePublisher] Attempted to push empty stream image.";
    return;
  }
  std::lock_guard<std::mutex> lock(streamImageMutex_);
  streamImagesByStamp_[key] = image.clone();
  while(streamImagesByStamp_.size() > streamImageBufferSize_) {
    streamImagesByStamp_.erase(streamImagesByStamp_.begin());
  }
  LOG(INFO) << "[RealtimePublisher] Buffered RGB ts=" << key
            << " (buffer=" << streamImagesByStamp_.size() << ")";
}

bool okvis::TrajectoryOutput::popStreamImage(uint64_t key, cv::Mat& image) {
  std::lock_guard<std::mutex> lock(streamImageMutex_);
  if(streamImagesByStamp_.empty()) {
    LOG(WARNING) << "[RealtimePublisher] popStreamImage called with empty buffer.";
    return false;
  }
  auto it = streamImagesByStamp_.find(key);
  if(it == streamImagesByStamp_.end()) {
    LOG(WARNING) << "[RealtimePublisher] Exact timestamp not found; using oldest.";
    it = streamImagesByStamp_.begin();
  }
  image = it->second.clone();
  streamImagesByStamp_.erase(it);
  return !image.empty();
}

void okvis::TrajectoryOutput::submitRealtimeFrame(
    const okvis::State& state,
    int frameIdx,
    const std::string& imageName,
    const cv::Mat& image,
    const std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d>>& pointsWorld,
    const std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d>>& depthPointsWorld) {
  if(!realtimePublisher_ || !realtimePublisher_->isReady()) {
    LOG(WARNING) << "[RealtimePublisher] submitRealtimeFrame skipped (publisher not ready).";
    return;
  }
  if(image.empty() || !jsonCameraSet_) {
    LOG(WARNING) << "[RealtimePublisher] submitRealtimeFrame missing image or intrinsics.";
    return;
  }
  okvis::kinematics::Transformation T_WC = state.T_WS * jsonT_SC_;
  okvis::kinematics::Transformation T_CW = T_WC.inverse();
  Eigen::Matrix4d T = T_CW.T();
  realtimePublisher_->submitFrame(
      toKey(state.timestamp),
      static_cast<uint32_t>(std::max(0, frameIdx)),
      T,
      image,
      pointsWorld,
      depthPointsWorld,
      imageName);
  // LOG(INFO) << "[RealtimePublisher] Submitted frame idx=" << frameIdx
  //           << " ts=" << toKey(state.timestamp)
  //           << " pts=" << pointsWorld.size()
  //           << " depth_pts=" << depthPointsWorld.size();
}
