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
 * @file TrajectoryOutput.hpp
 * @brief Header file for the TrajectoryOutput class.
 * @author Stefan Leutenegger
 */

#ifndef OKVIS_TRAJECTORYOUTPUT_HPP
#define OKVIS_TRAJECTORYOUTPUT_HPP

#include "okvis/QueuedTrajectory.hpp"
#include <fstream>
#include <memory>
#include <string>
#include <optional>
#include <unordered_map>
#include <map>
#include <mutex>
#include <filesystem>

#include <Eigen/Core>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include <opencv2/core.hpp>
#pragma GCC diagnostic pop

#include <okvis/assert_macros.hpp>
#include <okvis/threadsafe/ThreadsafeQueue.hpp>
#include <okvis/ViInterface.hpp>
#include <okvis/QueuedTrajectory.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/cameras/NCameraSystem.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/RealtimePublisher.hpp>


namespace okvis {

/**
 * @brief The TrajectoryOutput class: a simple writer of trajectory files and visualiser of
 * pose/trajectory.
 */
class TrajectoryOutput {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  /**
   * @brief Default constructor.
   * @param draw Whether to visualise a top view.
   */
  TrajectoryOutput(bool draw = true);

  /**
   * @brief Constructor with filename to write CSV trajectory.
   * @param filename Write CSV trajectory to this file.
   * @param rpg If true, uses the RPG format, otherwise the EuRoC format.
   * @param draw Whether to visualise a top view.
   */
  TrajectoryOutput(const std::string & filename, bool rpg = false, bool draw = true);

  /// \brief Destructor to flush/close any open files.
  ~TrajectoryOutput();

  /**
   * @brief Set CSV file.
   * @param filename Write CSV trajectory to this file.
   * @param rpg If true, uses the RPG format, otherwise the EuRoC format.
   */
  void setCsvFile(const std::string & filename, bool rpg = false);

  /**
   * @brief Set RGB CSV file.
   * @param filename Write CSV trajectory to this file.
   */
  void setRGBCsvFile(const std::string & filename);

  /**
   * @brief Set JSON file for trajectory (real-time JSONL stream).
   * @param filename Write JSON trajectory to this file.
   */
  void setJsonFile(const std::string & filename);

  /**
   * @brief Configure JSON export with camera intrinsics/extrinsics.
   * @param ncamera   Camera system (for intrinsics and T_SC).
   * @param camIdx    Camera index to export.
   * @param imagePrefix Prefix for image names (e.g., "frame").
   * @param imageExt  File extension (e.g., "png").
   * @param zeroPad   Zero padding for frame numbers.
   * @return true on success.
   */
  bool setJsonCamera(const okvis::cameras::NCameraSystem& ncamera,
                     size_t camIdx = 0,
                     const std::string& imagePrefix = "frame",
                     const std::string& imageExt = "png",
                     int zeroPad = 6);

  /**
   * @brief Configure depth point cloud export behaviour.
   * @param minRangeMeters Minimum accepted depth (in metres).
   * @param maxRangeMeters Maximum accepted depth (in metres).
   * @param depthScaleMetersPerUnit Conversion from stored depth units to metres.
   * @param pixelStride Downsampling stride when sampling pixels.
   */
  void setDepthExportConfig(double minRangeMeters,
                            double maxRangeMeters,
                            double depthScaleMetersPerUnit = 1.0,
                            int pixelStride = 4);

  /**
   * @brief Process the state (write it to trajectory file and visualise). Set as callback.
   * @param[in] state The current state to process.
   * @param[in] trackingState The additional tracking info to process.
   * @param[in] updatedStates All the states that the estimator updated.
   * @param[in] landmarks All the landmarks that the estimator updated.
   */
  void processState(const State& state, const TrackingState & trackingState,
                    std::shared_ptr<const AlignedMap<StateId, State>> updatedStates,
                    std::shared_ptr<const okvis::MapPointVector> landmarks);

  /**
   * \brief          Add an IMU measurement.
   * \param stamp    The measurement timestamp.
   * \param alpha    The acceleration measured at this time.
   * \param omega    The angular velocity measured at this time.
   * \return True on success.
   */
  bool addImuMeasurement(const okvis::Time& stamp,
                         const Eigen::Vector3d& alpha,
                         const Eigen::Vector3d& omega);

  /**
   * @brief Compute state at RGB image perception and store pose to RGB image.
   * @param timestamp RGB image recording timestamp.
   * @param image RGB image.
   */
  bool processRGBImage(const okvis::Time &, const cv::Mat &);

  /**
   * @brief Buffer a depth image for later sparse point cloud export.
   * @param timestamp Depth image timestamp.
   * @param camIdx Camera index the depth image belongs to.
   * @param depthImage Depth map (float32, metres preferred).
   */
  bool processDepthImage(const okvis::Time& timestamp,
                         size_t camIdx,
                         const cv::Mat& depthImage);

  /// \brief Draw the top view now.
  /// \param outImg The output image to draw into.
  void drawTopView(cv::Mat & outImg);

private:
  // helpers for image naming and timestamp mapping
  std::string makeJsonImageName();
  std::string nextImageName();
  uint64_t toKey(const okvis::Time& t) const;
  bool lookupImageName(const okvis::Time& t, std::string& name);
  int imageIndexFromName(const std::string& name) const;
  void appendImageList(const std::string& name, int idx);
  void collectLandmarkPoints(
      const okvis::State& state,
      std::shared_ptr<const okvis::MapPointVector> landmarks,
      std::vector<Eigen::Vector3d,
                  Eigen::aligned_allocator<Eigen::Vector3d>>& pointsWorld) const;
  bool computeDepthPointCloudWorld(
      const okvis::State& state,
      const cv::Mat& depthImage,
      std::vector<Eigen::Vector3d,
                  Eigen::aligned_allocator<Eigen::Vector3d>>& pointsWorld) const;
  bool writePointCloud(
      const okvis::State& state,
      const std::vector<Eigen::Vector3d,
                        Eigen::aligned_allocator<Eigen::Vector3d>>& pointsWorld,
      const std::string& imageName);
  bool writeDepthPointCloud(
      const okvis::State& state,
      const std::vector<Eigen::Vector3d,
                        Eigen::aligned_allocator<Eigen::Vector3d>>& pointsWorld,
      const std::string& imageName);
  bool writeStateToJson(const okvis::State& state,
                        const std::string& imageName);
  void configureRealtimePublisherFromEnv();
  void tryAttachRealtimePublisher();
  void pushStreamImage(uint64_t key, const cv::Mat& image);
  bool popStreamImage(uint64_t key, cv::Mat& image);
  void submitRealtimeFrame(
      const okvis::State& state,
      int frameIdx,
      const std::string& imageName,
      const cv::Mat& image,
      const std::vector<Eigen::Vector3d,
                        Eigen::aligned_allocator<Eigen::Vector3d>>& pointsWorld,
      const std::vector<Eigen::Vector3d,
                        Eigen::aligned_allocator<Eigen::Vector3d>>& depthPointsWorld);

  std::fstream csvFile_; ///< The CSV file.
  std::fstream rgbCsvFile_;  ///< The RGB CSV file.
  std::fstream jsonFile_; ///< The JSON file (newline-delimited objects).
  std::ofstream imageListFile_; ///< image_list.txt
  std::string jsonDir_; ///< Directory of the JSON file.
  std::string imagesDir_; ///< Directory to save images.
  std::string pointCloudDir_; ///< Directory to save per-frame sparse point clouds.
  // std::string pointCloudPlyDir_; ///< Directory to save per-frame sparse point clouds as PLY.
  std::string imageListPath_; ///< image_list.txt path
  bool imageListEnabled_ = false;
  bool pointCloudEnabled_ = false; ///< Whether point cloud saving is enabled.
  // bool pointCloudPlyEnabled_ = false; ///< Whether point cloud PLY saving is enabled.
  std::unordered_map<uint64_t, std::string> imageNameByStamp_; ///< timestamp->image name
  std::mutex imageMutex_; ///< protect imageNameByStamp_ and counter
  bool jsonEnabled_ = false; ///< Whether JSON writing is enabled.
  bool firstJsonEntry_ = true; ///< Tracks comma placement for JSON array.
  bool jsonCameraSet_ = false; ///< Whether camera intrinsics/extrinsics are set.
  size_t jsonFrameCounter_ = 0; ///< Frame index for image naming.
  std::string jsonImagePrefix_ = "frame"; ///< Prefix for image names.
  std::string jsonImageExt_ = "png"; ///< Extension for image names.
  int jsonImageZeroPad_ = 6; ///< Zero padding for frame numbers.
  size_t jsonCameraIdx_ = 0; ///< Camera index configured for JSON export.
  okvis::kinematics::Transformation jsonT_SC_; ///< Camera-to-sensor transform.
  struct JsonIntrinsics {
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    int width = 0;
    int height = 0;
    double focal = 0.0;
    bool valid = false;
  } jsonIntrinsics_;
  bool rpg_ = false; ///< Whether to use the RPG format (instead of EuRoC)

  bool draw_ = true; ///< Whether to draw a top view.

  /// \brief Helper for graph states.
  typedef std::tuple<
      State, TrackingState, std::shared_ptr<const AlignedMap<StateId, State>>,
      std::shared_ptr<const okvis::MapPointVector>> GraphStates;

  threadsafe::Queue<std::shared_ptr<GraphStates>> states_; ///< Graph states.
  Trajectory trajectory_; /// Trajectory object to be queried at any time;
  threadsafe::Queue<ImuMeasurement> imuMeasurements_; ///< Graph states.

  cv::Mat _image; ///< Image.
  int _imageSize = 500; ///< Pixel size of the image.
  std::vector<cv::Point2d> _path; ///< Path in 2d.
  std::vector<cv::Point2d> _gps; ///< GPS measurements in 2d.
  std::vector<double> _heights; ///< Heights on the path.
  double _scale = 1.0; ///< Scale of the visualisation.
  double _min_x = -0.5; ///< Minimum x coordinate the visualisation.
  double _min_y = -0.5; ///< Minimum y coordinate the visualisation.
  double _min_z = -0.5; ///< Minimum z coordinate the visualisation.
  double _max_x = 0.5; ///< Maximum x coordinate the visualisation.
  double _max_y = 0.5; ///< Maximum y coordinate the visualisation.
  double _max_z = 0.5; ///< Maximum z coordinate the visualisation.
  const double _frameScale = 0.2;  ///< [m]

  okvis::QueuedTrajectory<cv::Mat> rgbTrajectory_;  ///< RGB trajectory with Queue.

  // depth-based sparse point cloud data
  std::string pointCloudDepthDir_;
  bool depthPointCloudDirReady_ = false;
  bool depthPointCloudEnabled_ = false;
  size_t depthCameraIdx_ = 0;
  double depthMinRange_ = 0.2;
  double depthMaxRange_ = 15.0;
  double depthScaleMetersPerUnit_ = 1.0;
  int depthPixelStride_ = 4;
  size_t depthImageBufferSize_ = 20;
  std::map<uint64_t, cv::Mat> depthImagesByStamp_;
  std::mutex depthMutex_;

  // realtime streaming support
  bool realtimeEnvChecked_ = false;
  bool realtimeStreamingConfigured_ = false;
  std::string realtimeHost_;
  uint16_t realtimePort_ = 0;
  size_t streamImageBufferSize_ = 20;
  std::map<uint64_t, cv::Mat> streamImagesByStamp_;
  std::mutex streamImageMutex_;
  std::shared_ptr<RealtimePublisher> realtimePublisher_;

  /// \brief Convert metric coordinates to pixels.
  /// \param pointInMeters Point in [m].
  /// \return Point in [pixels].
  cv::Point2d convertToImageCoordinates(const cv::Point2d & pointInMeters) const;

  /// \brief Draw the visualisation (causal, gray).
  void drawPath();

  /// \brief Draw the GPS into the top view.
  void drawGps();

  /// \brief Draw the visualisation (non-causal, coloured by height).
  void drawPathNoncausal();

  /// \brief Create a trajectory csv file and write its header.
  static bool createCsvFile(const std::string &filename, std::fstream& stream,
                            const bool rpg = false);

  /// \brief Write state into csv file.
  static bool writeStateToCsv(std::fstream& csvFile, const okvis::State& state,
                              const bool rpg = false);

  /// \brief Write state into json file (trajectory camera-centric format).
  bool writeStateToJson(const okvis::State& state);

  okvis::kinematics::Transformation _T_AW; ///< Transf. betw. this world coord. & another agent.

};

}

#endif // OKVIS_TRAJECTORYOUTPUT_HPP
