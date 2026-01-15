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
 * @file okvis2x_app_synchronous.cpp
 * @brief A synchronous app without neural networks
 * @author Simon Boche
 * @author Jaehyung Jung
 * @author Sebastian Barbas Laina
 */

#include <iostream>
#include <stdlib.h>
#include <memory>
#include <functional>
#include <utility>

#include <boost/filesystem.hpp>
#include <Eigen/Core>
#include <opencv2/highgui/highgui.hpp>

#include <okvis/XDatasetReader.hpp>
#include <okvis/ViParametersReader.hpp>
#include <okvis/ThreadedSlam.hpp>
#include <okvis/TrajectoryOutput.hpp>
#include <okvis/SubmappingInterface.hpp>


int main(int argc, char **argv)
{

  // argv[1] --> okvis config
  // argv[2] --> mapping config file
  // argv[3] --> dataset path
  // argv[4] --> (optional) save directory

  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 0;  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
  FLAGS_colorlogtostderr = 1;
  FLAGS_minloglevel = 0;

  // read configuration file
  std::string configFilename(argv[1]);
  std::string seConfigFilename(argv[2]);
  okvis::ViParametersReader viParametersReader(configFilename);
  okvis::ViParameters parameters;
  viParametersReader.getParameters(parameters);
  se::SubMapConfig submapConfig(seConfigFilename);
  okvis::SupereightMapType::Config mapConfig;
  mapConfig.readYaml(seConfigFilename);
  okvis::SupereightMapType::DataType::Config dataConfig;
  dataConfig.readYaml(seConfigFilename);
  const bool isDepth = parameters.lidar ? false : true;
  const bool isLidar = !isDepth;
  const bool isSubmapping = parameters.output.enable_submapping;
  bool hasDepthCamera = false;
  for(size_t camIdx = 0; camIdx < parameters.nCameraSystem.numCameras(); ++camIdx) {
    if(parameters.nCameraSystem.isCameraConfigured(camIdx) &&
       parameters.nCameraSystem.cameraType(camIdx).depthType.isDepthCamera) {
      hasDepthCamera = true;
      break;
    }
  }
  const bool useDepthForSlam = isSubmapping && isDepth;
  const bool streamDepthForOutput = hasDepthCamera;

  // dataset reader
  std::string path(argv[3]);
  std::shared_ptr<okvis::XDatasetReader> datasetReader;
  okvis::Duration deltaT(0.0); // time tolerance to callbacks
  if (isSubmapping) {
    datasetReader.reset(new okvis::XDatasetReader(path, deltaT, parameters, isLidar, false, isDepth));
  }
  else {
    datasetReader.reset(new okvis::XDatasetReader(path, deltaT, parameters, false, false, streamDepthForOutput));
  }
  if(streamDepthForOutput && !useDepthForSlam) {
    LOG(INFO) << "[App] Depth streaming enabled for output only (not used in SLAM).";
  }
  

  // also check DBoW2 vocabulary
  boost::filesystem::path executable(argv[0]);
  std::string dBowVocDir = executable.remove_filename().string();
  std::ifstream infile(dBowVocDir+"/small_voc.yml.gz");
  if(!infile.good()) {
    LOG(ERROR) << "The dbow file is not valid: " 
      << dBowVocDir << "/small_voc.yml.gz";
    return EXIT_FAILURE;
  }

  // Output Folder
  std::string savePath;
  if(argc == 4) {
    savePath = path;
  }
  else if(argc == 5){
    savePath = std::string(argv[4]);
    submapConfig.resultsDirectory = savePath;
  }
  else {
    LOG(ERROR) << "Usesage: ./" << argv[0] << 
      "[config-okvis2.yaml] [config-se2.yaml] [input-directory] (optional)[output-directory]";
    return EXIT_FAILURE;
  }

  // Setup OKVIS estimator
  std::shared_ptr<okvis::ThreadedSlam> estimator(nullptr);
  estimator.reset(new okvis::ThreadedSlam(parameters, dBowVocDir, submapConfig));
  estimator->setBlocking(true);
  LOG(INFO) << "[App] ThreadedSlam instantiated and set to blocking mode.";

  // Setup the submapping interface
  std::shared_ptr<okvis::SubmappingInterface> seInterface(nullptr);
  if (isSubmapping) {
    seInterface.reset(new okvis::SubmappingInterface(mapConfig, dataConfig, submapConfig, parameters));
    seInterface->setT_BS(parameters.imu.T_BS);
    seInterface->setBlocking(true);
  }

  // Select a proper output name
  std::string mode = "slam";
  if(!parameters.estimator.do_loop_closures) {
    mode = "vio";
  }
  if(parameters.camera.online_calibration.do_extrinsics) {
    mode = mode+"-calib";
  }
  estimator->setFinalTrajectoryCsvFile(savePath+"/okvis2-" + mode + "-final_trajectory.csv");
  estimator->setMapCsvFile(savePath+"/okvis2-" + mode + "-final_map.csv");

  // Determine which camera to use for exports (prefer a depth camera if available)
  size_t exportCamIdx = 0;
  for(size_t camIdx = 0; camIdx < parameters.nCameraSystem.numCameras(); ++camIdx) {
    if(parameters.nCameraSystem.isCameraConfigured(camIdx) &&
       parameters.nCameraSystem.cameraType(camIdx).depthType.isDepthCamera) {
      exportCamIdx = camIdx;
      break;
    }
  }

  // Setup the trajectory output writer
  std::shared_ptr<okvis::TrajectoryOutput> writer;
  LOG(INFO) << "[App] Creating TrajectoryOutput (path=" << savePath << ")";
  writer.reset(new okvis::TrajectoryOutput(
      savePath+"/okvis2-" + mode + "_trajectory.csv",
      false,
      parameters.output.display_topview));
  LOG(INFO) << "[App] TrajectoryOutput CSV ready.";
  LOG(INFO) << "[App] Setting JSON file...";
  writer->setJsonFile(savePath+"/okvis2-" + mode + "_trajectory.json");
  LOG(INFO) << "[App] Setting JSON camera...";
  writer->setJsonCamera(parameters.nCameraSystem, exportCamIdx, "frame", "png", 6);
  LOG(INFO) << "[App] TrajectoryOutput initialised (mode=" << mode
            << ", exportCamIdx=" << exportCamIdx << ")";
  
  // save depth image (sw)
  // writer->setDepthExportConfig(0.4, 10.0, 0.001, 50); // for real-sense
  writer->setDepthExportConfig(0.4, 10.0,
                               parameters.output.depth_export_scale,
                               parameters.output.depth_export_stride);

  if (isSubmapping) {
    // Set callbacks in the estimator
    estimator->setOptimisedGraphCallback([&] (const okvis::State& _1, const okvis::TrackingState& _2,
                                                    std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>> _3,
                                                    std::shared_ptr<const okvis::MapPointVector> _4){
      writer->processState(_1,_2,_3,_4);
      seInterface->stateUpdateCallback(_1,_2,_3);
    });

    // Set a callback in the submapping interface
    if(submapConfig.useMap2MapFactors || submapConfig.useMap2LiveFactors) {
      seInterface->setAlignCallback(std::bind(&okvis::ThreadedSlam::addSubmapAlignmentConstraints, estimator,
                                                std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                                                std::placeholders::_4, std::placeholders::_5, std::placeholders::_6));
    }
  }
  else {
    estimator->setOptimisedGraphCallback([&] (const okvis::State& _1, const okvis::TrackingState& _2,
                                                    std::shared_ptr<const okvis::AlignedMap<okvis::StateId, okvis::State>> _3,
                                                    std::shared_ptr<const okvis::MapPointVector> _4){
      writer->processState(_1,_2,_3,_4);
    });
  }

  // connect reader to estimator
  datasetReader->setImuCallback(
          std::bind(&okvis::ThreadedSlam::addImuMeasurement, estimator,
                    std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
  datasetReader->setImagesCallback(
          [writer, estimator, datasetReader, exportCamIdx, useDepthForSlam](const okvis::Time& ts,
                              const std::map<size_t, cv::Mat>& images,
                              const std::map<size_t, cv::Mat>& depthImages){
            // feed estimator (optionally skip depth)
            if(useDepthForSlam) {
              estimator->addImages(ts, images, depthImages);
            } else {
              estimator->addImages(ts, images, std::map<size_t, cv::Mat>{});
            }
            // save a color image for exportCamIdx only
            if(!images.empty()) {
              const auto it = images.find(exportCamIdx);
              if(it != images.end() && !it->second.empty()) {
                // Try to reload original color image by filename to avoid grayscale feed
                std::string fname;
                cv::Mat colorImg;
                if(datasetReader->getLastImageFilename(exportCamIdx, fname)) {
                  colorImg = cv::imread(fname, cv::IMREAD_COLOR);
                }
                const cv::Mat* chosen = nullptr;
                if(!colorImg.empty()) {
                  chosen = &colorImg;
                } else {
                  chosen = &it->second;
                }
                if(chosen && !chosen->empty()) {
                  writer->processRGBImage(ts, *chosen);
                }
              }
            }

            // save depth image (sw)
            auto itDepth = depthImages.find(exportCamIdx);
            if(itDepth == depthImages.end() && !depthImages.empty()) {
              itDepth = depthImages.begin();
              LOG_EVERY_N(INFO, 200)
                  << "[DepthPC] exportCamIdx=" << exportCamIdx
                  << " not found in depthImages; using camIdx="
                  << itDepth->first << " instead.";
            }
            if(itDepth != depthImages.end() && !itDepth->second.empty()) {
              writer->processDepthImage(ts, itDepth->first, itDepth->second);
            }
            
            return true;
          });

  if(parameters.gps){
      if((*parameters.gps).type == "cartesian"){
          datasetReader->setGpsCallback(
                  std::bind(&okvis::ThreadedSlam::addGpsMeasurement, estimator,
                            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
      }
      else if((*parameters.gps).type == "geodetic" || (*parameters.gps).type == "geodetic-leica"){
          datasetReader->setGeodeticGpsCallback(
                  std::bind(&okvis::ThreadedSlam::addGeodeticGpsMeasurement, estimator,
                            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                            std::placeholders::_4, std::placeholders::_5, std::placeholders::_6));
      }
  }
  if (isSubmapping) {
    if (isLidar) {
      datasetReader->setLidarCallback([&] (const okvis::Time& _1, const Eigen::Vector3d& _2){
        bool estimatorAdd = true;
        if(submapConfig.useMap2LiveFactors) {
            estimatorAdd = estimator->addLidarMeasurement(_1, _2);
        }
        bool mapAdd = seInterface->addLidarMeasurement(_1, _2);
        return (estimatorAdd && mapAdd);
      });
    }
    else if (isDepth) {
      datasetReader->setDepthImageCallback([&] (std::map<size_t, std::vector<okvis::CameraMeasurement>>& frames){
        bool estimatorAdd = false;
        bool mapAdd = false;
        for(const auto& cam_idx_frames : frames) {
          for(const auto& cam_measurement : cam_idx_frames.second) {
            if(!cam_measurement.measurement.depthImage.empty() && !cam_measurement.measurement.sigmaImage.empty()) {
              estimatorAdd = estimator->addDepthMeasurement(cam_measurement.timeStamp,
                                                                  cam_measurement.measurement.depthImage,
                                                                  cam_measurement.measurement.sigmaImage);
            } 
            else if(!cam_measurement.measurement.depthImage.empty()) {
              estimatorAdd = estimator->addDepthMeasurement(cam_measurement.timeStamp,
                                                                  cam_measurement.measurement.depthImage);
            }
          }
        }
        mapAdd = seInterface->addDepthMeasurement(frames);
        return (estimatorAdd && mapAdd);    
      });
    }

    // Start submapping interface
    seInterface->start();
  }
  else if(streamDepthForOutput) {
    // Depth-only export when SLAM is not using depth.
    datasetReader->setDepthImageCallback([&] (std::map<size_t, std::vector<okvis::CameraMeasurement>>& frames){
      LOG_FIRST_N(INFO, 5) << "[DepthPC] Depth callback invoked (frames=" << frames.size()
                           << ", exportCamIdx=" << exportCamIdx << ")";
      bool saved = false;
      for(const auto& cam_idx_frames : frames) {
        if(cam_idx_frames.first != exportCamIdx) {
          continue;
        }
        for(const auto& cam_measurement : cam_idx_frames.second) {
          LOG_FIRST_N(INFO, 5) << "[DepthPC] Depth sample ts=" << cam_measurement.timeStamp
                               << " camIdx=" << cam_idx_frames.first
                               << " rows=" << cam_measurement.measurement.depthImage.rows
                               << " cols=" << cam_measurement.measurement.depthImage.cols
                               << " type=" << cam_measurement.measurement.depthImage.type()
                               << " empty=" << cam_measurement.measurement.depthImage.empty();
          if(!cam_measurement.measurement.depthImage.empty()) {
            writer->processDepthImage(cam_measurement.timeStamp,
                                      cam_idx_frames.first,
                                      cam_measurement.measurement.depthImage);
            LOG_FIRST_N(INFO, 5) << "[DepthPC] Queued depth ts=" << cam_measurement.timeStamp
                                 << " camIdx=" << cam_idx_frames.first;
            saved = true;
            break;
          }
        }
        if(saved) {
          break;
        }
      }
      return saved;
    });
  }

  // Start streaming
  if(!datasetReader->startStreaming()) {
      LOG(ERROR) << "Failure with datasetReader streaming.";
      return EXIT_FAILURE;
  }
  LOG(INFO) << "[App] DatasetReader streaming started.";

  // Estimator Loop
  okvis::Time startTime = okvis::Time::now();

  int progress = 0;
  bool datasetreaderFinished = false;
  while(true){

      if(!datasetreaderFinished){
          estimator->processFrame();
          std::map<std::string, cv::Mat> images;
          estimator->display(images);
          for(const auto & image : images) {
            cv::imshow(image.first, image.second);
          }
          cv::Mat topView;
          writer->drawTopView(topView);
          if(!topView.empty()) {
            cv::imshow("OKVIS 2 Top View", topView);
          }
          cv::Mat submapPlot;
          if(isSubmapping && seInterface->publishSubmapTopView(submapPlot)){
            if(!submapPlot.empty()) {
              cv::imshow("Top View Submaps", submapPlot);
            }
          }
          if(!images.empty() || !topView.empty() || !submapPlot.empty()) {
            cv::waitKey(2);
          }
      }

      // check if done
      if(!datasetReader->isStreaming()) {
          datasetreaderFinished = true;
          std::cout << "\r DatasetReader Finished!" << std::endl;

          if(datasetreaderFinished){
              if (isSubmapping) {
                while(!seInterface->finishedIntegrating()){
                  std::this_thread::sleep_for(std::chrono::milliseconds(100));
                  LOG(INFO) << "waiting for integrator to finish!";
                }
              }
              estimator->writeFinalTrajectoryCsv();
              if(parameters.estimator.do_final_ba) {
                LOG(INFO) << "final full BA...";
                estimator->doFinalBa();
                estimator->setFinalTrajectoryCsvFile(savePath+"/okvis2-" + mode + "-final-ba_trajectory.csv");
                estimator->writeFinalTrajectoryCsv();

                // Wait until the other processor finishes again from the finalBA stateUpdate callback.
                if (isSubmapping) {
                  while(!seInterface->finishedIntegrating()){
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    LOG(INFO) << "waiting for integrator to finish!";
                  }
                }
              }
              LOG(INFO) <<"total processing time OKVIS only " << (okvis::Time::now() - startTime) << " s" << std::endl;

              if (isSubmapping) {
                seInterface->setFinished();
                seInterface->printAssociations();
                if (submapConfig.write_mesh_output) {
                  seInterface->saveAllSubmapMeshes();
                }
                cv::Mat finalSubmapPlot;
                if(seInterface->publishSubmapTopView(finalSubmapPlot)){
                    if(!finalSubmapPlot.empty()){
                        cv::imwrite(savePath+"submaps.png", finalSubmapPlot);
                    }
                }
              }
              estimator->saveMap(); // This saves landmarks map
              LOG(INFO) <<"total processing time " << (okvis::Time::now() - startTime) << " s" << std::endl;
              break;
          }
      }

      // display progress
      int newProgress = int(datasetReader->completion()*100.0);
#ifndef DEACTIVATE_TIMERS
      if (newProgress>progress) {
          LOG(INFO) << okvis::timing::Timing::print();
      }
#endif
      if (newProgress>progress) {
          progress = newProgress;
          std::cout << "\rProgress: "
                    << progress << "% "
                    << std::flush;
      }
  }
  return EXIT_SUCCESS;
}