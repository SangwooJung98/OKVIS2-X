/**
 * OKVIS2-X - Realtime publisher for TCP streaming.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

namespace okvis {

  /**
   * @brief Streams per-frame RGB + pose + sparse/depth point cloud data to an
   *        external consumer over TCP in a simple binary protocol.
   */
class RealtimePublisher {
 public:
  RealtimePublisher(const std::string& host, uint16_t port);
  ~RealtimePublisher();

  RealtimePublisher(const RealtimePublisher&) = delete;
  RealtimePublisher& operator=(const RealtimePublisher&) = delete;

  void setCameraIntrinsics(double fx, double fy, double cx, double cy,
                           uint32_t width, uint32_t height,
                           double depthMin, double depthMax,
                           uint32_t cameraIndex);

  bool isReady() const;

  void submitFrame(
      uint64_t timestampNs,
      uint32_t frameIdx,
      const Eigen::Matrix4d& T_CW,
      const cv::Mat& imageBgr,
      const std::vector<Eigen::Vector3d,
                        Eigen::aligned_allocator<Eigen::Vector3d>>& pointsWorld,
      const std::vector<Eigen::Vector3d,
                        Eigen::aligned_allocator<Eigen::Vector3d>>& depthPointsWorld,
      const std::string& imageName);

 private:
  struct PacketHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t type;
    uint32_t payloadSize;
  };

  static constexpr uint32_t kMagic = 0x4F334753;  // "O3GS"
  static constexpr uint16_t kVersion = 2;
  static constexpr uint16_t kTypeHandshake = 1;
  static constexpr uint16_t kTypeFrame = 2;

  void run();
  bool ensureConnected();
  bool sendBuffer(const std::vector<uint8_t>& buffer);
  void closeSocket();
  std::vector<uint8_t> makeHandshakePacket();
  std::vector<uint8_t> makeFramePacket(
      uint64_t timestampNs,
      uint32_t frameIdx,
      const Eigen::Matrix4d& T_CW,
      const cv::Mat& imageBgr,
      const std::vector<Eigen::Vector3d,
                        Eigen::aligned_allocator<Eigen::Vector3d>>& pointsWorld,
      const std::vector<Eigen::Vector3d,
                        Eigen::aligned_allocator<Eigen::Vector3d>>& depthPointsWorld,
      const std::string& imageName);
  void enqueuePacket(std::vector<uint8_t>&& packet);

  std::string host_;
  uint16_t port_;
  int socketFd_ = -1;
  std::thread worker_;
  std::mutex queueMutex_;
  std::condition_variable queueCv_;
  std::deque<std::vector<uint8_t>> packetQueue_;
  std::atomic<bool> running_{false};

  mutable std::mutex handshakeMutex_;
  double fx_ = 0.0;
  double fy_ = 0.0;
  double cx_ = 0.0;
  double cy_ = 0.0;
  double depthMin_ = 0.1;
  double depthMax_ = 50.0;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  uint32_t cameraIndex_ = 0;
  bool intrinsicsSet_ = false;
  bool handshakeDirty_ = false;
  bool handshakeSentForConnection_ = false;
};

}  // namespace okvis


