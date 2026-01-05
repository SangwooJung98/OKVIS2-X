#include "okvis/RealtimePublisher.hpp"

#include <arpa/inet.h>
#include <glog/logging.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstring>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace okvis {

namespace {

#pragma pack(push, 1)
struct HandshakePayloadPacked {
  uint32_t width;
  uint32_t height;
  double fx;
  double fy;
  double cx;
  double cy;
  double depthMin;
  double depthMax;
  uint32_t cameraIndex;
};
#pragma pack(pop)
static_assert(sizeof(HandshakePayloadPacked) ==
                  2 * sizeof(uint32_t) + 6 * sizeof(double) + sizeof(uint32_t),
              "HandshakePayloadPacked has unexpected padding");

#pragma pack(push, 1)
struct FrameFixedPacked {
  uint64_t timestamp;
  uint32_t frameIdx;
  uint32_t rows;
  uint32_t cols;
  uint32_t channels;
  uint32_t imageBytes;
  uint32_t pointCount;
  uint32_t nameLength;
  double T_CW[16];
};
#pragma pack(pop)
static_assert(sizeof(FrameFixedPacked) ==
                  sizeof(uint64_t) + 7 * sizeof(uint32_t) + 16 * sizeof(double),
              "FrameFixedPacked has unexpected padding");

cv::Mat ensureBgr(const cv::Mat& image) {
  if (image.empty()) {
    return cv::Mat();
  }
  if (image.type() == CV_8UC3) {
    return image.clone();
  }
  cv::Mat tmp;
  if (image.channels() == 1) {
    cv::cvtColor(image, tmp, cv::COLOR_GRAY2BGR);
  } else if (image.channels() == 4) {
    cv::cvtColor(image, tmp, cv::COLOR_BGRA2BGR);
  } else {
    cv::Mat converted;
    image.convertTo(converted, CV_8UC3, 255.0);
    cv::cvtColor(converted, tmp, cv::COLOR_RGB2BGR);
  }
  return tmp;
}

void appendBytes(std::vector<uint8_t>& buffer, const void* data, size_t length) {
  const auto* ptr = reinterpret_cast<const uint8_t*>(data);
  buffer.insert(buffer.end(), ptr, ptr + length);
}

}  // namespace

RealtimePublisher::RealtimePublisher(const std::string& host, uint16_t port)
    : host_(host), port_(port) {
  running_.store(true);
  worker_ = std::thread(&RealtimePublisher::run, this);
  LOG(INFO) << "[RealtimePublisher] Streaming enabled towards " << host_ << ":"
            << port_;
  LOG(INFO) << "[RealtimePublisher] Worker thread launched.";
}

RealtimePublisher::~RealtimePublisher() {
  running_.store(false);
  queueCv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
  closeSocket();
}

void RealtimePublisher::setCameraIntrinsics(double fx, double fy, double cx,
                                            double cy, uint32_t width,
                                            uint32_t height, double depthMin,
                                            double depthMax,
                                            uint32_t cameraIndex) {
  std::lock_guard<std::mutex> lock(handshakeMutex_);
  fx_ = fx;
  fy_ = fy;
  cx_ = cx;
  cy_ = cy;
  width_ = width;
  height_ = height;
  depthMin_ = depthMin;
  depthMax_ = depthMax;
  cameraIndex_ = cameraIndex;
  intrinsicsSet_ = true;
  handshakeDirty_ = true;
}

bool RealtimePublisher::isReady() const { return intrinsicsSet_; }

void RealtimePublisher::submitFrame(
    uint64_t timestampNs, uint32_t frameIdx, const Eigen::Matrix4d& T_CW,
    const cv::Mat& imageBgr,
    const std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d>>& pointsWorld,
    const std::string& imageName) {
  if (!running_.load() || !isReady()) {
    return;
  }

  cv::Mat bgr = ensureBgr(imageBgr);
  if (bgr.empty()) {
    LOG_EVERY_N(WARNING, 200)
        << "[RealtimePublisher] Received empty image for streaming.";
    return;
  }

  auto packet =
      makeFramePacket(timestampNs, frameIdx, T_CW, bgr, pointsWorld, imageName);
  if (packet.empty()) {
    return;
  }
  enqueuePacket(std::move(packet));
}

void RealtimePublisher::run() {
  while (running_.load()) {
    std::vector<uint8_t> packet;
    {
      std::unique_lock<std::mutex> lock(queueMutex_);
      queueCv_.wait(lock, [&] {
        return !packetQueue_.empty() || !running_.load();
      });
      if (!running_.load()) {
        break;
      }
      packet = std::move(packetQueue_.front());
      packetQueue_.pop_front();
    }

    LOG(INFO) << "[RealtimePublisher] Dequeued packet (queue="
              << packetQueue_.size() << ")";
    bool sent = false;
    while (running_.load() && !sent) {
      if (!ensureConnected()) {
        LOG(WARNING) << "[RealtimePublisher] Connection unavailable; retrying.";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        continue;
      }

      if (!handshakeSentForConnection_ || handshakeDirty_) {
        handshakeDirty_ = false;
        auto handshake = makeHandshakePacket();
        if (handshake.empty()) {
          LOG_EVERY_N(WARNING, 200)
              << "[RealtimePublisher] Handshake not ready; drop frame.";
          break;
        }
        if (!sendBuffer(handshake)) {
          LOG(WARNING) << "[RealtimePublisher] Failed to send handshake.";
          std::this_thread::sleep_for(std::chrono::milliseconds(250));
          continue;
        }
        handshakeSentForConnection_ = true;
        LOG(INFO) << "[RealtimePublisher] Handshake sent.";
      }

      if (sendBuffer(packet)) {
        sent = true;
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
      }
    }
  }
}

bool RealtimePublisher::ensureConnected() {
  if (socketFd_ >= 0) {
    return true;
  }

  struct addrinfo hints {};
  std::memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  struct addrinfo* result = nullptr;
  const std::string portString = std::to_string(port_);
  const int ret =
      ::getaddrinfo(host_.c_str(), portString.c_str(), &hints, &result);
  if (ret != 0) {
    LOG_EVERY_N(ERROR, 100)
        << "[RealtimePublisher] getaddrinfo failed: " << gai_strerror(ret);
    return false;
  }

  for (auto* rp = result; rp != nullptr; rp = rp->ai_next) {
    int fd = ::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (fd < 0) {
      LOG(WARNING) << "[RealtimePublisher] socket() failed: "
                   << std::strerror(errno);
      continue;
    }
    if (::connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) {
      socketFd_ = fd;
      handshakeSentForConnection_ = false;
      LOG(INFO) << "[RealtimePublisher] Connected to " << host_ << ":"
                << port_;
      break;
    }
    ::close(fd);
    LOG(WARNING) << "[RealtimePublisher] connect() failed: "
                 << std::strerror(errno);
  }

  ::freeaddrinfo(result);
  return socketFd_ >= 0;
}

bool RealtimePublisher::sendBuffer(const std::vector<uint8_t>& buffer) {
  if (socketFd_ < 0) {
    return false;
  }
  size_t totalSent = 0;
  while (totalSent < buffer.size()) {
    const ssize_t sent =
        ::send(socketFd_, buffer.data() + totalSent, buffer.size() - totalSent,
               MSG_NOSIGNAL);
    if (sent < 0) {
      LOG_EVERY_N(WARNING, 200)
          << "[RealtimePublisher] send failed, reconnecting.";
      closeSocket();
      return false;
    }
    totalSent += static_cast<size_t>(sent);
  }
  return true;
}

void RealtimePublisher::closeSocket() {
  if (socketFd_ >= 0) {
    ::close(socketFd_);
    socketFd_ = -1;
  }
  handshakeSentForConnection_ = false;
}

std::vector<uint8_t> RealtimePublisher::makeHandshakePacket() {
  std::lock_guard<std::mutex> lock(handshakeMutex_);
  if (!intrinsicsSet_) {
    return {};
  }

  HandshakePayloadPacked payload{width_, height_, fx_,       fy_,      cx_,
                                 cy_,    depthMin_, depthMax_, cameraIndex_};

  std::vector<uint8_t> buffer(sizeof(PacketHeader));
  buffer.reserve(sizeof(PacketHeader) + sizeof(payload));
  appendBytes(buffer, &payload, sizeof(payload));

  PacketHeader header{kMagic, kVersion, kTypeHandshake,
                      static_cast<uint32_t>(buffer.size() -
                                            sizeof(PacketHeader))};
  std::memcpy(buffer.data(), &header, sizeof(header));
  return buffer;
}

std::vector<uint8_t> RealtimePublisher::makeFramePacket(
    uint64_t timestampNs, uint32_t frameIdx, const Eigen::Matrix4d& T_CW,
    const cv::Mat& imageBgr,
    const std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d>>& pointsWorld,
    const std::string& imageName) {
  std::vector<uint8_t> encoded;
  if (!cv::imencode(".png", imageBgr, encoded)) {
    LOG_EVERY_N(WARNING, 200)
        << "[RealtimePublisher] Failed to encode PNG for streaming.";
    return {};
  }

  const uint32_t rows = static_cast<uint32_t>(imageBgr.rows);
  const uint32_t cols = static_cast<uint32_t>(imageBgr.cols);
  const uint32_t channels = static_cast<uint32_t>(imageBgr.channels());
  const uint32_t imageBytes = static_cast<uint32_t>(encoded.size());
  const uint32_t pointCount = static_cast<uint32_t>(pointsWorld.size());
  const uint32_t nameLength = static_cast<uint32_t>(imageName.size());

  std::vector<float> pointsFlat;
  pointsFlat.reserve(pointsWorld.size() * 3);
  for (const auto& pt : pointsWorld) {
    pointsFlat.push_back(static_cast<float>(pt[0]));
    pointsFlat.push_back(static_cast<float>(pt[1]));
    pointsFlat.push_back(static_cast<float>(pt[2]));
  }

  FrameFixedPacked fixed{};
  fixed.timestamp = timestampNs;
  fixed.frameIdx = frameIdx;
  fixed.rows = rows;
  fixed.cols = cols;
  fixed.channels = channels;
  fixed.imageBytes = imageBytes;
  fixed.pointCount = pointCount;
  fixed.nameLength = nameLength;

  Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(fixed.T_CW) =
      T_CW;

  std::vector<uint8_t> buffer(sizeof(PacketHeader));
  buffer.reserve(sizeof(PacketHeader) + sizeof(fixed) + nameLength +
                 imageBytes + pointsFlat.size() * sizeof(float));
  appendBytes(buffer, &fixed, sizeof(fixed));
  if (nameLength > 0) {
    appendBytes(buffer, imageName.data(), nameLength);
  }
  if (!encoded.empty()) {
    appendBytes(buffer, encoded.data(), encoded.size());
  }
  if (!pointsFlat.empty()) {
    appendBytes(buffer, pointsFlat.data(),
                pointsFlat.size() * sizeof(float));
  }

  PacketHeader header{kMagic, kVersion, kTypeFrame,
                      static_cast<uint32_t>(buffer.size() -
                                            sizeof(PacketHeader))};
  std::memcpy(buffer.data(), &header, sizeof(header));
  return buffer;
}

void RealtimePublisher::enqueuePacket(std::vector<uint8_t>&& packet) {
  {
    std::lock_guard<std::mutex> lock(queueMutex_);
    constexpr size_t kMaxQueue = 256;
    if (packetQueue_.size() >= kMaxQueue) {
      packetQueue_.pop_front();
      LOG_EVERY_N(WARNING, 100)
          << "[RealtimePublisher] Queue full, dropping oldest frame.";
    }
    packetQueue_.push_back(std::move(packet));
  }
  queueCv_.notify_one();
}

}  // namespace okvis


