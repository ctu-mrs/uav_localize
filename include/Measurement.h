#ifndef MEASUREMENT_H
#define MEASUREMENT_H

#include <Eigen/Eigen>
#include "uav_localize/LocalizationHypothesis.h"

namespace uav_localize
{
  class Measurement
  {
  public:
    enum source_t
    {
      depth_detection = uav_localize::LocalizationHypothesis::SOURCE_DEPTH_DETECTION,
      rgb_tracking    = uav_localize::LocalizationHypothesis::SOURCE_RGB_TRACKING,
      lkf_prediction  = uav_localize::LocalizationHypothesis::SOURCE_LKF_PREDICTION
    };

  public:
    bool reliable() const
    {
      return source == source_t::depth_detection;
    }

    std::string source_name() const
    {
      std::string ret;
      switch (source)
      {
        case depth_detection:
          ret = "uav_detect::Detections";
          break;
        case rgb_tracking:
          ret = "uav_track::Trackings";
          break;
        default:
          ret = "unknown";
          break;
      }
      return ret;
    }

  public:
    Eigen::Vector3d position;
    Eigen::Matrix3d covariance;
    source_t source;
    ros::Time stamp;
  };
};

#endif // MEASUREMENT_H
