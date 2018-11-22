#ifndef MEASUREMENT_H
#define MEASUREMENT_H

#include <Eigen/Eigen>
#include "uav_localize/LocalizedUAV.h"

namespace uav_localize
{
  class Measurement
  {
  public:
    enum source_t
    {
      depth_detection = uav_localize::LocalizedUAV::SOURCE_DEPTH_DETECTION,
      rgb_tracking = uav_localize::LocalizedUAV::SOURCE_RGB_TRACKING
    };

  public:
    bool reliable() const
    {
      return source == source_t::depth_detection;
    }

  public:
    Eigen::Vector3d position;
    Eigen::Matrix3d covariance;
    source_t source;
  };
};

#endif // MEASUREMENT_H
