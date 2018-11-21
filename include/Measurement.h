#ifndef MEASUREMENT_H
#define MEASUREMENT_H

#include <Eigen/Eigen>

namespace uav_localize
{
  class Measurement
  {
  public:
    enum source_t
    {
      depth_detection,
      rgb_tracking
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
