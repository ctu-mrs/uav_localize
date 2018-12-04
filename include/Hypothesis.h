#ifndef HYPOTHESIS_H
#define HYPOTHESIS_H

#include "main.h"
#include "Measurement.h"
#include "mrs_lib/Lkf.h"

namespace uav_localize
{
  class Hypothesis
  {
    public:
      Hypothesis(const int id,
                const int n, const int m, const int p,
                const Eigen::MatrixXd A, const Eigen::MatrixXd B,
                const Eigen::MatrixXd R, const Eigen::MatrixXd Q,
                const Eigen::MatrixXd P)
        : lkf(n, m, p, A, B, R, Q, P), id(id), m_n_corrections(0)
      {};

      mrs_lib::Lkf lkf;
      const int32_t id;

      void correction(const Measurement& meas)
      {
        if (meas.reliable())
          m_n_corrections++;
        m_last_measurement = meas;
        lkf.setMeasurement(meas.position, meas.covariance);
        lkf.doCorrection();
      }

      int get_n_corrections(void) const
      {
        return m_n_corrections;
      }

      Measurement get_last_measurement(void) const
      {
        return m_last_measurement;
      }

      Eigen::Vector3d get_position() const
      {
        return lkf.getStates().block<3, 1>(0, 0);
      }

      Eigen::Matrix3d get_position_covariance() const
      {
        return lkf.getCovariance().block<3, 3>(0, 0);
      }

    private:
      int64_t m_n_corrections;
      Measurement m_last_measurement;
  };
}

#endif // HYPOTHESIS_H
