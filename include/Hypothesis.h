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
        : lkf(n, m, p, A, B, R, Q, P), id(id), m_n_corrections(0), m_last_loglikelihood(0)
      {};

      mrs_lib::Lkf lkf;
      const int32_t id;

      void correction(const Measurement& meas, double meas_loglikelihood)
      {
        if (meas.reliable())
          m_n_corrections++;
        m_last_measurement = meas;
        m_last_loglikelihood = meas_loglikelihood;
        lkf.setMeasurement(meas.position, meas.covariance);
        lkf.doCorrection();
      }

      std::tuple<Eigen::Vector3d, Eigen::Matrix3d> calc_innovation(const Measurement& meas) const
      {
        const Eigen::Matrix3d P = lkf.getP().block<3, 3>(0, 0); // simplify the matrix calculations a bit by ignoring the higher derivation states
        const Eigen::Vector3d inn = meas.position - P*get_position();
        const Eigen::Matrix3d inn_cov = meas.covariance + P*get_position_covariance()*P.transpose();
        return std::tuple(inn, inn_cov);
      }

      inline int get_n_corrections(void) const
      {
        return m_n_corrections;
      }
      
      inline double get_last_loglikelihood() const
      {
        return m_last_loglikelihood;
      }

      inline Measurement get_last_measurement(void) const
      {
        return m_last_measurement;
      }

      inline Eigen::Vector3d get_position() const
      {
        return lkf.getStates().block<3, 1>(0, 0);
      }

      inline Eigen::Matrix3d get_position_covariance() const
      {
        return lkf.getCovariance().block<3, 3>(0, 0);
      }

    private:
      int64_t m_n_corrections;
      Measurement m_last_measurement;
      double m_last_loglikelihood;
  };
}

#endif // HYPOTHESIS_H
