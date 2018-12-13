#ifndef HYPOTHESIS_H
#define HYPOTHESIS_H

#include "main.h"
#include "Measurement.h"
#include "mrs_lib/Lkf.h"

namespace uav_localize
{
  class Hypothesis
  {
    private:
      /* Definitions of the LKF (consts, typedefs, etc.) //{ */
      static const int c_n_states = 6;
      static const int c_n_inputs = 0;
      static const int c_n_measurements = 3;

      typedef Eigen::Matrix<double, c_n_states, 1> lkf_x_t;
      typedef Eigen::Matrix<double, c_n_inputs, 1> lkf_u_t;
      typedef Eigen::Matrix<double, c_n_measurements, 1> lkf_z_t;

      typedef Eigen::Matrix<double, c_n_states, c_n_states>             lkf_A_t;  // system matrix n*n
      typedef Eigen::Matrix<double, c_n_states, c_n_inputs>             lkf_B_t;  // input matrix n*m
      typedef Eigen::Matrix<double, c_n_measurements, c_n_states>       lkf_P_t;  // measurement mapping p*n
      typedef Eigen::Matrix<double, c_n_states, c_n_states>             lkf_R_t;  // measurement covariance p*p
      typedef Eigen::Matrix<double, c_n_measurements, c_n_measurements> lkf_Q_t;  // process covariance n*n

      inline static lkf_A_t create_A(double dt)
      {
        lkf_A_t A;
        A << 1, 0, 0, dt, 0, 0,
             0, 1, 0, 0, dt, 0,
             0, 0, 1, 0, 0, dt,
             0, 0, 0, 1, 0, 0,
             0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 1;
        return A;
      }

      inline static lkf_P_t create_P()
      {
        lkf_P_t P;
        P << 1, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0;
        return P;
      }

      inline lkf_R_t create_R(double dt, double pos_std, double vel_std) const
      {
        lkf_R_t R = lkf_R_t::Identity();
        R.block<3, 3>(0, 0) *= dt * pos_std;
        R.block<3, 3>(3, 3) *= dt * vel_std;
        return R;
      }
      //}

    public:
      Hypothesis(const int id, const Measurement& init_meas, double init_vel_std)
        : id(id), m_n_corrections(0), m_loglikelihood(0),
          m_lkf(c_n_states, c_n_inputs, c_n_measurements, lkf_A_t(), lkf_B_t(), lkf_R_t(), lkf_Q_t(), create_P())
      {
        m_last_lkf_update = init_meas.stamp;

        // Initialize the LKF using the initialization measurement
        lkf_x_t init_state;
        init_state.block<3, 1>(0, 0) = init_meas.position;
        init_state.block<3, 1>(3, 0) = Eigen::Vector3d::Zero();
        lkf_R_t init_state_cov;
        init_state_cov.setZero();
        init_state_cov.block<3, 3>(0, 0) = init_meas.covariance;
        init_state_cov.block<3, 3>(3, 3) = init_vel_std * Eigen::Matrix3d::Identity();

        m_lkf.setStates(init_state);
        m_lkf.setCovariance(init_state_cov);
      };

      const int32_t id;

      void prediction_step(const ros::Time& stamp, double pos_std, double vel_std)
      {
        const double dt = (stamp - m_last_lkf_update).toSec();
        const lkf_A_t A = create_A(dt);
        const lkf_R_t R = create_R(dt, pos_std, vel_std);
        m_lkf.setA(A);
        m_lkf.setR(R);
        m_lkf.iterateWithoutCorrection();
        m_last_lkf_update = stamp;
      }

      void correction_step(const Measurement& meas, double meas_loglikelihood)
      {
        if (meas.stamp > m_last_lkf_update)
        {
          m_last_lkf_update = meas.stamp;
          m_lkf.setMeasurement(meas.position, meas.covariance);
        } else
        {
          const double dt = (meas.stamp - m_last_lkf_update).toSec();
          const lkf_A_t invA = create_A(dt);
          m_lkf.setMeasurement(meas.position, meas.covariance);
          m_lkf.doReCorrection(invA);
        }
        // TODO: This is probably not entirely true - verbessern
        m_loglikelihood = m_loglikelihood + meas_loglikelihood;
        m_last_measurement = meas;
        m_lkf.doCorrection();
        if (meas.reliable())
          m_n_corrections++;
      }

      std::tuple<Eigen::Vector3d, Eigen::Matrix3d> calc_innovation(const Measurement& meas) const
      {
        /* const Eigen::Matrix3d P = lkf.getP().block<3, 3>(0, 0); // simplify the matrix calculations a bit by ignoring the higher derivation states */
        /* const Eigen::Vector3d inn = meas.position - P*get_position(); */
        /* const Eigen::Matrix3d inn_cov = meas.covariance + P*get_position_covariance()*P.transpose(); */
        // version for this specific case (position is directly measured)
        const lkf_z_t inn = meas.position - get_position();
        const lkf_R_t inn_cov = meas.covariance + get_position_covariance();
        return std::tuple(inn, inn_cov);
      }

      inline int get_n_corrections(void) const
      {
        return m_n_corrections;
      }
      
      inline double get_loglikelihood() const
      {
        return m_loglikelihood;
      }

      inline Measurement get_last_measurement(void) const
      {
        return m_last_measurement;
      }

      inline Eigen::Vector3d get_position() const
      {
        return m_lkf.getStates().block<3, 1>(0, 0);
      }

      inline Eigen::Matrix3d get_position_covariance() const
      {
        return m_lkf.getCovariance().block<3, 3>(0, 0);
      }

      inline Eigen::Vector3d get_position_at(const ros::Time& stamp) const
      {
        const Eigen::Vector3d pos = get_position();
        /* const double dt = */ 
        return m_lkf.getStates().block<3, 1>(0, 0);
      }

      // TODO: Check that this is correct!
      inline Eigen::Matrix3d get_position_covariance_at([[maybe_unused]] const ros::Time& stamp) const
      {
        return m_lkf.getCovariance().block<3, 3>(0, 0);
      }

    private:
      int64_t m_n_corrections;
      Measurement m_last_measurement;
      double m_loglikelihood;
      ros::Time m_last_lkf_update;

      mrs_lib::Lkf m_lkf;
  };
}

#endif // HYPOTHESIS_H
