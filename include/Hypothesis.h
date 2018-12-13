#ifndef HYPOTHESIS_H
#define HYPOTHESIS_H

#include "main.h"
#include "Measurement.h"
#include "Lkf.h"

namespace uav_localize
{
  class Hypothesis
  {
    private:
      /* Definitions of the LKF (consts, typedefs, etc.) //{ */
      static const int n_states = 6;
      static const int n_inputs = 0;
      static const int n_measurements = 3;
      using Lkf = uav_localize::Lkf<n_states, n_inputs, n_measurements>;

      inline static Lkf::A_t create_A(double dt)
      {
        Lkf::A_t A;
        A << 1, 0, 0, dt, 0, 0,
             0, 1, 0, 0, dt, 0,
             0, 0, 1, 0, 0, dt,
             0, 0, 0, 1, 0, 0,
             0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 1;
        return A;
      }

      inline static Lkf::H_t create_H()
      {
        Lkf::H_t H;
        H << 1, 0, 0, 0, 0, 0,
             0, 1, 0, 0, 0, 0,
             0, 0, 1, 0, 0, 0;
        return H;
      }

      inline Lkf::Q_t create_Q(double dt, double pos_std, double vel_std) const
      {
        Lkf::Q_t Q = Lkf::Q_t::Identity();
        Q.block<3, 3>(0, 0) *= dt * pos_std;
        Q.block<3, 3>(3, 3) *= dt * vel_std;
        return Q;
      }
      //}

    public:
      /* constructor //{ */
      Hypothesis(const int id, const Measurement& init_meas, double init_vel_std, size_t hist_len)
        : id(id), m_n_corrections(0), m_loglikelihood(0)/*, m_lkfs(hist_len)*/
      {
        m_last_lkf_update = init_meas.stamp;
      
        // Initialize the LKF using the initialization measurement
        Lkf lkf(Lkf::A_t(), Lkf::B_t(), create_H(), Lkf::P_t(), Lkf::Q_t(), Lkf::R_t());
        lkf.x.block<3, 1>(0, 0) = init_meas.position;
        lkf.x.block<3, 1>(3, 0) = Eigen::Vector3d::Zero();
        lkf.Q.setZero();
        lkf.Q.block<3, 3>(0, 0) = init_meas.covariance;
        lkf.Q.block<3, 3>(3, 3) = init_vel_std * Eigen::Matrix3d::Identity();
      
        m_lkf = lkf;
        /* m_lkfs.push_back(lkf); */
      };
      //}

      const int32_t id;

      /* prediction_step() method //{ */
      void prediction_step(const ros::Time& stamp, double pos_std, double vel_std)
      {
        const double dt = (stamp - m_last_lkf_update).toSec();
        const Lkf::A_t A = create_A(dt);
        const Lkf::Q_t Q = create_Q(dt, pos_std, vel_std);
        m_lkf.A = A;
        m_lkf.Q = Q;
        m_lkf.prediction_step();
        m_last_lkf_update = stamp;
      }
      //}

      /* correction_step() method //{ */
      void correction_step(const Measurement& meas, double meas_loglikelihood)
      {
        if (meas.stamp > m_last_lkf_update)
        {
          m_last_lkf_update = meas.stamp;
          m_lkf.z = meas.position;
          m_lkf.R = meas.covariance;
        } else
        {
          /* const double dt = (meas.stamp - m_last_lkf_update).toSec(); */
          /* const lkf_A_t invA = create_A(dt); */
          /* m_lkf.z = meas.position; */
          /* m_lkf.R = meas.covariance; */
          /* m_lkf.doReCorrection(invA); */
        }
        // TODO: This is probably not entirely true - verbessern
        m_loglikelihood = m_loglikelihood + meas_loglikelihood;
        m_last_measurement = meas;
        m_lkf.correction_step();
        if (meas.reliable())
          m_n_corrections++;
      }
      //}

      /* calc_innovation() method //{ */
      std::tuple<Lkf::z_t, Lkf::R_t> calc_innovation(const Measurement& meas) const
      {
        /* const Eigen::Matrix3d P = lkf.getP().block<3, 3>(0, 0); // simplify the matrix calculations a bit by ignoring the higher derivation states */
        /* const Eigen::Vector3d inn = meas.position - P*get_position(); */
        /* const Eigen::Matrix3d inn_cov = meas.covariance + P*get_position_covariance()*P.transpose(); */
        // version for this specific case (position is directly measured)
        const Lkf::z_t inn = meas.position - get_position();
        const Lkf::R_t inn_cov = meas.covariance + get_position_covariance();
        return std::tuple(inn, inn_cov);
      }
      //}

      /* get_n_corrections() method //{ */
      inline int get_n_corrections(void) const
      {
        return m_n_corrections;
      }
      //}
      
      /* get_loglikelihood() method //{ */
      inline double get_loglikelihood() const
      {
        return m_loglikelihood;
      }
      //}

      /* get_last_measurement() method //{ */
      inline Measurement get_last_measurement(void) const
      {
        return m_last_measurement;
      }
      //}

      /* get_position() method //{ */
      inline Lkf::z_t get_position() const
      {
        Lkf::z_t ret = m_lkf.x.block<3, 1>(0, 0);
        return ret;
      }
      //}

      /* get_position_covariance() method //{ */
      inline Lkf::R_t get_position_covariance() const
      {
        Lkf::R_t ret = m_lkf.P.block<3, 3>(0, 0);
        return ret;
      }
      //}

      /* get_position_at() method //{ */
      inline Lkf::z_t get_position_at(const ros::Time& stamp) const
      {
        const Lkf::z_t pos = get_position();
        /* const double dt = */ 
        return pos;
      }
      //}

      /* get_position_covariance_at() method //{ */
      // TODO: Check that this is correct!
      inline Lkf::R_t get_position_covariance_at([[maybe_unused]] const ros::Time& stamp) const
      {
        Lkf::R_t ret = m_lkf.P.block<3, 3>(0, 0);
        return ret;
      }
      //}

    private:
      int64_t m_n_corrections;
      Measurement m_last_measurement;
      double m_loglikelihood;
      ros::Time m_last_lkf_update;

      Lkf m_lkf;
      /* boost::circular_buffer<Lkf> m_lkfs; */
  };
}

#endif // HYPOTHESIS_H
