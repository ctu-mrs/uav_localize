#include "Hypothesis.h"

namespace uav_localize
{
  Hypothesis::Hypothesis(const int id, const Measurement& init_meas, double init_vel_std, size_t hist_len)
      : id(id), m_n_corrections(0), m_loglikelihood(0), m_lkfs(hist_len), m_measurements(hist_len)
  {
    m_last_lkf_update = init_meas.stamp;
    m_last_measurement = init_meas;

    // Initialize the LKF using the initialization measurement
    Lkf lkf(Lkf::A_t(), Lkf::B_t(), create_H(), Lkf::P_t(), Lkf::Q_t(), Lkf::R_t());
    lkf.x.block<3, 1>(0, 0) = init_meas.position;
    lkf.x.block<3, 1>(3, 0) = Eigen::Vector3d::Zero();
    lkf.P.setZero();
    lkf.P.block<3, 3>(0, 0) = init_meas.covariance;
    lkf.P.block<3, 3>(3, 3) = init_vel_std * Eigen::Matrix3d::Identity();
    lkf.source = init_meas.source;
    lkf.stamp = init_meas.stamp;

    m_lkfs.push_back(lkf);
    m_measurements.push_back(init_meas);
  };

  void Hypothesis::prediction_step(const ros::Time& stamp, double pos_std, double vel_std)
  {
    Lkf n_lkf = predict(m_lkfs.back(), stamp, pos_std, vel_std);
    m_lkfs.push_back(n_lkf);
    m_last_lkf_update = stamp;
  }

  void Hypothesis::correction_step(const Measurement& meas, [[maybe_unused]] double meas_loglikelihood)
  {
    assert(!m_lkfs.empty() && !m_measurements.empty());
    // there must already be at least one lkf in the buffer (which should be true)
    const lkf_bfr_t::iterator lkf_prev_it = remove_const(find_prev(meas.stamp, m_lkfs), m_lkfs);
    const meas_bfr_t::iterator meas_next_it = remove_const(find_prev(meas.stamp, m_measurements), m_measurements)+1;

    // insert the new measurement into the measurement buffer (potentially kicking out the oldest measurement
    // at the beginning of the buffer)
    const meas_bfr_t::const_iterator meas_new_it = m_measurements.insert(meas_next_it, meas); // TODO: find out why this is freezing the program!
    // update the LKFs according to the new measurement history
    update_lkf_history(lkf_prev_it, meas_new_it);

    if (meas.reliable())
      m_n_corrections++;
  }

  std::tuple<Hypothesis::Lkf::z_t, Hypothesis::Lkf::R_t> Hypothesis::calc_innovation(const Measurement& meas) const
  {
    /* const Eigen::Matrix3d P = lkf.getP().block<3, 3>(0, 0); // simplify the matrix calculations a bit by ignoring the higher derivation states */
    /* const Eigen::Vector3d inn = meas.position - P*get_position(); */
    /* const Eigen::Matrix3d inn_cov = meas.covariance + P*get_position_covariance()*P.transpose(); */
    // version for this specific case (position is directly measured)
    const Lkf::z_t inn = meas.position - get_position();
    const Lkf::R_t inn_cov = meas.covariance + get_position_covariance();
    return std::tuple(inn, inn_cov);
  }

  inline int Hypothesis::get_n_corrections(void) const
  {
    return m_n_corrections;
  }

  inline double Hypothesis::get_loglikelihood() const
  {
    return m_loglikelihood;
  }

  inline Measurement Hypothesis::get_last_measurement(void) const
  {
    return m_last_measurement;
  }

  inline Hypothesis::Lkf::z_t Hypothesis::get_position() const
  {
    Lkf::z_t ret = m_lkfs.back().x.block<3, 1>(0, 0);
    return ret;
  }

  inline Hypothesis::Lkf::R_t Hypothesis::get_position_covariance() const
  {
    Lkf::R_t ret = m_lkfs.back().P.block<3, 3>(0, 0);
    return ret;
  }

  inline Hypothesis::Lkf::z_t Hypothesis::get_position_at(const ros::Time& stamp) const
  {
    const Lkf::z_t pos = get_position();
    /* const double dt = */
    return pos;
  }

  inline Hypothesis::Lkf::R_t Hypothesis::get_position_covariance_at([[maybe_unused]] const ros::Time& stamp) const
  {
    Lkf::R_t ret = m_lkfs.back().P.block<3, 3>(0, 0);
    return ret;
  }

  void Hypothesis::update_lkf_history(const lkf_bfr_t::iterator& first_lkf_it, const meas_bfr_t::const_iterator& first_meas_it)
  {
    using lkf_it = lkf_bfr_t::iterator;
    using meas_it = meas_bfr_t::const_iterator;
    lkf_it cur_lkf_it = first_lkf_it;
    meas_it cur_meas_it = first_meas_it;
    Lkf updating_lkf = *cur_lkf_it;

    while (cur_lkf_it != m_lkfs.end())
    {
      Lkf& cur_lkf = *cur_lkf_it;

      while (cur_meas_it != m_measurements.end()
          && (cur_lkf_it+1 == m_lkfs.end() || (*cur_meas_it).stamp < (cur_lkf_it+1)->stamp))
      {
        const Measurement& cur_meas = *cur_meas_it;
        updating_lkf = correct_at_time(updating_lkf, cur_meas.position, cur_meas.covariance, cur_meas.stamp);
        updating_lkf.source = cur_meas.source;
        cur_meas_it++;
      }

      // predict the LKF at the time of the next one in the queue
      updating_lkf = predict(updating_lkf, cur_lkf.stamp);
      // replace the newer LKF in the queue with the continuation of the first one
      cur_lkf = updating_lkf;

      if (cur_meas_it == m_measurements.end())
        break;
      cur_lkf_it++;
    }

    // if there are some more measurements after the last LKF time stamp, use them to generate new LKFs 
    while (cur_meas_it != m_measurements.end())
    {
      const Measurement& cur_meas = *cur_meas_it;
      updating_lkf = correct_at_time(updating_lkf, cur_meas.position, cur_meas.covariance, cur_meas.stamp);
      updating_lkf.source = cur_meas.source;
      m_lkfs.push_back(updating_lkf);
      cur_meas_it++;
    }
  }

  Hypothesis::Lkf Hypothesis::predict(Lkf lkf, const ros::Time& to_stamp, double pos_std, double vel_std)
  {
    const double dt = (to_stamp - lkf.stamp).toSec();
    lkf.Q = create_Q(dt, pos_std, vel_std);
    return predict(lkf, to_stamp);
  }

  Hypothesis::Lkf Hypothesis::predict(Lkf lkf, const ros::Time& to_stamp)
  {
    const double dt = (to_stamp - lkf.stamp).toSec();
    lkf.A = create_A(dt);
    lkf.prediction_step();
    lkf.stamp = to_stamp;
    return lkf;
  }

  Hypothesis::Lkf Hypothesis::correct_at_time(Lkf lkf, const Lkf::z_t& meas_pos, const Lkf::R_t& meas_cov, const ros::Time& meas_stamp)
  {
    lkf = predict(lkf, meas_stamp);
    lkf.z = meas_pos;
    lkf.R = meas_cov;
    lkf.correction_step();
    lkf.stamp = meas_stamp;
    return lkf;
  }
}  // namespace uav_localize
