#include "Hypothesis.h"

namespace uav_localize
{
  /* constructor //{ */
  Hypothesis::Hypothesis(const int id, const Measurement& init_meas, double lkf_init_vel_std, double lkf_pos_std, double lkf_vel_std, size_t hist_len)
      : id(id), m_n_corrections(0), m_loglikelihood(0), m_lkf_pos_std(lkf_pos_std), m_lkf_vel_std(lkf_vel_std), m_lkfs(hist_len), m_measurements(hist_len)
  {
    // Initialize the LKF using the initialization measurement
    Lkf lkf(Lkf::A_t(), Lkf::B_t(), create_H(), Lkf::P_t(), Lkf::Q_t(), Lkf::R_t());
    lkf.x.block<3, 1>(0, 0) = init_meas.position;
    lkf.x.block<3, 1>(3, 0) = Eigen::Vector3d::Zero();
    lkf.P.setZero();
    lkf.P.block<3, 3>(0, 0) = init_meas.covariance;
    lkf.P.block<3, 3>(3, 3) *= lkf_init_vel_std * Eigen::Matrix3d::Identity();
    lkf.source = init_meas.source;
    lkf.stamp = init_meas.stamp;

    m_lkfs.push_back(lkf);
    m_measurements.push_back(init_meas);
  };
  //}

  /* prediction_step() method //{ */
  void Hypothesis::prediction_step(const ros::Time& stamp, double pos_std, double vel_std)
  {
    Lkf n_lkf = predict(m_lkfs.back(), stamp, pos_std, vel_std);
    m_lkfs.push_back(n_lkf);
  }
  //}

  /* correction_step() method //{ */
  void Hypothesis::correction_step(const Measurement& meas, [[maybe_unused]] double meas_loglikelihood)
  {
    assert(!m_lkfs.empty() && !m_measurements.empty());
    // there must already be at least one lkf in the buffer (which should be true)
    const lkf_bfr_t::iterator lkf_prev_it = remove_const(find_prev(meas.stamp, m_lkfs), m_lkfs);
    const meas_bfr_t::iterator meas_next_it = remove_const(find_prev(meas.stamp, m_measurements), m_measurements) + 1;

    // insert the new measurement into the measurement buffer (potentially kicking out the oldest measurement
    // at the beginning of the buffer)
    const meas_bfr_t::const_iterator meas_new_it = m_measurements.insert(meas_next_it, meas);  // TODO: find out why this is freezing the program!
    // update the LKFs according to the new measurement history
    update_lkf_history(lkf_prev_it, meas_new_it);

    if (meas.reliable())
      m_n_corrections++;
  }
  //}

  /* calc_innovation() method //{ */
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
  //}

  /* get_n_corrections() method //{ */
  int Hypothesis::get_n_corrections(void) const
  {
    return m_n_corrections;
  }
  //}

  /* get_loglikelihood() method //{ */
  double Hypothesis::get_loglikelihood() const
  {
    return m_loglikelihood;
  }
  //}

  /* get_last_measurement() method //{ */
  Measurement Hypothesis::get_last_measurement(void) const
  {
    return m_measurements.back();
  }
  //}

  /* get_last_lkf() method //{ */
  const Hypothesis::Lkf& Hypothesis::get_last_lkf(void) const
  {
    return m_lkfs.back();
  }
  //}

  /* get_lkfs() method //{ */
  const Hypothesis::lkf_bfr_t& Hypothesis::get_lkfs(void) const
  {
    return m_lkfs;
  }
  //}

  /* get_position() method //{ */
  Hypothesis::Lkf::z_t Hypothesis::get_position() const
  {
    Lkf::z_t ret = m_lkfs.back().x.block<3, 1>(0, 0);
    return ret;
  }
  //}

  /* get_position_covariance() method //{ */
  Hypothesis::Lkf::R_t Hypothesis::get_position_covariance() const
  {
    Lkf::R_t ret = m_lkfs.back().P.block<3, 3>(0, 0);
    return ret;
  }
  //}

  /* get_position_at() method //{ */
  Hypothesis::Lkf::z_t Hypothesis::get_position_at([[maybe_unused]] const ros::Time& stamp) const
  {
    const Lkf::z_t pos = get_position();
    /* const double dt = */
    return pos;
  }
  //}

  /* get_position_covariance_at() method //{ */
  Hypothesis::Lkf::R_t Hypothesis::get_position_covariance_at([[maybe_unused]] const ros::Time& stamp) const
  {
    Lkf::R_t ret = m_lkfs.back().P.block<3, 3>(0, 0);
    return ret;
  }
  //}

  /* update_lkf_history() method //{ */
  void Hypothesis::update_lkf_history(const lkf_bfr_t::iterator& first_lkf_it, const meas_bfr_t::const_iterator& first_meas_it)
  {
    using lkf_it = lkf_bfr_t::iterator;
    using meas_it = meas_bfr_t::const_iterator;
    lkf_it cur_lkf_it = first_lkf_it;
    meas_it cur_meas_it = first_meas_it;
    Lkf updating_lkf = *cur_lkf_it;
    int added = 0;

    while (cur_lkf_it != m_lkfs.end())
    {
      Lkf& cur_lkf = *cur_lkf_it;

      while (cur_meas_it != m_measurements.end() && (cur_lkf_it + 1 == m_lkfs.end() || (*cur_meas_it).stamp < (cur_lkf_it + 1)->stamp))
      {
        const Measurement& cur_meas = *cur_meas_it;
        updating_lkf = correct_at_time(updating_lkf, cur_meas);
        cur_meas_it++;
      }

      // insert a new LKF corresponding to this measurement
      if (updating_lkf.stamp != cur_lkf.stamp)
      {
        added++;
        cur_lkf_it = m_lkfs.insert(cur_lkf_it+1, updating_lkf);
        // predict the LKF at the time of the next one in the queue
        updating_lkf = predict(updating_lkf, cur_lkf.stamp);
      }
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
      updating_lkf = correct_at_time(updating_lkf, cur_meas);
      m_lkfs.push_back(updating_lkf);
      cur_meas_it++;
    }
    std::cout << "added lkfs: " << added << std::endl;
  }
  //}

  /* predict() method //{ */
  Hypothesis::Lkf Hypothesis::predict(Lkf lkf, const ros::Time& to_stamp, double pos_std, double vel_std)
  {
    m_lkf_pos_std = pos_std;
    m_lkf_vel_std = vel_std;
    return predict(lkf, to_stamp);
  }

  Hypothesis::Lkf Hypothesis::predict(Lkf lkf, const ros::Time& to_stamp)
  {
    const double dt = (to_stamp - lkf.stamp).toSec();
    lkf.Q = create_Q(dt, m_lkf_pos_std, m_lkf_vel_std);
    lkf.A = create_A(dt);
    lkf.prediction_step();
    lkf.stamp = to_stamp;
    lkf.source = Measurement::source_t::lkf_prediction;
    return lkf;
  }
  //}

  /* correct_at_time() method //{ */
    // predicts the LKF to the time of the measurement and then does the correction step using the measurement
  Hypothesis::Lkf Hypothesis::correct_at_time(Lkf lkf, const Measurement& meas)
  {
    lkf = predict(lkf, meas.stamp);
    lkf.z = meas.position;
    lkf.R = meas.covariance;
    lkf.correction_step();
    lkf.stamp = meas.stamp;
    lkf.source = meas.source;
    return lkf;
  }
  //}
}  // namespace uav_localize
