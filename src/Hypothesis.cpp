#include "Hypothesis.h"

namespace uav_localize
{
  /* constructor //{ */
  Hypothesis::Hypothesis(const int id, const Measurement& init_meas, double lkf_init_vel_std, double lkf_pos_std, double lkf_vel_std, size_t hist_len)
      : id(id), m_n_corrections(0), m_loglikelihood(0), m_lkf_pos_std(lkf_pos_std), m_lkf_vel_std(lkf_vel_std), m_lkfs(hist_len)
  {
    // Initialize the LKF using the initialization measurement
    Lkf lkf(Lkf::A_t(), Lkf::B_t(), create_H(), Lkf::P_t(), Lkf::Q_t(), Lkf::R_t());
    lkf.x.block<3, 1>(0, 0) = init_meas.position;
    lkf.x.block<3, 1>(3, 0) = Eigen::Vector3d::Zero();
    lkf.P.setZero();
    lkf.P.block<3, 3>(0, 0) = init_meas.covariance;
    lkf.P.block<3, 3>(3, 3) *= lkf_init_vel_std * Eigen::Matrix3d::Identity();
    lkf.correction_meas = init_meas;
    lkf.stamp = init_meas.stamp;

    m_lkfs.push_back(lkf);
  };
  //}

  /* /1* prediction_step() method //{ *1/ */
  /* void Hypothesis::prediction_step(const ros::Time& stamp, double pos_std, double vel_std) */
  /* { */
  /*   Lkf n_lkf = predict(m_lkfs.back(), stamp, pos_std, vel_std); */
  /*   m_lkfs.push_back(n_lkf); */
  /* } */
  /* //} */

  /* correction_step() method //{ */
  void Hypothesis::correction_step(const Measurement& meas, [[maybe_unused]] double meas_loglikelihood)
  {
    assert(!m_lkfs.empty());
    // there must already be at least one lkf in the buffer (which should be true)
    const lkf_bfr_t::iterator lkf_prev_it = remove_const(find_prev(meas.stamp, m_lkfs), m_lkfs);
    const lkf_bfr_t::iterator lkf_next_it = lkf_prev_it+1;

    // create the new LKF
    Lkf lkf_n = correct_at_time(*lkf_prev_it, meas);
    // insert the new LKF into the LKF buffer (potentially kicking out the oldest LKF
    // at the beginning of the buffer)
    const lkf_bfr_t::iterator lkf_new_it = m_lkfs.insert(lkf_next_it, lkf_n);
    // update the LKFs according to the new LKF history
    update_lkf_history(lkf_new_it);

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
    return m_lkfs.back().correction_meas;
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
    Lkf::z_t ret = m_latest_lkf.x.block<3, 1>(0, 0);
    return ret;
  }
  //}

  /* get_position_covariance() method //{ */
  Hypothesis::Lkf::R_t Hypothesis::get_position_covariance() const
  {
    Lkf::R_t ret = m_latest_lkf.P.block<3, 3>(0, 0);
    return ret;
  }
  //}

  /* /1* get_position_at() method //{ *1/ */
  /* Hypothesis::Lkf::z_t Hypothesis::get_position_at([[maybe_unused]] const ros::Time& stamp) const */
  /* { */
  /*   const lkf_bfr_t::iterator lkf_prev_it = remove_const(find_prev(meas.stamp, m_lkfs), m_lkfs); */
  /*   const Lkf::z_t pos = get_position(); */
  /*   return pos; */
  /* } */
  /* //} */

  /* get_position_and_covariance_at() method //{ */
  std::tuple<Hypothesis::Lkf::z_t, Hypothesis::Lkf::R_t> Hypothesis::get_position_and_covariance_at(const ros::Time& stamp) const
  {
    Hypothesis::Lkf lkf = *find_prev(stamp, m_lkfs);
    lkf = predict(lkf, stamp);
    Lkf::z_t z_ret = lkf.x.block<3, 1>(0, 0);
    Lkf::R_t R_ret = lkf.P.block<3, 3>(0, 0);
    return std::tuple(z_ret, R_ret);
  }
  //}

  /* update_lkf_history() method //{ */
  void Hypothesis::update_lkf_history(const lkf_bfr_t::iterator& lkf_start_it)
  {
    using lkf_it_t = lkf_bfr_t::iterator;
    lkf_it_t lkf_cur_it = lkf_start_it;
    lkf_it_t lkf_next_it = lkf_start_it+1;

    while (lkf_next_it != m_lkfs.end())
    {
      Lkf& cur_lkf = *lkf_cur_it;
      Lkf& next_lkf = *lkf_next_it;

      next_lkf = correct_at_time(cur_lkf, next_lkf.correction_meas);

      lkf_cur_it++;
      lkf_next_it++;
    }
  }
  //}

  /* predict() method //{ */
  Hypothesis::Lkf Hypothesis::predict(Lkf lkf, const ros::Time& to_stamp, double pos_std, double vel_std)
  {
    m_lkf_pos_std = pos_std;
    m_lkf_vel_std = vel_std;
    return predict(lkf, to_stamp);
  }

  Hypothesis::Lkf Hypothesis::predict(Lkf lkf, const ros::Time& to_stamp) const
  {
    const double dt = (to_stamp - lkf.stamp).toSec();
    lkf.Q = create_Q(dt, m_lkf_pos_std, m_lkf_vel_std);
    lkf.A = create_A(dt);
    lkf.prediction_step();
    lkf.stamp = to_stamp;
    lkf.correction_meas.source = Measurement::source_t::lkf_prediction;
    return lkf;
  }
  //}

  /* predict_to() method //{ */
  void Hypothesis::predict_to(const ros::Time& to_stamp, double pos_std, double vel_std)
  {
    Hypothesis::Lkf lkf = m_lkfs.back();
    lkf = predict(lkf, to_stamp, pos_std,  vel_std);
    m_latest_lkf = lkf;
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
    lkf.correction_meas = meas;
    return lkf;
  }
  //}
}  // namespace uav_localize
