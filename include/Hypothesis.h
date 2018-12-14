#ifndef HYPOTHESIS_H
#define HYPOTHESIS_H

#include "main.h"
#include "Measurement.h"
#include "Lkf.h"

namespace uav_localize
{
  /* remove_cons() function //{ */
  template <typename T>
  inline typename T::iterator remove_const(const typename T::const_iterator& it, T& cont)
  {
    typename T::iterator ret = cont.begin();
    std::advance(ret, std::distance((typename T::const_iterator)ret, it));
    return ret;
  }
  //}

  class Hypothesis
  {
  private:
    /* Definitions of the LKF (consts, typedefs, etc.) //{ */
    static const int n_states = 6;
    static const int n_inputs = 0;
    static const int n_measurements = 3;
    using Lkf = uav_localize::Lkf_stamped<n_states, n_inputs, n_measurements>;

    inline static Lkf::A_t create_A(double dt)
    {
      Lkf::A_t A;
      A << 1, 0, 0, dt, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1;
      return A;
    }

    inline static Lkf::H_t create_H()
    {
      Lkf::H_t H;
      H << 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0;
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

    using lkf_bfr_t = boost::circular_buffer<Lkf>;
    using meas_bfr_t = boost::circular_buffer<Measurement>;

  public:
    /* constructor //{ */
    Hypothesis(const int id, const Measurement& init_meas, double init_vel_std, size_t hist_len)
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

      m_lkfs.push_back(lkf);
    };
    //}

    const int32_t id;

    /* prediction_step() method //{ */
    void prediction_step(const ros::Time& stamp, double pos_std, double vel_std)
    {
      Lkf n_lkf = predict(m_lkfs.back(), stamp, pos_std, vel_std);
      m_lkfs.push_back(n_lkf);
      m_last_lkf_update = stamp;
    }
    //}

    /* correction_step() method //{ */
    void correction_step(const Measurement& meas, [[maybe_unused]] double meas_loglikelihood)
    {
      // there must already be at least one lkf in the buffer (which should be true)
      const lkf_bfr_t::iterator lkf_prev_it = remove_const(find_prev(meas.stamp, m_lkfs), m_lkfs);
      const meas_bfr_t::iterator meas_next_it = remove_const(find_prev(meas.stamp, m_measurements), m_measurements)+1;

      // insert the new measurement into the measurement buffer (potentially kicking out the oldest measurement
      // at the beginning of the buffer)
      const meas_bfr_t::const_iterator meas_new_it = m_measurements.insert(meas_next_it, meas);
      // update the LKFs according to the new measurement history
      update_lkf_history(lkf_prev_it, meas_new_it);

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
      Lkf::z_t ret = m_lkfs.back().x.block<3, 1>(0, 0);
      return ret;
    }
    //}

    /* get_position_covariance() method //{ */
    inline Lkf::R_t get_position_covariance() const
    {
      Lkf::R_t ret = m_lkfs.back().P.block<3, 3>(0, 0);
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
      Lkf::R_t ret = m_lkfs.back().P.block<3, 3>(0, 0);
      return ret;
    }
    //}

  private:
    int64_t m_n_corrections;
    double m_loglikelihood;
    Measurement m_last_measurement;
    ros::Time m_last_lkf_update;

    lkf_bfr_t m_lkfs;
    meas_bfr_t m_measurements;

  private:
    /* find_prev() method //{ */
    template <class T>
    inline void slice_in_half(const typename T::const_iterator b_in, const typename T::const_iterator e_in,
                              typename T::const_iterator& b_out, typename T::const_iterator& m_out, typename T::const_iterator& e_out)
    {
      b_out = b_in;
      m_out = b_in + (e_in - b_in) / 2;
      e_out = e_in;
    }

    template <class T>
    const typename T::const_iterator find_prev(const ros::Time& stamp, const T& bfr)
    {
      using it_t = typename T::const_iterator;
      it_t b = std::begin(bfr);
      if (bfr.empty())
        return b;
      it_t e = std::end(bfr) - 1;
      it_t m = e;
      do
      {
        const double cmp = ((*m).stamp - stamp).toSec();
        if (cmp > 0.0)
          slice_in_half<T>(b, m, b, m, e);  // avoiding tuples for better performance
        else if (cmp < 0.0)
          slice_in_half<T>(m, e, b, m, e);
        else
          break;
      } while (b != m);
      return m;
    }
    //}


    void update_lkf_history(const lkf_bfr_t::iterator& first_lkf_it, const meas_bfr_t::const_iterator& first_meas_it)
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
          updating_lkf = predict(updating_lkf, cur_meas.stamp);
          updating_lkf = correct(updating_lkf, cur_meas.position, cur_meas.covariance, cur_meas.stamp);
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

      /* // if the last LKF was updated with a new measurement, add it to the */ 
      /* if (cur_lkf_it != m_lkfs.end()) */
      /* { */
        
      /* } */
    }


    /* predict() method //{ */
    Lkf predict(Lkf lkf, const ros::Time& to_stamp, double pos_std, double vel_std)
    {
      const double dt = (to_stamp - lkf.stamp).toSec();
      lkf.Q = create_Q(dt, pos_std, vel_std);
      return predict(lkf, to_stamp);
    }
    //}

    /* predict() method //{ */
    Lkf predict(Lkf lkf, const ros::Time& to_stamp)
    {
      const double dt = (to_stamp - lkf.stamp).toSec();
      lkf.A = create_A(dt);
      lkf.prediction_step();
      lkf.stamp = to_stamp;
      return lkf;
    }
    //}

    /* correct() method //{ */
    Lkf correct(Lkf lkf, const Lkf::z_t& meas_pos, const Lkf::R_t& meas_cov, const ros::Time& meas_stamp)
    {
      lkf.z = meas_pos;
      lkf.R = meas_cov;
      lkf.correction_step();
      lkf.stamp = meas_stamp;
      return lkf;
    }
    //}
  };
}  // namespace uav_localize

#endif  // HYPOTHESIS_H
