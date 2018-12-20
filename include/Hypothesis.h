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
    Hypothesis(const int id, const Measurement& init_meas, double init_vel_std, size_t hist_len);
    //}

    const int32_t id;

    /* prediction_step() method //{ */
    void prediction_step(const ros::Time& stamp, double pos_std, double vel_std);
    //}

    /* correction_step() method //{ */
    void correction_step(const Measurement& meas, [[maybe_unused]] double meas_loglikelihood);
    //}

    /* calc_innovation() method //{ */
    std::tuple<Lkf::z_t, Lkf::R_t> calc_innovation(const Measurement& meas) const;
    //}

    /* get_n_corrections() method //{ */
    inline int get_n_corrections(void) const;
    //}

    /* get_loglikelihood() method //{ */
    inline double get_loglikelihood() const;
    //}

    /* get_last_measurement() method //{ */
    inline Measurement get_last_measurement(void) const;
    //}

    /* get_position() method //{ */
    inline Lkf::z_t get_position() const;
    //}

    /* get_position_covariance() method //{ */
    inline Lkf::R_t get_position_covariance() const;
    //}

    /* get_position_at() method //{ */
    inline Lkf::z_t get_position_at(const ros::Time& stamp) const;
    //}

    /* get_position_covariance_at() method //{ */
    // TODO: change this to some kind of interpolation between LKF covariances with closest time stamps?
    inline Lkf::R_t get_position_covariance_at([[maybe_unused]] const ros::Time& stamp) const;
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

    void update_lkf_history(const lkf_bfr_t::iterator& first_lkf_it, const meas_bfr_t::const_iterator& first_meas_it);

    /* predict() method //{ */
    Lkf predict(Lkf lkf, const ros::Time& to_stamp, double pos_std, double vel_std);
    //}

    /* predict() method //{ */
    Lkf predict(Lkf lkf, const ros::Time& to_stamp);
    //}

    /* ccorrect_at_time() method //{ */
    // predicts the LKF to the time of the measurement and then does the correction step using the measurement
    Lkf correct_at_time(Lkf lkf, const Lkf::z_t& meas_pos, const Lkf::R_t& meas_cov, const ros::Time& meas_stamp);
    //}
  };
}  // namespace uav_localize

#endif  // HYPOTHESIS_H
