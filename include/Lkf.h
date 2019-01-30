#include <Eigen/Dense>
#include <std_msgs/Time.h>
#include "Measurement.h"

namespace uav_localize
{
  template <unsigned n_states, unsigned n_inputs, unsigned n_measurements>
  class Lkf_base
  {
  public:
    /* LKF definitions (typedefs, constants etc) //{ */
    static const unsigned n = n_states;
    static const unsigned m = n_inputs;
    static const unsigned p = n_measurements;
    
    typedef Eigen::Matrix<double, n, 1> x_t;  // state vector n*1
    typedef Eigen::Matrix<double, m, 1> u_t;  // input vector m*1
    typedef Eigen::Matrix<double, p, 1> z_t;  // measurement vector p*1
    
    typedef Eigen::Matrix<double, n, n> A_t;  // system matrix n*n
    typedef Eigen::Matrix<double, n, m> B_t;  // input matrix n*m
    typedef Eigen::Matrix<double, p, n> H_t;  // measurement mapping p*n
    typedef Eigen::Matrix<double, n, n> P_t;  // state covariance n*n
    typedef Eigen::Matrix<double, n, n> Q_t;  // process covariance n*n
    typedef Eigen::Matrix<double, p, p> R_t;  // measurement covariance p*p

    typedef Eigen::Matrix<double, n, p> K_t;  // kalman gain n*p
    
    struct InverseException : public std::exception
    {
      const char *what() const throw() {
        return "LKF: could not compute matrix inversion!!! Fix your covariances (the measurement's is probably too low...)";
      }
    };
    //}

  public:
    Lkf_base() {};
    Lkf_base(const A_t& A, const B_t& B, const H_t& H, const P_t& P, const Q_t& Q, const R_t& R)
      : A(A), B(B), H(H), P(P), Q(Q), R(R)
    {};

    /* prediction_step() method //{ */
    void prediction_step()
    {
      x = state_predict(A, x, B, u);
      P = covariance_predict(A, P, Q);
    }
    //}

    /* correction_step() method //{ */
    void correction_step()
    {
      // the correction phase
      R_t tmp = H*P*H.transpose() + R;
    
      Eigen::ColPivHouseholderQR<R_t> qr(tmp);
      if (!qr.isInvertible())
      {
        // add some stuff to the tmp matrix diagonal to make it invertible
        R_t ident = R_t::Identity();
        tmp += 1e-9*ident;
        qr.compute(tmp);
        if (!qr.isInvertible())
        {
          // never managed to make this happen except for explicitly putting NaNs in the input
          throw InverseException();
        }
      }
      tmp = qr.inverse();
    
      const K_t K = P*H.transpose()*tmp;
      x           = x + K * (z - (H*x));
      P           = (P_t::Identity() - (K*H))*P;
    }
    //}

  public:
    A_t A;  // system matrix n*n
    B_t B;  // input matrix n*m
    H_t H;  // measurement mapping p*n
    P_t P;  // state vector covariance n*n
    Q_t Q;  // process covariance n*n
    R_t R;  // measurement covariance p*p

    x_t x;  // state vector
    z_t z;  // measurement vector
    u_t u;  // system input vector

  private:
    /* covariance_predict() method //{ */
    static P_t covariance_predict(const A_t& A, const P_t& P, const Q_t& Q)
    {
      return A*P*A.transpose() + Q;
    }
    //}

    /* state_predict() method //{ */
    template<bool check=n_inputs>
    static inline
    typename std::enable_if<check==0, x_t>::type
    state_predict(const A_t& A, const x_t& x, [[maybe_unused]] const B_t& B, [[maybe_unused]] const u_t& u)
    {
      return A*x;
    }
    
    template<bool check=n_inputs>
    static inline
    typename std::enable_if<check!=0, x_t>::type
    state_predict(const A_t& A, const x_t& x, const B_t& B, const u_t& u)
    {
      return A*x + B*u;
    }
    //}

  };

  template <unsigned n_states, unsigned n_inputs, unsigned n_measurements>
  class Lkf_stamped : public Lkf_base<n_states, n_inputs, n_measurements>
  {
  public:
    /* LKF definitions (typedefs, constants etc) //{ */
    using Base_class = Lkf_base<n_states, n_inputs, n_measurements>;

    using x_t = typename Base_class::x_t ;  // state vector n*1
    using u_t = typename Base_class::u_t ;  // input vector m*1
    using z_t = typename Base_class::z_t ;  // measurement vector p*1
    using A_t = typename Base_class::A_t ;  // system matrix n*n
    using B_t = typename Base_class::B_t ;  // input matrix n*m
    using H_t = typename Base_class::H_t ;  // measurement mapping p*n

    using P_t = typename Base_class::P_t ;  // state covariance n*n
    using Q_t = typename Base_class::Q_t ;  // process covariance n*n
    using R_t = typename Base_class::R_t ;  // measurement covariance p*p

    using K_t = typename Base_class::K_t ;  // kalman gain n*p
    //}

  public:
    Lkf_stamped()
      : Base_class()
    {
      correction_meas.source = Measurement::source_t::lkf_prediction;
    };
    Lkf_stamped(const A_t& A, const B_t& B, const H_t& H, const P_t& P, const Q_t& Q, const R_t& R)
      : Base_class(A, B, H, P, Q, R)
    {
      correction_meas.source = Measurement::source_t::lkf_prediction;
    };

  public:
    ros::Time stamp;
    Measurement correction_meas;
  };
}  // namespace uav_localize
