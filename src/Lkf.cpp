#include "Lkf.h"

namespace uav_localize
{

  /* state_predict_impl() method //{ */
  template<bool check=n_inputs>
  typename std::enable_if<n_inputs==0, x_t>::type
  Lkf::state_predict_impl()
  {
    return A*x;
  }
  
  template<bool check=n_inputs>
  typename std::enable_if<n_inputs!=0, x_t>::type
  Lkf::state_predict_impl()
  {
    return A*x + B*input;
  }
  //}

  /* prediction_impl() //{ */

  // implementation of the prediction step
  void Lkf::prediction_impl(void) {

    // the prediction phase
    if (m > 0) {
      x = A * x + B * input;
    } else {
      x = A * x;
    }

    cov = A * cov * A.transpose() + R;

  }

  //}

  /* correction_impl() //{ */

  // implementation of the correction step
  void Lkf::correction_impl(void) {

    // the correction phase
    MatrixXd tmp = P * cov * P.transpose() + Q;

    ColPivHouseholderQR<MatrixXd> qr(tmp);
    if (!qr.isInvertible())
    {
      // add some stuff to the tmp matrix diagonal to make it invertible
      MatrixXd ident(tmp.rows(), tmp.cols());
      ident.setIdentity();
      tmp += 1e-9*ident;
      qr.compute(tmp);
      if (!qr.isInvertible())
      {
        // never managed to make this happen except for explicitly putting NaNs in the input
        ROS_ERROR("LKF: could not compute matrix inversion!!! Fix your covariances (the measurement's is probably too low...)");
        throw InverseException();
      }
      ROS_WARN("LKF: artificially inflating matrix for inverse computation! Check your covariances (the measurement's might be too low...)");
    }
    tmp = qr.inverse();

    MatrixXd K = cov * P.transpose() * tmp;
    x          = x + K * (mes - (P * x));
    cov        = (MatrixXd::Identity(n, n) - (K * P)) * cov;

  }

  //}
}
