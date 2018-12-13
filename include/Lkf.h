#include <Eigen/Dense>

namespace uav_localize
{
  template <unsigned n_states, unsigned n_inputs, unsigned n_measurements>
  class Lkf
  {
  public:
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
    typedef Eigen::Matrix<double, n, n> R_t;  // process covariance n*n
    typedef Eigen::Matrix<double, p, p> Q_t;  // measurement covariance p*p

  public:
    Lkf(const A_t& A, const B_t& B, const H_t& H, const P_t& P, const R_t& R, const Q_t& Q);

    Lkf(const Lkf& lkf);  // copy constructor

    Lkf& operator=(const Lkf& lkf);

  public:
    Eigen::MatrixXd A;  // system matrix n*n
    Eigen::MatrixXd B;  // input matrix n*m
    Eigen::MatrixXd R;  // process covariance n*n
    Eigen::MatrixXd P;  // state vector covariance n*n
    Eigen::MatrixXd Q;  // measurement covariance p*p
    Eigen::MatrixXd H;  // measurement mapping p*n

    Eigen::VectorXd x;  // state vector
    Eigen::VectorXd z;  // the last measurement
    Eigen::VectorXd u;  // the last system inpud

  private:
    void prediction_impl();
    void correction_impl();
  };
}  // namespace uav_localize
