#include "main.h"

#include <Eigen/Geometry>

#include <nodelet/nodelet.h>

#include "LkfAssociation.h"

#include <uav_localize/LocalizationParamsConfig.h>
#include <uav_localize/LocalizedUAV.h>

using namespace std;

// shortcut type to the dynamic reconfigure manager template instance
typedef mrs_lib::DynamicReconfigureMgr<uav_localize::LocalizationParamsConfig> drmgr_t;

namespace uav_localize
{
  using Lkf = uav_localize::LkfAssociation;

  class LocalizeSingle : public nodelet::Nodelet
  {
  private:
    /* measurement_t helper struct //{ */
    struct measurement_t
    {
      Eigen::Vector3d position;
      Eigen::Matrix3d covariance;
    };
    //}

  public:
    // --------------------------------------------------------------
    // |                 main implementation methods                |
    // --------------------------------------------------------------

    /* onInit() method //{ */
    void onInit()
    {
      ros::NodeHandle nh = nodelet::Nodelet::getMTPrivateNodeHandle();

      m_node_name = "LocalizeSingle";

      /* Load parameters from ROS //{*/
      mrs_lib::ParamLoader pl(nh, m_node_name);
      // LOAD STATIC PARAMETERS
      ROS_INFO("Loading static parameters:");
      pl.load_param("world_frame", m_world_frame, std::string("local_origin"));
      pl.load_param("lkf_dt", m_lkf_dt);
      const int process_loop_rate = pl.load_param2<int>("process_loop_rate");
      const int publish_loop_rate = pl.load_param2<int>("publish_loop_rate");
      pl.load_param("min_detection_height", m_min_detection_height);

      if (!pl.loaded_successfully())
      {
        ROS_ERROR("Some compulsory parameters were not loaded successfully, ending the node");
        ros::shutdown();
      }

      // LOAD DYNAMIC PARAMETERS
      m_drmgr_ptr = make_unique<drmgr_t>(nh, m_node_name);
      /* drmgr.map_param("xy_covariance_coeff", m_xy_covariance_coeff); */
      /* drmgr.map_param("z_covariance_coeff", m_z_covariance_coeff); */
      /* drmgr.map_param("max_update_divergence", m_max_update_divergence); */
      /* drmgr.map_param("max_lkf_uncertainty", m_max_lkf_uncertainty); */
      /* drmgr.map_param("lkf_process_noise_vel", m_lkf_process_noise_vel); */
      /* drmgr.map_param("lkf_process_noise_pos", m_lkf_process_noise_pos); */
      /* drmgr.map_param("init_vel_cov", m_init_vel_cov); */
      /* drmgr.map_param("min_corrs_to_consider", m_min_corrs_to_consider); */
      if (!m_drmgr_ptr->loaded_successfully())
      {
        ROS_ERROR("Some dynamic parameter default values were not loaded successfully, ending the node");
        ros::shutdown();
      }
      //}

      /* Create publishers and subscribers //{ */
      // Initialize transform listener
      m_tf_listener_ptr = std::make_unique<tf2_ros::TransformListener>(m_tf_buffer);
      // Subscribers
      mrs_lib::SubscribeMgr smgr(nh, m_node_name);
      m_sh_detections_ptr = smgr.create_handler_threadsafe<uav_detect::Detections>("detections", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
      m_sh_trackings_ptr = smgr.create_handler_threadsafe<uav_track::Trackings>("trackings", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
      m_sh_cinfo_ptr = smgr.create_handler_threadsafe<sensor_msgs::CameraInfo>("camera_info", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
      // Publishers
      m_pub_localized_uav = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("localized_uav", 10);
      m_pub_dbg_localized_uav = nh.advertise<uav_localize::LocalizedUAV>("dbg_localized_uav", 10);
      //}

      m_lkf_update_loop_timer = nh.createTimer(ros::Duration(m_lkf_dt), &LocalizeSingle::lkf_update_loop, this);
      m_process_loop_timer = nh.createTimer(ros::Rate(process_loop_rate), &LocalizeSingle::process_loop, this);
      m_publish_loop_timer = nh.createTimer(ros::Rate(publish_loop_rate), &LocalizeSingle::publish_loop, this);

      m_last_lkf_id = 0;

      cout << "----------------------------------------------------------" << std::endl;
    }
    //}

    /* process_loop() method //{ */
    void process_loop([[maybe_unused]] const ros::TimerEvent& evt)
    {
      const bool got_depth_detections = m_sh_detections_ptr->new_data();
      const bool got_rgb_tracking = m_sh_trackings_ptr->new_data();
      const bool got_something_to_process = got_depth_detections || got_rgb_tracking;
      const bool should_process = got_something_to_process && m_sh_cinfo_ptr->has_data();
      if (should_process)
      {
        if (!m_sh_cinfo_ptr->used_data())
          m_camera_model.fromCameraInfo(m_sh_cinfo_ptr->get_data());

        if (got_depth_detections)
        {
          const uav_detect::Detections last_detections_msg = m_sh_detections_ptr->get_data();
          std::vector<measurement_t> measurements = measurements_from_message(last_detections_msg);
          std::lock_guard<std::mutex> lck(m_lkfs_mtx);
          update_lkfs(measurements, m_lkfs);
        }

        if (got_rgb_tracking)
        {
          const uav_track::Trackings last_trackings_msg = m_sh_trackings_ptr->get_data();
          std::vector<measurement_t> measurements = measurements_from_message(last_trackings_msg);
          std::lock_guard<std::mutex> lck(m_lkfs_mtx);
          update_lkfs(measurements, m_lkfs);
        }
      }
    }
    //}

    /* publish_loop() method //{ */
    void publish_loop([[maybe_unused]] const ros::TimerEvent& evt)
    {
      Lkf const* most_certain_lkf = nullptr;
      {
        std::lock_guard<std::mutex> lck(m_lkfs_mtx);
        kick_out_uncertain_lkfs(m_lkfs);
        most_certain_lkf = find_most_certain_lkf(m_lkfs);
      }

      /* Publish message of the most likely LKF (if found) //{ */
      if (most_certain_lkf != nullptr)
      {
        geometry_msgs::PoseWithCovarianceStamped msg = create_message(*most_certain_lkf, ros::Time::now());
        m_pub_localized_uav.publish(msg);

        if (m_pub_dbg_localized_uav.getNumSubscribers() > 0)
        {
          uav_localize::LocalizedUAV dbg_msg = to_dbg_message(msg, most_certain_lkf->id);
          m_pub_dbg_localized_uav.publish(dbg_msg);
        }
      }
      //}

      std::string most_certain_lkf_name = "none";
      if (most_certain_lkf != nullptr)
        most_certain_lkf_name = "#" + std::to_string(most_certain_lkf->id);
      ROS_INFO_STREAM_THROTTLE(1.0, "[" << m_node_name << "]: #LKFs: " << m_lkfs.size() << " | pub. LKF: " << most_certain_lkf_name);
    }
    //}

    /* lkf_update_loop() method //{ */
    void lkf_update_loop(const ros::TimerEvent& evt)
    {
      double dt = (evt.current_real - evt.last_real).toSec();
      lkf_A_t A = create_A(dt);
      lkf_R_t R = create_R(dt);

      {
        std::lock_guard<std::mutex> lck(m_lkfs_mtx);
        for (auto& lkf : m_lkfs)
        {
          lkf.setA(A);
          lkf.setR(R);
          lkf.iterateWithoutCorrection();
        }
      }
    }
    //}

  private:
    // --------------------------------------------------------------
    // |                ROS-related member variables                |
    // --------------------------------------------------------------

    /* Parameters, loaded from ROS //{ */
    double m_lkf_dt;
    double m_min_detection_height;
    std::string m_world_frame;
    std::string m_node_name;
    /* double m_xy_covariance_coeff; */
    /* double m_z_covariance_coeff; */
    /* double m_max_update_divergence; */
    /* double m_max_lkf_uncertainty; */
    /* double m_lkf_process_noise_pos; */
    /* double m_lkf_process_noise_vel; */
    /* double m_init_vel_cov; */
    /* int m_min_corrs_to_consider; */
    //}

    /* ROS related variables (subscribers, timers etc.) //{ */
    std::unique_ptr<drmgr_t> m_drmgr_ptr;
    tf2_ros::Buffer m_tf_buffer;
    std::unique_ptr<tf2_ros::TransformListener> m_tf_listener_ptr;
    mrs_lib::SubscribeHandlerPtr<uav_detect::Detections> m_sh_detections_ptr;
    mrs_lib::SubscribeHandlerPtr<uav_track::Trackings> m_sh_trackings_ptr;
    mrs_lib::SubscribeHandlerPtr<sensor_msgs::CameraInfo> m_sh_cinfo_ptr;
    ros::Publisher m_pub_localized_uav;
    ros::Publisher m_pub_dbg_localized_uav;
    ros::Timer m_lkf_update_loop_timer;
    ros::Timer m_process_loop_timer;
    ros::Timer m_publish_loop_timer;
    //}

  private:
    // --------------------------------------------------------------
    // |                helper implementation methods               |
    // --------------------------------------------------------------

    /* get_detections() method overloads //{ */
    const std::vector<uav_detect::Detection>& get_detections(const uav_detect::Detections& msg)
    {
      return msg.detections;
    }

    const std::vector<uav_track::Tracking>& get_detections(const uav_track::Trackings& msg)
    {
      return msg.trackings;
    }
    //}

    /* detection_to_measurement() method overloads //{ */
    measurement_t detection_to_measurement(const uav_detect::Detection& det)
    {
      measurement_t ret;

      /* calculate 3D position //{ */
      {
        const double u = det.x * det.roi.width + det.roi.x_offset;
        const double v = det.y * det.roi.height + det.roi.y_offset;
        const double x = (u - m_camera_model.cx() - m_camera_model.Tx()) / m_camera_model.fx();
        const double y = (v - m_camera_model.cy() - m_camera_model.Ty()) / m_camera_model.fy();
        ret.position << x, y, 1.0;
        ret.position *= det.depth;
      }
      //}

      /* calculate 3D covariance //{ */
      {
        ret.covariance = ret.covariance.Identity();
        const double xy_covariance_coeff = m_drmgr_ptr->config.depth_detections__xy_covariance_coeff;
        const double z_covariance_coeff = m_drmgr_ptr->config.depth_detections__z_covariance_coeff;
        Eigen::Matrix3d pos_cov = calc_position_covariance(ret.position, xy_covariance_coeff, z_covariance_coeff);
        ret.covariance.block<3, 3>(0, 0) = pos_cov;
      }
      //}

      return ret;
    }

    measurement_t detection_to_measurement(const uav_track::Tracking& trk)
    {
      measurement_t ret;

      /* calculate 3D position //{ */
      {
        const double u = trk.x * trk.roi.width + trk.roi.x_offset;
        const double v = trk.y * trk.roi.height + trk.roi.y_offset;
        const double x = (u - m_camera_model.cx() - m_camera_model.Tx()) / m_camera_model.fx();
        const double y = (v - m_camera_model.cy() - m_camera_model.Ty()) / m_camera_model.fy();
        ret.position << x, y, 1.0;
        ret.position *= trk.estimated_distance;
      }
      //}

      /* calculate 3D covariance //{ */
      {
        ret.covariance = ret.covariance.Identity();
        const double xy_covariance_coeff = m_drmgr_ptr->config.rgb_trackings__xy_covariance_coeff;
        const double z_covariance_coeff = m_drmgr_ptr->config.rgb_trackings__z_covariance_coeff;
        Eigen::Matrix3d pos_cov = calc_position_covariance(ret.position, xy_covariance_coeff, z_covariance_coeff);
        ret.covariance.block<3, 3>(0, 0) = pos_cov;
      }
      //}

      return ret;
    }
    //}

    /* measurements_from_message() method //{ */
    template <typename MessageType>
    std::vector<measurement_t> measurements_from_message(const MessageType& msg)
    {
      std::vector<measurement_t> ret;
      std::string sensor_frame = msg.header.frame_id;
      // Construct a new world to camera transform
      Eigen::Affine3d s2w_tf;
      bool tf_ok = get_transform_to_world(sensor_frame, msg.header.stamp, s2w_tf);

      if (!tf_ok)
        return ret;

      const auto& dets = get_detections(msg);
      ret.reserve(dets.size());

      /* Calculate 3D positions and covariances of the detections, push them to the output vector //{ */
      for (const auto& det : dets)
      {
        measurement_t measurement = detection_to_measurement(det);

        measurement.position = s2w_tf * measurement.position;
        if (!position_valid(measurement.position))
        {
          ROS_WARN_THROTTLE(1.0, "[%s]: Global position of detection [%.2f, %.2f, %.2f] is invalid (source: %s)!", m_node_name.c_str(), measurement.position(0),
                            measurement.position(1), measurement.position(2), get_msg_name(msg).c_str());
          continue;
        }

        measurement.covariance = rotate_covariance(measurement.covariance, s2w_tf.rotation());
        if (measurement.covariance.array().isNaN().any())
        {
          ROS_ERROR_THROTTLE(1.0, "[%s]: Constructed covariance of detection [%.2f, %.2f, %.2f] contains NaNs (source: %s)!", m_node_name.c_str(), measurement.position(0),
                             measurement.position(1), measurement.position(2), get_msg_name(msg).c_str());
          continue;
        }

        ret.push_back(measurement);
      }
      //}
      return ret;
    }
    //}

    /* update_lkfs() method //{ */
    // this method updates the LKFs with the supplied measurements and creates new ones
    // for new measurements
    void update_lkfs(const std::vector<measurement_t>& measurements, std::list<Lkf>& lkfs)
    {
      vector<int> meas_used(measurements.size(), 0);

      /* Assign a measurement to each LKF based on the smallest divergence and update the LKF //{ */
      for (auto& lkf : lkfs)
      {
        double divergence;
        size_t closest_it = find_closest_measurement(lkf, measurements, divergence);

        // Evaluate whether the divergence is small enough to justify the update
        if (divergence < m_drmgr_ptr->config.depth_detections__max_update_divergence)
        {
          Eigen::Vector3d closest_pos = measurements.at(closest_it).position;
          Eigen::Matrix3d closest_cov = measurements.at(closest_it).covariance;
          lkf.setMeasurement(closest_pos, closest_cov);
          lkf.doCorrection(true);
          meas_used.at(closest_it)++;
        }
      }
      //}

      /* Instantiate new LKFs for unused measurements (these are not considered as candidates for the most certain LKF) //{ */
      {
        for (size_t it = 0; it < measurements.size(); it++)
        {
          if (meas_used.at(it) < 1)
          {
            Lkf new_lkf = create_new_lkf(measurements.at(it), m_last_lkf_id, m_drmgr_ptr->config.init_vel_cov);
            m_lkfs.push_back(new_lkf);
            /* new_lkfs++; */
          }
        }
      }
      //}
    }
    //}

    /* kick_out_uncertain_lkfs() method //{ */
    void kick_out_uncertain_lkfs(std::list<Lkf>& lkfs) const
    {
      for (list<Lkf>::iterator it = std::begin(lkfs); it != std::end(lkfs); it++)
      {
        auto& lkf = *it;

        // First, check the uncertainty
        double uncertainty = calc_LKF_uncertainty(lkf);
        if (uncertainty > m_drmgr_ptr->config.max_lkf_uncertainty || std::isnan(uncertainty))
        {
          it = lkfs.erase(it);
          it--;
          /* kicked_out_lkfs++; */
        }
      }
    }
    //}

    /* find_most_certain_lkf() method //{ */
    Lkf const* find_most_certain_lkf(const std::list<Lkf>& lkfs) const
    {
      // the LKF must have received at least min_corrs_to_consider corrections
      // in order to be considered for the search of the most certain LKF
      int max_corrections = m_drmgr_ptr->config.min_corrs_to_consider;
      double picked_uncertainty = std::numeric_limits<double>::max();
      Lkf const* most_certain_lkf = nullptr;

      for (auto& lkf : lkfs)
      {
        double uncertainty = calc_LKF_uncertainty(lkf);

        // The LKF is picked if it has higher number of corrections than the found maximum.
        // If it has the same number of corrections as a previously found maximum then uncertainties are
        // compared to decide which is going to be picked.
        if (
            // current LKF has higher number of corrections as is the current max. found
            lkf.getNCorrections() > max_corrections
            // OR cur LKF has same number of corrections but lower uncertainty
            || (lkf.getNCorrections() == max_corrections && uncertainty < picked_uncertainty))
        {
          most_certain_lkf = &lkf;
          max_corrections = lkf.getNCorrections();
          picked_uncertainty = uncertainty;
        }
      }
      return most_certain_lkf;
    }
    //}

    /* get_transform_to_world() method //{ */
    bool get_transform_to_world(const string& frame_id, ros::Time stamp, Eigen::Affine3d& tf_out) const
    {
      try
      {
        const ros::Duration timeout(1.0 / 100.0);
        geometry_msgs::TransformStamped transform;
        // Obtain transform from snesor into world frame
        transform = m_tf_buffer.lookupTransform(m_world_frame, frame_id, stamp, timeout);

        // Obtain transform from camera frame into world
        tf_out = tf2::transformToEigen(transform.transform);
      }
      catch (tf2::TransformException& ex)
      {
        ROS_WARN_THROTTLE(1.0, "[%s]: Error during transform from \"%s\" frame to \"%s\" frame.\n\tMSG: %s", m_node_name.c_str(), frame_id.c_str(),
                          m_world_frame.c_str(), ex.what());
        return false;
      }
      return true;
    }
    //}

    /* position_valid() method //{ */
    bool position_valid(const Eigen::Vector3d& pos) const
    {
      return pos(2) > m_min_detection_height;
    }
    //}

    /* calc_position_covariance() method //{ */
    /* position_sf is position of the detection in 3D in the frame of the sensor (camera) */
    static Eigen::Matrix3d calc_position_covariance(const Eigen::Vector3d& position_sf, const double xy_covariance_coeff, const double z_covariance_coeff)
    {
      /* Calculates the corresponding covariance matrix of the estimated 3D position */
      Eigen::Matrix3d pos_cov = Eigen::Matrix3d::Identity();  // prepare the covariance matrix
      const double tol = 1e-9;
      pos_cov(0, 0) = pos_cov(1, 1) = xy_covariance_coeff;

      pos_cov(2, 2) = position_sf(2) * sqrt(position_sf(2)) * z_covariance_coeff;
      if (pos_cov(2, 2) < 0.33 * z_covariance_coeff)
        pos_cov(2, 2) = 0.33 * z_covariance_coeff;

      // Find the rotation matrix to rotate the covariance to point in the direction of the estimated position
      const Eigen::Vector3d a(0.0, 0.0, 1.0);
      const Eigen::Vector3d b = position_sf.normalized();
      const Eigen::Vector3d v = a.cross(b);
      const double sin_ab = v.norm();
      const double cos_ab = a.dot(b);
      Eigen::Matrix3d vec_rot = Eigen::Matrix3d::Identity();
      if (sin_ab < tol)  // unprobable, but possible - then it is identity or 180deg
      {
        if (cos_ab + 1.0 < tol)  // that would be 180deg
        {
          vec_rot << -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0;
        }     // otherwise its identity
      } else  // otherwise just construct the matrix
      {
        Eigen::Matrix3d v_x;
        v_x << 0.0, -v(2), v(1), v(2), 0.0, -v(0), -v(1), v(0), 0.0;
        vec_rot = Eigen::Matrix3d::Identity() + v_x + (1 - cos_ab) / (sin_ab * sin_ab) * (v_x * v_x);
      }
      pos_cov = rotate_covariance(pos_cov, vec_rot);  // rotate the covariance to point in direction of est. position
      return pos_cov;
    }
    //}

    /* get_msg_name() method overloads //{ */
    inline static std::string get_msg_name([[maybe_unused]] const uav_detect::Detections& msg)
    {
      return "uav_detect::Detections";
    }
    inline static std::string get_msg_name([[maybe_unused]] const uav_track::Trackings& msg)
    {
      return "uav_track::Trackings";
    }
    //}

    /* rotate_covariance() method //{ */
    inline static Eigen::Matrix3d rotate_covariance(const Eigen::Matrix3d& covariance, const Eigen::Matrix3d& rotation)
    {
      return rotation * covariance * rotation.transpose();  // rotate the covariance to point in direction of est. position
    }
    //}

    /* calc_LKF_uncertainty() method //{ */
    static double calc_LKF_uncertainty(const Lkf& lkf)
    {
      Eigen::Matrix3d position_covariance = lkf.getCovariance().block<3, 3>(0, 0);
      double determinant = position_covariance.determinant();
      return sqrt(determinant);
    }
    //}

    /* calc_LKF_meas_divergence() method //{ */
    static double calc_LKF_meas_divergence(const Eigen::Vector3d& mu0, const Eigen::Matrix3d& sigma0, const Eigen::Vector3d& mu1, const Eigen::Matrix3d& sigma1)
    {
      return kullback_leibler_divergence(mu0, sigma0, mu1, sigma1);
    }
    //}

    /* find_closest_measurement() method //{ */
    /* returns position of the closest measurement in the pos_covs vector */
    static size_t find_closest_measurement(const Lkf& lkf, const std::vector<measurement_t>& pos_covs, double& min_divergence_out)
    {
      const Eigen::Vector3d& lkf_pos = lkf.getStates().block<3, 1>(0, 0);
      const Eigen::Matrix3d& lkf_cov = lkf.getCovariance().block<3, 3>(0, 0);
      double min_divergence = std::numeric_limits<double>::max();
      size_t min_div_it = 0;

      // Find measurement with smallest divergence from this LKF and assign the measurement to it
      for (size_t it = 0; it < pos_covs.size(); it++)
      {
        const auto& pos_cov = pos_covs.at(it);
        if (pos_cov.covariance.array().isNaN().any())
          ROS_ERROR("Covariance of LKF contains NaNs!");
        const Eigen::Vector3d& det_pos = pos_cov.position;
        const Eigen::Matrix3d& det_cov = pos_cov.covariance;
        const double divergence = calc_LKF_meas_divergence(det_pos, det_cov, lkf_pos, lkf_cov);

        if (divergence < min_divergence)
        {
          min_divergence = divergence;
          min_div_it = it;
        }
      }
      min_divergence_out = min_divergence;
      return min_div_it;
    }
    //}

    /* create_message() method //{ */
    geometry_msgs::PoseWithCovarianceStamped create_message(const Lkf& lkf, ros::Time stamp) const
    {
      geometry_msgs::PoseWithCovarianceStamped msg;

      msg.header.frame_id = m_world_frame;
      msg.header.stamp = stamp;

      {
        const Eigen::Vector3d position = lkf.getStates().block<3, 1>(0, 0);
        msg.pose.pose.position.x = position(0);
        msg.pose.pose.position.y = position(1);
        msg.pose.pose.position.z = position(2);
      }

      msg.pose.pose.orientation.w = 1.0;

      {
        const Eigen::Matrix3d covariance = lkf.getCovariance().block<3, 3>(0, 0);
        for (int r = 0; r < 6; r++)
        {
          for (int c = 0; c < 6; c++)
          {
            if (r < 3 && c < 3)
              msg.pose.covariance[r * 6 + c] = covariance(r, c);
            else if (r == c)
              msg.pose.covariance[r * 6 + c] = 666;
          }
        }
      }

      return msg;
    }
    //}

    /* to_dbg_message() method //{ */
    static uav_localize::LocalizedUAV to_dbg_message(const geometry_msgs::PoseWithCovarianceStamped& orig_msg, uint32_t lkf_id)
    {
      uav_localize::LocalizedUAV msg;

      msg.header = orig_msg.header;
      msg.position.x = orig_msg.pose.pose.position.x;
      msg.position.y = orig_msg.pose.pose.position.y;
      msg.position.z = orig_msg.pose.pose.position.z;
      msg.lkf_id = lkf_id;

      return msg;
    }
    //}

  private:
    // --------------------------------------------------------------
    // |                  general member variables                  |
    // --------------------------------------------------------------

    /* camera model member variable //{ */
    image_geometry::PinholeCameraModel m_camera_model;
    //}

  private:
    /* LKF - related member variables //{ */
    std::mutex m_lkfs_mtx;  // mutex for synchronization of the m_lkfs variable
    std::list<Lkf> m_lkfs;  // all currently active LKFs
    int m_last_lkf_id;      // ID of the last created LKF - used when creating a new LKF to generate a new unique ID
    //}

    /* Definitions of the LKF (consts, typedefs, etc.) //{ */
    static const int c_n_states = 6;
    static const int c_n_inputs = 0;
    static const int c_n_measurements = 3;

    typedef Eigen::Matrix<double, c_n_states, 1> lkf_x_t;
    typedef Eigen::Matrix<double, c_n_inputs, 1> lkf_u_t;
    typedef Eigen::Matrix<double, c_n_measurements, 1> lkf_z_t;

    typedef Eigen::Matrix<double, c_n_states, c_n_states> lkf_A_t;
    typedef Eigen::Matrix<double, c_n_states, c_n_inputs> lkf_B_t;
    typedef Eigen::Matrix<double, c_n_measurements, c_n_states> lkf_P_t;
    typedef Eigen::Matrix<double, c_n_states, c_n_states> lkf_R_t;
    typedef Eigen::Matrix<double, c_n_measurements, c_n_measurements> lkf_Q_t;

    inline static lkf_A_t create_A(double dt)
    {
      lkf_A_t A;
      A << 1, 0, 0, dt, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1;
      return A;
    }

    inline static lkf_P_t create_P()
    {
      lkf_P_t P;
      P << 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0;
      return P;
    }

    inline lkf_R_t create_R(double dt) const
    {
      lkf_R_t R = lkf_R_t::Identity();
      R.block<3, 3>(0, 0) *= dt * m_drmgr_ptr->config.lkf_process_noise_pos;
      R.block<3, 3>(3, 3) *= dt * m_drmgr_ptr->config.lkf_process_noise_vel;
      return R;
    }
    //}

    /* create_new_lkf() method //{ */
    static Lkf create_new_lkf(const measurement_t& initialization, int& last_lkf_id, const double init_vel_cov)
    {
      const lkf_A_t A;  // changes in dependence on the measured dt, so leave blank for now
      const lkf_B_t B;  // zero rows zero cols matrix
      const lkf_P_t P = create_P();
      const lkf_R_t R;  // depends on the measured dt, so leave blank for now
      const lkf_Q_t Q;  // depends on the measurement, so leave blank for now

      Lkf new_lkf(last_lkf_id, LocalizeSingle::c_n_states, LocalizeSingle::c_n_inputs, LocalizeSingle::c_n_measurements, A, B, R, Q, P);

      // Initialize the LKF using the new measurement
      lkf_x_t init_state;
      init_state.block<3, 1>(0, 0) = initialization.position;
      init_state.block<3, 1>(3, 0) = Eigen::Vector3d::Zero();
      lkf_R_t init_state_cov;
      init_state_cov.setZero();
      init_state_cov.block<3, 3>(0, 0) = initialization.covariance;
      init_state_cov.block<3, 3>(3, 3) = init_vel_cov * Eigen::Matrix3d::Identity();

      new_lkf.setStates(init_state);
      new_lkf.setCovariance(init_state_cov);
      return new_lkf;
    }
    //}

  private:
    // --------------------------------------------------------------
    // |        detail implementation methods (maybe unused)        |
    // --------------------------------------------------------------

    /* kullback_leibler_divergence() method //{ */
    // This method calculates the kullback-leibler divergence of two three-dimensional normal distributions.
    // It is used for deciding which measurement to use for which LKF.
    static double kullback_leibler_divergence(const Eigen::Vector3d& mu0, const Eigen::Matrix3d& sigma0, const Eigen::Vector3d& mu1,
                                              const Eigen::Matrix3d& sigma1)
    {
      const unsigned k = 3;  // number of dimensions -- DON'T FORGET TO CHANGE IF NUMBER OF DIMENSIONS CHANGES!
      const double div = 0.5
                         * ((sigma1.inverse() * sigma0).trace() + (mu1 - mu0).transpose() * (sigma1.inverse()) * (mu1 - mu0) - k
                            + log((sigma1.determinant()) / sigma0.determinant()));
      return div;
    }
    //}

  };  // class LocalizeSingle : public nodelet::Nodelet
};    // namespace uav_localize

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(uav_localize::LocalizeSingle, nodelet::Nodelet)
