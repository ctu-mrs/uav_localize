#include "main.h"

#include <Eigen/Geometry>

#include <nodelet/nodelet.h>

#include "Measurement.h"
#include "Hypothesis.h"

#include <uav_localize/LocalizationParamsConfig.h>
#include <uav_localize/LocalizationHypotheses.h>
#include <sensor_msgs/PointCloud.h>

using namespace std;

// shortcut type to the dynamic reconfigure manager template instance
typedef mrs_lib::DynamicReconfigureMgr<uav_localize::LocalizationParamsConfig> drmgr_t;

namespace uav_localize
{
  class LocalizeSingle : public nodelet::Nodelet
  {
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
      NODELET_INFO("[LocalizeSingle]: Loading static parameters:");
      pl.loadParam("world_frame", m_world_frame, std::string("local_origin"));
      const int process_loop_rate = pl.loadParam2<int>("process_loop_rate");
      const int publish_loop_rate = pl.loadParam2<int>("publish_loop_rate");
      const int info_loop_rate = pl.loadParam2<int>("info_loop_rate");
      pl.loadParam("min_detection_height", m_min_detection_height);
      if (!pl.loadedSuccessfully())
      {
        NODELET_ERROR("[LocalizeSingle]: Some compulsory parameters were not loaded successfully, ending the node");
        ros::shutdown();
      }

      // LOAD DYNAMIC PARAMETERS
      m_drmgr_ptr = make_unique<drmgr_t>(nh, m_node_name);
      if (!m_drmgr_ptr->loaded_successfully())
      {
        NODELET_ERROR("[LocalizeSingle]: Some default values of dynamically reconfigurable parameters were not loaded successfully, ending the node:");
        ros::shutdown();
      }
      
      //}

      /* Create publishers and subscribers //{ */
      // Initialize transform listener
      m_tf_listener_ptr = std::make_unique<tf2_ros::TransformListener>(m_tf_buffer);
      // Subscribers
      mrs_lib::SubscribeHandlerOptions shopts;
      shopts.no_message_timeout = ros::Duration(5.0);
      mrs_lib::construct_object(m_sh_depth_detections, shopts, "detections");
      mrs_lib::construct_object(m_sh_cnn_detections, shopts, "cnn_detections");
      /* mrs_lib::construct_object(m_sh_trackings, shopts, "trackings"); */
      mrs_lib::construct_object(m_sh_cinfo, shopts, "camera_info");
      // Publishers
      m_pub_localized_uav = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("localized_uav", 10);
      m_pub_dgb_hypotheses = nh.advertise<uav_localize::LocalizationHypotheses>("dbg_hypotheses", 10);
      m_pub_dgb_pcl_hyps = nh.advertise<sensor_msgs::PointCloud>("dbg_hypotheses_pcl", 10);
      m_pub_dgb_pcl_meas = nh.advertise<sensor_msgs::PointCloud>("dbg_measurements_pcl", 10);
      //}

      /* Initialize other variables //{ */
      m_depth_detections = 0;
      m_cnn_detections = 0;
      m_rgb_trackings = 0;
      m_most_certain_hyp_name = "none";

      m_last_hyp_id = -1;
      //}

      /* Initialize timers //{ */
      /* m_lkf_update_loop_timer = nh.createTimer(ros::Rate(update_loop_rate), &LocalizeSingle::lkf_update_loop, this); */
      m_process_loop_timer = nh.createTimer(ros::Rate(process_loop_rate), &LocalizeSingle::process_loop, this);
      m_publish_loop_timer = nh.createTimer(ros::Rate(publish_loop_rate), &LocalizeSingle::publish_loop, this);
      m_info_loop_timer = nh.createTimer(ros::Rate(info_loop_rate), &LocalizeSingle::info_loop, this);
      //}

      std::cout << "----------------------------------------------------------" << std::endl;
    }
    //}

    /* process_loop() method //{ */
    void process_loop([[maybe_unused]] const ros::TimerEvent& evt)
    {
      const bool got_depth_detections = m_sh_depth_detections.newMsg();
      const bool got_cnn_detections = m_sh_cnn_detections.newMsg();
      /* const bool got_rgb_tracking = m_sh_trackings_ptr.newMsg(); */
      const bool got_rgb_tracking = false;
      const bool got_something_to_process = got_depth_detections || got_cnn_detections || got_rgb_tracking;
      const bool should_process = got_something_to_process && m_sh_cinfo.hasMsg();
      if (should_process)
      {
        /* Initialize the camera model if not initialized yet //{ */
        if (!m_sh_cinfo.usedMsg())
          m_camera_model.fromCameraInfo(m_sh_cinfo.getMsg());
        //}

        if (got_depth_detections)
        {
          /* Update the hypotheses using this message //{ */
          {
            const uav_detect::DetectionsConstPtr last_detections_msg = m_sh_depth_detections.getMsg();
            const std::vector<Measurement> measurements = measurements_from_message(last_detections_msg);
            /* if (measurements.empty()) */
            /*   NODELET_WARN("Received empty message from source %s", get_msg_name(last_detections_msg).c_str()); */
            {
              std::lock_guard<std::mutex> lck(m_hyps_mtx);
              update_hyps(measurements, m_hyps);
            }

            /* Publish debug pointcloud message of measurements //{ */
            if (m_pub_dgb_pcl_meas.getNumSubscribers() > 0)
            {
              sensor_msgs::PointCloudConstPtr pcl_msg = create_pcl_message(measurements, last_detections_msg->header.stamp);
              m_pub_dgb_pcl_meas.publish(pcl_msg);
            }
            //}
          }
          //}

          /* Update the number of received depth detections //{ */
          {
            std::lock_guard<std::mutex> lck(m_stat_mtx);
            m_depth_detections++;
          }
          //}
        }

        if (got_cnn_detections)
        {
          /* Update the hypotheses using this message //{ */
          {
            const cnn_detect::DetectionsConstPtr last_detections_msg = m_sh_cnn_detections.getMsg();
            const std::vector<Measurement> measurements = measurements_from_message(last_detections_msg);
            /* if (measurements.empty()) */
            /*   NODELET_WARN("Received empty message from source %s", get_msg_name(last_detections_msg).c_str()); */
            {
              std::lock_guard<std::mutex> lck(m_hyps_mtx);
              update_hyps(measurements, m_hyps);
            }

            /* Publish debug pointcloud message of measurements //{ */
            if (m_pub_dgb_pcl_meas.getNumSubscribers() > 0)
            {
              sensor_msgs::PointCloudConstPtr pcl_msg = create_pcl_message(measurements, last_detections_msg->header.stamp);
              m_pub_dgb_pcl_meas.publish(pcl_msg);
            }
            //}
          }
          //}

          /* Update the number of received depth detections //{ */
          {
            std::lock_guard<std::mutex> lck(m_stat_mtx);
            m_cnn_detections++;
          }
          //}
        }

        /* if (got_rgb_tracking) */
        /* { */
        /*   /1* Update the hypotheses using this message //{ *1/ */
        /*   { */
        /*     const uav_track::TrackingsConstPtr last_trackings_msg = m_sh_trackings_ptr.getMsg(); */
        /*     const std::vector<Measurement> measurements = measurements_from_message(last_trackings_msg); */
        /*     /1* if (measurements.empty()) *1/ */
        /*     /1*   NODELET_WARN("Received empty message from source %s", get_msg_name(last_trackings_msg).c_str()); *1/ */
        /*     { */
        /*       std::lock_guard<std::mutex> lck(m_hyps_mtx); */
        /*       update_hyps(measurements, m_hyps); */
        /*     } */
        /*   } */
        /*   //} */

        /*   /1* Update the number of received rgb trackings //{ *1/ */
        /*   { */
        /*     std::lock_guard<std::mutex> lck(m_stat_mtx); */
        /*     m_rgb_trackings++; */
        /*   } */
        /*   //} */
        /* } */
      }
    }
    //}

    /* publish_loop() method //{ */
    void publish_loop([[maybe_unused]] const ros::TimerEvent& evt)
    {
      ros::Time stamp = ros::Time::now();

      /* for (const auto& hyp : m_hyps) */
      /* { */
      /*   cout << "#" << hyp.id << " l: " << hyp.get_loglikelihood() << endl; */
      /* } */

      /* Find the most certain hypothesis //{ */
      Hypothesis const* most_certain_hyp = nullptr;
      {
        std::lock_guard<std::mutex> lck(m_hyps_mtx);
        predict_to(m_hyps, stamp);
        kick_out_uncertain_hyps(m_hyps);
        most_certain_hyp = find_most_certain_hyp(m_hyps);
      }
      //}

      /* Publish message of the most certain hypothesis (if found) //{ */
      if (most_certain_hyp != nullptr)
      {
        geometry_msgs::PoseWithCovarianceStampedConstPtr msg = create_message(*most_certain_hyp, stamp);
        /* m_tf_buffer.transform(*msg, *msg, "rs_d435_color_optical_frame", ros::Duration(0.01)); */
        m_pub_localized_uav.publish(msg);
      }
      //}

      /* Publish debug message of hypotheses //{ */
      if (m_pub_dgb_hypotheses.getNumSubscribers() > 0)
      {
        std::lock_guard<std::mutex> lck(m_hyps_mtx);
        int hyp_id = most_certain_hyp == nullptr ? -1 : most_certain_hyp->id;
        uav_localize::LocalizationHypothesesConstPtr dbg_msg = create_dbg_message(m_hyps, hyp_id, stamp);
        m_pub_dgb_hypotheses.publish(dbg_msg);
      }
      //}

      /* Publish debug pointcloud message of hypotheses //{ */
      if (m_pub_dgb_pcl_hyps.getNumSubscribers() > 0)
      {
        std::lock_guard<std::mutex> lck(m_hyps_mtx);
        int hyp_id = most_certain_hyp == nullptr ? -1 : most_certain_hyp->id;
        sensor_msgs::PointCloudConstPtr pcl_msg = create_pcl_message(m_hyps, hyp_id, stamp);
        m_pub_dgb_pcl_hyps.publish(pcl_msg);
      }
      //}
      
      /* Update the name of the most certin hypothesis to be displayed //{ */
      {
        std::lock_guard<std::mutex> lck(m_stat_mtx);
        if (most_certain_hyp == nullptr)
          m_most_certain_hyp_name = "none";
        else
          m_most_certain_hyp_name = "#" + std::to_string(most_certain_hyp->id);
      }
      //}
    }
    //}

    /* /1* lkf_update_loop() method //{ *1/ */
    /* void lkf_update_loop(const ros::TimerEvent& evt) */
    /* { */
    /*   // Iterate LKFs in all hypotheses */
    /*   std::lock_guard<std::mutex> lck(m_hyps_mtx); */
    /*   for (auto& hyp : m_hyps) */
    /*   { */
    /*     hyp.prediction_step(evt.current_real, m_drmgr_ptr->config.lkf_process_noise_pos,  m_drmgr_ptr->config.lkf_process_noise_vel); */
    /*   } */
    /* } */
    /* //} */

    /* info_loop() method //{ */
    void info_loop([[maybe_unused]] const ros::TimerEvent& evt)
    {
      const float dt = (evt.current_real - evt.last_real).toSec();
      float depth_detections_rate;
      float cnn_detections_rate;
      float rgb_trackings_rate;
      int n_hypotheses;
      std::string most_certain_hyp_name;
      {
        std::lock_guard<std::mutex> lck(m_stat_mtx);
        depth_detections_rate = round(m_depth_detections / dt);
        cnn_detections_rate = round(m_cnn_detections / dt);
        rgb_trackings_rate = round(m_rgb_trackings / dt);
        most_certain_hyp_name = m_most_certain_hyp_name;
        m_depth_detections = 0;
        m_cnn_detections = 0;
        m_rgb_trackings = 0;
      }
      {
        std::lock_guard<std::mutex> lck(m_hyps_mtx);
        n_hypotheses = m_hyps.size();
      }
      NODELET_INFO_STREAM("[" << m_node_name << "]: det. rate: " << round(depth_detections_rate) << "/" << round(cnn_detections_rate) << " Hz | trk. rate: " << round(rgb_trackings_rate)
                          << " Hz | #hyps: " << n_hypotheses << " | pub. hyp.: " << most_certain_hyp_name);
    }
    //}

  private:
    // --------------------------------------------------------------
    // |                ROS-related member variables                |
    // --------------------------------------------------------------

    /* Parameters, loaded from ROS //{ */
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
    mrs_lib::SubscribeHandler<uav_detect::Detections> m_sh_depth_detections;
    mrs_lib::SubscribeHandler<cnn_detect::Detections> m_sh_cnn_detections;
    /* mrs_lib::SubscribeHandlerPtr<uav_track::Trackings> m_sh_trackings_ptr; */
    mrs_lib::SubscribeHandler<sensor_msgs::CameraInfo> m_sh_cinfo;
    ros::Publisher m_pub_localized_uav;
    ros::Publisher m_pub_dgb_hypotheses;
    ros::Publisher m_pub_dgb_pcl_hyps;
    ros::Publisher m_pub_dgb_pcl_meas;
    /* ros::Timer m_lkf_update_loop_timer; */
    ros::Timer m_process_loop_timer;
    ros::Timer m_publish_loop_timer;
    ros::Timer m_info_loop_timer;
    //}

  private:
    // --------------------------------------------------------------
    // |                helper implementation methods               |
    // --------------------------------------------------------------

    /* get_detections() method overloads //{ */
    const std::vector<uav_detect::Detection>& get_detections(const uav_detect::DetectionsConstPtr& msg)
    {
      return msg->detections;
    }

    const std::vector<cnn_detect::Detection>& get_detections(const cnn_detect::DetectionsConstPtr& msg)
    {
      return msg->detections;
    }

    /* const std::vector<uav_track::Tracking>& get_detections(const uav_track::TrackingsConstPtr& msg) */
    /* { */
    /*   return msg->trackings; */
    /* } */
    //}

    /* detection_to_measurement() method //{ */
    /* position_from_detection() method //{ */
    float get_depth(const uav_detect::Detection& det)
    {
      return det.depth;
    }
    float get_depth(const cnn_detect::Detection& det)
    {
      return det.depth;
    }
    /* float get_depth(const uav_track::Tracking& det) */
    /* { */
    /*   return det.estimated_depth; */
    /* } */

    template <typename Detection>
    Eigen::Vector3d position_from_detection(const Detection& det)
    {
      const double u = det.x * det.roi.width + det.roi.x_offset;
      const double v = det.y * det.roi.height + det.roi.y_offset;
      const double x = (u - m_camera_model.cx() - m_camera_model.Tx()) / m_camera_model.fx();
      const double y = (v - m_camera_model.cy() - m_camera_model.Ty()) / m_camera_model.fy();
      Eigen::Vector3d ret(x, y, 1.0);
      ret *= get_depth(det);
      return ret;
    }
    //}

    /* covariance_from_detection() method overloads //{ */
    Eigen::Matrix3d covariance_from_detection([[maybe_unused]] const uav_detect::Detection& det, const Eigen::Vector3d& position)
    {
      Eigen::Matrix3d ret;
      const double xy_covariance_coeff = m_drmgr_ptr->config.depth_detections__xy_covariance_coeff;
      const double z_covariance_coeff = m_drmgr_ptr->config.depth_detections__z_covariance_coeff;
      ret = calc_position_covariance(position, xy_covariance_coeff, z_covariance_coeff);
      return ret;
    }
    Eigen::Matrix3d covariance_from_detection([[maybe_unused]] const cnn_detect::Detection& det, const Eigen::Vector3d& position)
    {
      Eigen::Matrix3d ret;
      const double xy_covariance_coeff = m_drmgr_ptr->config.cnn_detections__xy_covariance_coeff;
      const double z_covariance_coeff = m_drmgr_ptr->config.cnn_detections__z_covariance_coeff;
      ret = calc_position_covariance(position, xy_covariance_coeff, z_covariance_coeff);
      return ret;
    }
    /* Eigen::Matrix3d covariance_from_detection([[maybe_unused]] const uav_track::Tracking& det, const Eigen::Vector3d& position) */
    /* { */
    /*   Eigen::Matrix3d ret; */
    /*   const double xy_covariance_coeff = m_drmgr_ptr->config.depth_detections__xy_covariance_coeff; */
    /*   const double z_covariance_coeff = m_drmgr_ptr->config.depth_detections__z_covariance_coeff; */
    /*   ret = calc_position_covariance(position, xy_covariance_coeff, z_covariance_coeff); */
    /*   return ret; */
    /* } */
    //}

    /*  detection_source() method overloads//{ */
    Measurement::source_t source_of_detection([[maybe_unused]] const uav_detect::Detection&)
    {
      return Measurement::source_t::depth_detection;
    }
    Measurement::source_t source_of_detection([[maybe_unused]] const cnn_detect::Detection&)
    {
      return Measurement::source_t::cnn_detection;
    }
    /* Measurement::source_t source_of_detection([[maybe_unused]] const uav_track::Tracking&) */
    /* { */
    /*   return Measurement::source_t::rgb_tracking; */
    /* } */
    //}

    template <typename Detection>
    Measurement detection_to_measurement(const Detection& det)
    {
      Measurement ret;
      ret.position = position_from_detection(det);
      ret.covariance = covariance_from_detection(det, ret.position);
      ret.source = source_of_detection(det);
      return ret;
    }
    //}

    /* measurements_from_message() method //{ */
    template <typename MessageType>
    std::vector<Measurement> measurements_from_message(const MessageType& msg)
    {
      std::vector<Measurement> ret;
      const ros::Time msg_stamp = msg->header.stamp;
      const std::string sensor_frame = msg->header.frame_id;
      // Construct a new world to camera transform
      Eigen::Affine3d s2w_tf;
      bool tf_ok = get_transform_to_world(sensor_frame, msg_stamp, s2w_tf);

      if (!tf_ok)
        return ret;

      const auto& dets = get_detections(msg);
      ret.reserve(dets.size());

      /* Construct the measurements, push them to the output vector //{ */
      for (const auto& det : dets)
      {
        Measurement measurement = detection_to_measurement(det);
        measurement.stamp = msg->header.stamp;

        measurement.position = s2w_tf * measurement.position;
        if (!position_valid(measurement.position))
        {
          NODELET_WARN_THROTTLE(1.0, "[%s]: Global position of detection [%.2f, %.2f, %.2f] is invalid (source: %s)!", m_node_name.c_str(), measurement.position(0),
                            measurement.position(1), measurement.position(2), get_msg_name(msg).c_str());
          continue;
        }

        measurement.covariance = rotate_covariance(measurement.covariance, s2w_tf.rotation());
        if (measurement.covariance.array().isNaN().any())
        {
          NODELET_ERROR_THROTTLE(1.0, "[%s]: Constructed covariance of detection [%.2f, %.2f, %.2f] contains NaNs (source: %s)!", m_node_name.c_str(),
                             measurement.position(0), measurement.position(1), measurement.position(2), get_msg_name(msg).c_str());
          continue;
        }

        ret.push_back(measurement);
      }
      //}
      return ret;
    }
    //}

    /* inflate_covariance() method //{ */
    Measurement inflate_covariance(Measurement meas)
    {
      const Measurement::source_t source = meas.source;
      double coeff;
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
      switch (source)
      {
        case Measurement::source_t::depth_detection:
          coeff = m_drmgr_ptr->config.depth_detections__cov_inflation_coeff;
          break;
        case Measurement::source_t::cnn_detection:
          coeff = m_drmgr_ptr->config.cnn_detections__cov_inflation_coeff;
          break;
        case Measurement::source_t::rgb_tracking:
          coeff = m_drmgr_ptr->config.rgb_trackings__cov_inflation_coeff;
          break;
        case Measurement::source_t::lkf_prediction:
          coeff = std::numeric_limits<double>::max(); // this should not happen...
          break;
      }
#pragma GCC diagnostic pop
      meas.covariance *= coeff;
      return meas;
    }
    //}

    /* update_hyps() method //{ */
    // this method updates the hypothesis with the supplied measurements and creates new ones
    // for new measurements
    void update_hyps(const std::vector<Measurement>& measurements, std::list<Hypothesis>& hyps)
    {
      vector<int> meas_used(measurements.size(), 0);

      /* Assign a measurement to each LKF based on the smallest likelihood and update the LKF //{ */
      for (auto& hyp : hyps)
      {
        const auto [closest_it, loglikelihood] = find_closest_measurement(hyp, measurements);

        // Evaluate whether the likelihood is small enough to justify the update
        if (closest_it >= 0)
        {
          const auto meas = inflate_covariance(measurements.at(closest_it));
          hyp.correction_step(meas, loglikelihood);
          meas_used.at(closest_it)++;
          NODELET_DEBUG("[LocalizeSingle]: Updated hypothesis ID%d using measurement from %s (l: %f)", hyp.id, measurements.at(closest_it).source_name().c_str(), exp(loglikelihood));
        }
      }
      //}

      /* Instantiate new hypotheses for unused reliable measurements //{ */
      {
        int new_hyps = 0;
        for (size_t it = 0; it < measurements.size(); it++)
        {
          if (meas_used.at(it) < 1 && measurements.at(it).reliable())
          {
            Hypothesis new_hyp(++m_last_hyp_id, measurements.at(it), m_drmgr_ptr->config.init_vel_cov, m_drmgr_ptr->config.lkf_process_noise_pos, m_drmgr_ptr->config.lkf_process_noise_vel, 100);
            m_hyps.push_back(new_hyp);
            new_hyps++;
          }
        }
        if (!measurements.empty())
          NODELET_DEBUG("[LocalizeSingle]: Created %d new hypotheses for total of %lu (got %lu new measurements from %s)", new_hyps, m_hyps.size(),
                        measurements.size(), measurements.at(0).source_name().c_str());
      }
      //}
    }
    //}

    /* predict_to() method //{ */
    // this method predicts the hypotheses to the specified time stamp
    void predict_to(std::list<Hypothesis>& hyps, ros::Time stamp)
    {
      const double lkf_process_noise_pos = m_drmgr_ptr->config.lkf_process_noise_pos ;
      const double lkf_process_noise_vel = m_drmgr_ptr->config.lkf_process_noise_vel ;
      for (auto& hyp : hyps)
      {
        hyp.predict_to(stamp, lkf_process_noise_pos, lkf_process_noise_vel);
      }
    }
    //}

    /* kick_out_uncertain_hyps() method //{ */
    void kick_out_uncertain_hyps(std::list<Hypothesis>& hyps) const
    {
      int kicked_out_hyps = 0;
      for (list<Hypothesis>::iterator it = std::begin(hyps); it != std::end(hyps); it++)
      {
        auto& hyp = *it;

        // First, check the uncertainty
        double uncertainty = calc_hyp_uncertainty(hyp);
        if (uncertainty > m_drmgr_ptr->config.max_hyp_uncertainty || std::isnan(uncertainty))
        {
          it = hyps.erase(it);
          it--;
          kicked_out_hyps++;
        }
      }
      NODELET_DEBUG("[LocalizeSingle]: Kicked out %d new hypotheses for total of %lu", kicked_out_hyps , m_hyps.size());
    }
    //}

    /* find_most_certain_hyp() method //{ */
    Hypothesis const* find_most_certain_hyp(const std::list<Hypothesis>& hyps) const
    {
      // the LKF must have received at least min_corrs_to_consider corrections
      // in order to be considered for the search of the most certain hypothesis
      int max_corrections = m_drmgr_ptr->config.min_corrs_to_consider;
      double picked_uncertainty = std::numeric_limits<double>::max();
      Hypothesis const* most_certain_hyp = nullptr;

      for (auto& hyp : hyps)
      {
        double uncertainty = calc_hyp_uncertainty(hyp);

        // The hypothesis is picked if it has higher number of corrections than the found maximum.
        // If it has the same number of corrections as a previously found maximum then uncertainties are
        // compared to decide which is going to be picked.
        if (
            // current hypothesis has higher number of corrections as is the current max. found
            hyp.get_n_corrections() > max_corrections
            // OR cur hypothesis has same number of corrections but lower uncertainty
            || (hyp.get_n_corrections() == max_corrections && uncertainty < picked_uncertainty))
        {
          most_certain_hyp = &hyp;
          max_corrections = hyp.get_n_corrections();
          picked_uncertainty = uncertainty;
        }
      }
      return most_certain_hyp;
    }
    //}

    /* get_transform_to_world() method //{ */
    bool get_transform_to_world(const string& frame_id, ros::Time stamp, Eigen::Affine3d& tf_out) const
    {
      try
      {
        const ros::Duration timeout(1.0 / 10.0);
        geometry_msgs::TransformStamped transform;
        // Obtain transform from snesor into world frame
        transform = m_tf_buffer.lookupTransform(m_world_frame, frame_id, stamp, timeout);

        // Obtain transform from camera frame into world
        tf_out = tf2::transformToEigen(transform.transform);
      }
      catch (tf2::TransformException& ex)
      {
        NODELET_WARN_THROTTLE(1.0, "[%s]: Error during transform from \"%s\" frame to \"%s\" frame.\n\tMSG: %s", m_node_name.c_str(), frame_id.c_str(),
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
    /* Calculates the corresponding covariance matrix of the estimated 3D position.
     * The covariance is first constructed so that it has standard deviation 'xy_covariance_coeff'
     * in the x and y directions and 'z_covariance_coeff' scaled by the distance in the z direction.
     * Then it is rotated so that the former z direction of the covariance ellipsoid points in
     * the direction of the measurement 'position_sf'.
     * 'position_sf' is position of the detection in 3D in the frame of the sensor (camera).
     * */
    static Eigen::Matrix3d calc_position_covariance(const Eigen::Vector3d& position_sf, const double xy_covariance_coeff, const double z_covariance_coeff)
    {
      Eigen::Matrix3d pos_cov;
      /* Prepare the covariance matrix according to the supplied parameters //{ */
      pos_cov = Eigen::Matrix3d::Identity();
      pos_cov(0, 0) = pos_cov(1, 1) = xy_covariance_coeff;
      // the z coefficient is scaled according to the distance of the measurement d^1.5
      pos_cov(2, 2) = position_sf(2) * sqrt(abs(position_sf(2))) * z_covariance_coeff;
      // however, there is a lower limit to the covariance
      if (pos_cov(2, 2) < 0.33 * z_covariance_coeff)
        pos_cov(2, 2) = 0.33 * z_covariance_coeff;
      //}

      /* Rotate the covariance matri so that it points its z axis towards the measurement position //{ */
      // Prepare some helper vectors and variables
      const Eigen::Vector3d a(0.0, 0.0, 1.0);
      const Eigen::Vector3d b = position_sf.normalized();
      const Eigen::Vector3d v = a.cross(b);           // the axis of rotation
      const double sin_ab = v.norm();                 // sine of the angle between 'a' and the position vector
      const double cos_ab = a.dot(b);                 // cosine of the angle between 'a' and the position vector
      const double angle = atan2(sin_ab, cos_ab);     // the desired rotation angle
      const Eigen::Matrix3d vec_rot = Eigen::AngleAxisd(angle, v.normalized()).toRotationMatrix();
      pos_cov = rotate_covariance(pos_cov, vec_rot);  // rotate the covariance to point in direction of est. position
      //}

      return pos_cov;
    }
    //}

    /* get_msg_name() method overloads //{ */
    static std::string get_msg_name([[maybe_unused]] const uav_detect::DetectionsConstPtr& msg)
    {
      return "uav_detect::Detections";
    }
    static std::string get_msg_name([[maybe_unused]] const cnn_detect::DetectionsConstPtr& msg)
    {
      return "cnn_detect::Detections";
    }
    /* static std::string get_msg_name([[maybe_unused]] const uav_track::TrackingsConstPtr& msg) */
    /* { */
    /*   return "uav_track::Trackings"; */
    /* } */
    //}

    /* rotate_covariance() method //{ */
    static Eigen::Matrix3d rotate_covariance(const Eigen::Matrix3d& covariance, const Eigen::Matrix3d& rotation)
    {
      return rotation * covariance * rotation.transpose();  // rotate the covariance to point in direction of est. position
    }
    //}

    /* calc_hyp_uncertainty() method //{ */
    static double calc_hyp_uncertainty(const Hypothesis& hyp)
    {
      const Eigen::Matrix3d& position_covariance = hyp.get_position_covariance();
      const double determinant = position_covariance.determinant();
      return sqrt(determinant);
    }
    //}

    /* calc_hyp_meas_loglikelihood() method //{ */
    template <unsigned num_dimensions>
    static double calc_hyp_meas_loglikelihood(const Hypothesis& hyp, const Measurement& meas)
    {
      const auto [inn, inn_cov] = hyp.calc_innovation(meas);
      /* const Eigen::Matrix3d inn_cov = meas.covariance + hyp.get_position_covariance_at(meas.stamp); */
      static const double dylog2pi = num_dimensions*log(2*M_PI);
      const double a = inn.transpose() * inn_cov.inverse() * inn;
      /* const double a = mahalanobis_distance2; */
      const double b = log(inn_cov.determinant());
      return - (a + b + dylog2pi)/2.0;
    }
    //}

    /* max_gating_distance() method //{ */
    double max_gating_distance(Measurement::source_t source)
    {
      double ret = std::numeric_limits<double>::quiet_NaN();
      // The likelihood value must be mapped to the dynamic reconfigure for all values here
      // manually! Not mapping a value might cause unhandled runtime errors, so let's mark this
      // as a compilation error.
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
      switch (source)
      {
        case Measurement::source_t::depth_detection:
          ret = m_drmgr_ptr->config.depth_detections__max_gating_distance;
          break;
        case Measurement::source_t::cnn_detection:
          ret = m_drmgr_ptr->config.cnn_detections__max_gating_distance;
          break;
        case Measurement::source_t::rgb_tracking:
          ret = m_drmgr_ptr->config.rgb_trackings__max_gating_distance;
          break;
        case Measurement::source_t::lkf_prediction:
          ret = std::numeric_limits<double>::max(); // this should not happen...
          break;
      }
#pragma GCC diagnostic pop
      return ret;
    }
    //}

    /* find_closest_measurement() method //{ */
    /* returns position of the closest measurement in the pos_covs vector and its corresponding log-likelihood */
    std::pair<int, double> find_closest_measurement(const Hypothesis& hyp, const std::vector<Measurement>& measurements)
    {
      /* const Eigen::Vector3d hyp_pos = hyp.get_position(); */
      /* const Eigen::Matrix3d hyp_cov = hyp.get_position_covariance(); */
      double max_loglikelihood = std::numeric_limits<double>::lowest();
      int max_dis_it = -1;

      // Find measurement with smallest likelihood from this hypothesis and assign the measurement to it
      for (size_t it = 0; it < measurements.size(); it++)
      {
        const auto& meas = measurements.at(it);
        if (meas.covariance.array().isNaN().any())
          NODELET_ERROR("[LocalizeSingle]: Covariance of LKF contains NaNs!");
        /* const Eigen::Vector3d& det_pos = meas.position; */
        /* const Eigen::Matrix3d& det_cov = meas.covariance; */
        /* const auto [pos, pos_cov] = hyp.get_position_and_covariance_at(meas.stamp); */
        const double loglikelihood = calc_hyp_meas_loglikelihood<3>(hyp, meas);
        const double min_log = max_gating_distance(meas.source);
        /* std::cout << "l: " << loglikelihood << ">?" << min_log << std::endl; */
        if (loglikelihood > min_log)
        {
          /* const double likelihood = exp(loglikelihood); */

          if (loglikelihood > max_loglikelihood)
          {
            max_loglikelihood = loglikelihood;
            max_dis_it = it;
          }
        }
      }
      return std::pair(max_dis_it, max_loglikelihood);
    }
    //}

    /* create_message() method //{ */
    geometry_msgs::PoseWithCovarianceStampedConstPtr create_message(const Hypothesis& hyp, ros::Time stamp) const
    {
      geometry_msgs::PoseWithCovarianceStampedPtr msg = boost::make_shared<geometry_msgs::PoseWithCovarianceStamped>();

      msg->header.frame_id = m_world_frame;
      msg->header.stamp = stamp;

      {
        const Eigen::Vector3d position = hyp.get_position();
        msg->pose.pose.position.x = position(0);
        msg->pose.pose.position.y = position(1);
        msg->pose.pose.position.z = position(2);
      }

      msg->pose.pose.orientation.w = 1.0;

      {
        const Eigen::Matrix3d covariance = hyp.get_position_covariance();
        for (int r = 0; r < 6; r++)
        {
          for (int c = 0; c < 6; c++)
          {
            if (r < 3 && c < 3)
              msg->pose.covariance[r * 6 + c] = covariance(r, c);
            else if (r == c)
              msg->pose.covariance[r * 6 + c] = 666;
            else
              msg->pose.covariance[r * 6 + c] = 0.0;
          }
        }
      }

      return msg;
    }
    //}

    /* create_dbg_message() method //{ */
    uav_localize::LocalizationHypothesesConstPtr create_dbg_message(const std::list<Hypothesis>& hyps, int32_t main_hyp_id, const ros::Time& stamp)
    {
      uav_localize::LocalizationHypothesesPtr msg = boost::make_shared<uav_localize::LocalizationHypotheses>();

      msg->header.stamp = stamp;
      msg->header.frame_id = m_world_frame;
      msg->main_hypothesis_id = main_hyp_id;
      msg->hypotheses.reserve(hyps.size());

      for (const auto& hyp : hyps)
      {
        uav_localize::LocalizationHypothesis hyp_msg;
        hyp_msg.id = hyp.id;
        hyp_msg.n_corrections = hyp.get_n_corrections();
        hyp_msg.last_correction_stamp = hyp.get_last_lkf().stamp;

        const auto& lkfs = hyp.get_lkfs();
        hyp_msg.positions.reserve(lkfs.size());
        hyp_msg.position_sources.reserve(lkfs.size());
        for (const auto& lkf : lkfs)
        {
          geometry_msgs::Point lkf_pt;
          lkf_pt.x = lkf.x(0);
          lkf_pt.y = lkf.x(1);
          lkf_pt.z = lkf.x(2);
          hyp_msg.positions.push_back(lkf_pt);
          hyp_msg.position_sources.push_back(lkf.correction_meas.source);
        }

        msg->hypotheses.push_back(hyp_msg);
      }

      return msg;
    }
    //}

    /* create_pcl_message() method //{ */
    sensor_msgs::PointCloudConstPtr create_pcl_message(const std::list<Hypothesis>& hyps, int32_t main_hyp_id, const ros::Time& stamp)
    {
      sensor_msgs::PointCloudPtr msg = boost::make_shared<sensor_msgs::PointCloud>();

      msg->header.stamp = stamp;
      msg->header.frame_id = m_world_frame;
      msg->points.reserve(hyps.size());

      sensor_msgs::ChannelFloat32 ch_main_hyp;
      ch_main_hyp.name = "main hypothesis";
      ch_main_hyp.values.reserve(hyps.size());

      sensor_msgs::ChannelFloat32 ch_id;
      ch_id.name = "ID";
      ch_id.values.reserve(hyps.size());

      sensor_msgs::ChannelFloat32 ch_n_corrections;
      ch_n_corrections.name = "n. corrections";
      ch_n_corrections.values.reserve(hyps.size());

      sensor_msgs::ChannelFloat32 ch_last_correction_delay;
      ch_last_correction_delay.name = "last correction delay";
      ch_last_correction_delay.values.reserve(hyps.size());

      sensor_msgs::ChannelFloat32 ch_last_correction_source;
      ch_last_correction_source.name = "last correction source";
      ch_last_correction_source.values.reserve(hyps.size());

      for (const auto& hyp : hyps)
      {
        geometry_msgs::Point32 pt;
        const Eigen::Vector3d position = hyp.get_position();
        pt.x = position(0);
        pt.y = position(1);
        pt.z = position(2);
        msg->points.push_back(pt);

        ch_main_hyp.values.push_back(hyp.id == main_hyp_id);
        ch_id.values.push_back(hyp.id);
        ch_n_corrections.values.push_back(hyp.get_n_corrections());
        float delay = (stamp - hyp.get_last_measurement().stamp).toSec();
        ch_last_correction_delay.values.push_back(delay);
        ch_last_correction_source.values.push_back(hyp.get_last_measurement().source);
      }

      msg->channels.push_back(ch_main_hyp);
      msg->channels.push_back(ch_id);
      msg->channels.push_back(ch_n_corrections);
      msg->channels.push_back(ch_last_correction_delay);
      msg->channels.push_back(ch_last_correction_source);
      return msg;
    }
    //}

    /* create_pcl_message() method //{ */
    sensor_msgs::PointCloudConstPtr create_pcl_message(const std::vector<Measurement>& measurements, const ros::Time& stamp)
    {
      sensor_msgs::PointCloudPtr msg = boost::make_shared<sensor_msgs::PointCloud>();

      ros::Time cur_t = ros::Time::now();
      msg->header.stamp = stamp;
      msg->header.frame_id = m_world_frame;
      msg->points.reserve(measurements.size());

      sensor_msgs::ChannelFloat32 ch_source;
      ch_source.name = "source";
      ch_source.values.reserve(measurements.size());

      sensor_msgs::ChannelFloat32 ch_delay;
      ch_delay.values.reserve(measurements.size());

      for (const auto& meas : measurements)
      {
        geometry_msgs::Point32 pt;
        const Eigen::Vector3d position = meas.position;
        pt.x = position(0);
        pt.y = position(1);
        pt.z = position(2);
        msg->points.push_back(pt);

        ch_source.values.push_back(meas.source);
        float delay = (cur_t - meas.stamp).toSec();
        ch_delay.values.push_back(delay);
      }

      msg->channels.push_back(ch_source);
      msg->channels.push_back(ch_delay);
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

    /* Statistics related variables //{ */
    std::mutex m_stat_mtx;  // mutex for synchronization of the statistics variables
    unsigned m_rgb_trackings;
    unsigned m_depth_detections;
    unsigned m_cnn_detections;
    std::string m_most_certain_hyp_name;
    //}

  private:
    // --------------------------------------------------------------
    // |                hypotheses related variables                |
    // --------------------------------------------------------------

    /* Hypotheses - related member variables //{ */
    std::mutex m_hyps_mtx;         // mutex for synchronization of the m_hyps variable
    std::list<Hypothesis> m_hyps;  // all currently active hypotheses
    int32_t m_last_hyp_id;         // ID of the last created hypothesis - used when creating a new hypothesis to generate a new unique ID
    //}

  private:
    // --------------------------------------------------------------
    // |        detail implementation methods (maybe unused)        |
    // --------------------------------------------------------------

    /* kullback_leibler_divergence() method //{ */
    // This method calculates the Kullback-Leibler divergence of two n-dimensional normal distributions.
    template <unsigned num_dimensions>
    static double kullback_leibler_divergence(const Eigen::Vector3d& mu0, const Eigen::Matrix3d& sigma0, const Eigen::Vector3d& mu1,
                                              const Eigen::Matrix3d& sigma1)
    {
      const unsigned k = num_dimensions;  // number of dimensions -- DON'T FORGET TO CHANGE IF NUMBER OF DIMENSIONS CHANGES!
      const double div = 0.5
                         * ((sigma1.inverse() * sigma0).trace() + (mu1 - mu0).transpose() * (sigma1.inverse()) * (mu1 - mu0) - k
                            + log((sigma1.determinant()) / sigma0.determinant()));
      return div;
    }
    //}

    /* mahalanobis_distance2() method //{ */
    // This method calculates square of the Mahalanobis distance of an observation to a normal distributions.
    static double mahalanobis_distance2(const Eigen::Vector3d& x, const Eigen::Vector3d& mu1, const Eigen::Matrix3d& sigma1)
    {
      const auto diff = x - mu1;
      const double dist2 = diff.transpose() * sigma1.inverse() * diff;
      return dist2;
    }
    //}

  };  // class LocalizeSingle : public nodelet::Nodelet
};    // namespace uav_localize

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(uav_localize::LocalizeSingle, nodelet::Nodelet)
