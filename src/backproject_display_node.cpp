#include "main.h"
#include "display_utils.h"

#include <fstream>

#include <std_msgs/Bool.h>
#include <nav_msgs/Odometry.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <uav_localize/LocalizationHypotheses.h>

using namespace cv;
using namespace std;
using namespace uav_localize;

cv::Scalar get_color(int id)
{
  const cv::Scalar colors[] =
  {
    cv::Scalar(255, 0, 0),
    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 0, 255),
    cv::Scalar(255, 255, 0),
    cv::Scalar(255, 0, 255),
    cv::Scalar(0, 255, 255),
    cv::Scalar(255, 128, 128),
    cv::Scalar(128, 128, 255),
    cv::Scalar(128, 255, 128)
  };
  const int n_colors = 9;
  return colors[id % n_colors];
}

struct Source {int id; std::string name;};
static const std::vector<Source> possible_sources =
{
  {uav_localize::LocalizationHypothesis::SOURCE_DEPTH_DETECTION, "depth detection"},
  {uav_localize::LocalizationHypothesis::SOURCE_CNN_DETECTION, "CNN detection"},
  {uav_localize::LocalizationHypothesis::SOURCE_RGB_TRACKING, "RGB tracking"},
  {uav_localize::LocalizationHypothesis::SOURCE_LKF_PREDICTION, "LKF prediction"},
};
void draw_legend(cv::Mat& img)
{
  int offset = 1;
  cv::putText(img, "last source:", cv::Point(25, 30*offset++), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
  for (auto source : possible_sources)
  {
    cv::Scalar color = get_color(source.id);
    cv::putText(img, source.name, cv::Point(35, 30*offset++), FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
  }
}

bool show_distance = true;
bool show_ID = true;
bool show_all_hyps = false;
bool show_n_corrections = true;
bool show_correction_delay = true;
bool show_history = false;
struct Option {const int key; bool& option; const std::string txt, op1, op2;};
static const std::vector<Option> options =
{
  {'d', show_distance, "showing distance", "not ", ""},
  {'i', show_ID, "showing ID", "not ", ""},
  {'a', show_all_hyps, "showing all hypotheses", "not ", ""},
  {'c', show_n_corrections, "showing number of corrections", "not ", ""},
  {'t', show_correction_delay, "showing time since last correction", "not ", ""},
  {'g', show_history, "showing localization history", "not ", ""},
};
void print_options()
{
  ROS_INFO("Options (change by selecting the OpenCV window and pressing the corresponding key)");
  std::cout << "key:\ttoggles:" << std::endl;
  std::cout << "----------------------------" << std::endl;
  for (const auto& opt : options)
  {
    std::cout << ' ' << char(opt.key) << '\t' << opt.txt << std::endl;
  }
}
void eval_keypress(int key)
{
  for (const auto& opt : options)
  {
    if (key == opt.key)
    {
      ROS_INFO(("%s" + opt.txt).c_str(), opt.option?opt.op1.c_str():opt.op2.c_str());
      opt.option = !opt.option;
    }
  }
}

std::string to_str_prec(double num, unsigned prec = 3)
{
  std::stringstream strstr;
  strstr << std::fixed << std::setprecision(prec);
  strstr << num;
  return strstr.str();
}

cv::Mat color_if_depthmap(cv::Mat img, const std::string& encoding)
{
  /* static float max_depth = -1; */
  if (encoding == "mono16")
  {
    cv::Mat dm_im_colormapped;
    double min = 0;
    double max = 40000;
    cv::Mat unknown_pixels;
    cv::compare(img, 65535, unknown_pixels, cv::CMP_EQ);
    /* cv::minMaxIdx(img, &min, &max, nullptr, nullptr, ~unknown_pixels); */
    /* if (max_depth == -1) */
    /*   max_depth = max; */
    /* else */
    /*   max_depth = 0.99*max_depth + 0.01*max; */
    /* max = max_depth; */
    cv::Mat im_8UC1;
    img.convertTo(im_8UC1, CV_8UC1, 255.0 / (max-min), -min * 255.0 / (max-min)); 
    applyColorMap(im_8UC1, dm_im_colormapped, cv::COLORMAP_JET);
    cv::Mat blackness = cv::Mat::zeros(dm_im_colormapped.size(), dm_im_colormapped.type());
    blackness.copyTo(dm_im_colormapped, unknown_pixels);
    return dm_im_colormapped;
  } else if (encoding == "mono8")
  {
    cv::Mat im_8UC3;
    cv::cvtColor(img, im_8UC3, cv::COLOR_GRAY2BGR);
    return im_8UC3;
  } else if (encoding == "rgb8")
  {
    cv::Mat im_bgr;
    cv::cvtColor(img, im_bgr, cv::COLOR_RGB2BGR);
    return im_bgr;
  } else
  {
    return img;
  }
}

/* main //{ */

const double x_offset = 5.747291 - 3.970789;
const double y_offset = (-28.323279) - (-27.408026);
const double z_offset = 3.658255 - 194.121400;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "backproject_display_node");
  ROS_INFO("Node initialized.");

  ros::NodeHandle nh = ros::NodeHandle("~");

  {
    std::cout << "Waiting for valid time..." << std::endl;
    ros::Rate r(10);
    while (!ros::Time::isValid())
    {
      r.sleep();
      ros::spinOnce();
    }
  }

  double detection_timeout;
  nh.param("detection_timeout", detection_timeout, 0.5);
  ROS_INFO("detection_timeout: %f", detection_timeout);

  mrs_lib::SubscribeHandlerOptions shopts;
  shopts.no_message_timeout = ros::Duration(5.0);
  auto sh_hyps = mrs_lib::SubscribeHandler<uav_localize::LocalizationHypotheses>(shopts, "dbg_hypotheses");
  auto sh_img = mrs_lib::SubscribeHandler<sensor_msgs::Image>(shopts, "image_rect");
  auto sh_cinfo = mrs_lib::SubscribeHandler<sensor_msgs::CameraInfo>(shopts, "camera_info");
  auto sh_gt = mrs_lib::SubscribeHandler<nav_msgs::Odometry>(shopts, "ground_truth");
  auto sh_pos = mrs_lib::SubscribeHandler<nav_msgs::Odometry>(shopts, "odometry");

  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener(tf_buffer);

  print_options();

  int window_flags = WINDOW_AUTOSIZE | WINDOW_KEEPRATIO | WINDOW_GUI_NORMAL;
  std::string window_name = "backprojected_localization";
  cv::namedWindow(window_name, window_flags);
  image_geometry::PinholeCameraModel camera_model;
  ros::Rate r(100);
  /* cv::VideoWriter writer; */

  std::list<sensor_msgs::ImageConstPtr> img_buffer;
  ros::Time last_valid_hypothesis_stamp = ros::Time::now();

  /* std::ofstream ofile("err_over_dist.csv"); */
  /* ofile << "stamp,dist,est_dist,error" << std::endl; */

  while (ros::ok())
  {
    ros::spinOnce();

    if (sh_cinfo.hasMsg() && !sh_cinfo.usedMsg())
    {
      camera_model.fromCameraInfo(sh_cinfo.getMsg());

      /* string filename = "./backproject_output.avi";             // name of the output video file */
      /* ROS_INFO("[%s]: Initializing video writer to file %s", ros::this_node::getName().c_str(), filename.c_str()); */
      /* int codec = CV_FOURCC('F', 'F', 'V', '1');  // select desired codec (must be available at runtime) */
      /* /1* int codec = -1; *1/ */
      /* double fps = 25.0;                          // framerate of the created video stream */
      /* cv::Size vid_size(camera_model.cameraInfo().width, camera_model.cameraInfo().height); */
      /* writer.open(filename, codec, fps, vid_size, true); */
      /* writer.set(cv::VIDEOWRITER_PROP_QUALITY, 100.0); */
      /* // check if we succeeded */
      /* if (!writer.isOpened()) { */
      /*     cerr << "Could not open the output video file for write\n"; */
      /*     return -1; */
      /* } */
    }

    if (sh_img.newMsg())
      add_to_buffer(sh_img.getMsg(), img_buffer);

    if (sh_img.hasMsg() && sh_cinfo.usedMsg())
    {
      cv::Mat img;
      /* if (sh_dets.hasMsg()) */
      /* { */
      /*   ros::Time cur_det_t = sh_dets.getMsg()->header.stamp; */
      /*   if (cur_det_t != prev_det_t) */
      /*   { */
      /*     const double cur_freq = 1.0/(cur_det_t - prev_det_t).toSec(); */
      /*     det_freq = cur_freq; */
      /*     prev_det_t = cur_det_t; */
      /*   } */
      /* } */
      if (sh_hyps.hasMsg())
      {
        uav_localize::LocalizationHypotheses hyps_msg = *sh_hyps.getMsg();
        if (hyps_msg.main_hypothesis_id >= 0)
          last_valid_hypothesis_stamp = hyps_msg.header.stamp;
        const ros::Time cur_loc_t = hyps_msg.header.stamp;
        sensor_msgs::ImageConstPtr img_ros = find_closest(cur_loc_t, img_buffer);

        geometry_msgs::TransformStamped transform;
        geometry_msgs::TransformStamped transform_inv;
        try
        {
          transform = tf_buffer.lookupTransform(img_ros->header.frame_id, hyps_msg.header.frame_id, hyps_msg.header.stamp, ros::Duration(1.0));
          transform_inv = tf_buffer.lookupTransform(hyps_msg.header.frame_id, img_ros->header.frame_id, hyps_msg.header.stamp, ros::Duration(1.0));
        } catch (tf2::TransformException& ex)
        {
          ROS_WARN("Error during transform from \"%s\" frame to \"%s\" frame.\n\tMSG: %s", hyps_msg.header.frame_id.c_str(), img_ros->header.frame_id.c_str(), ex.what());
          continue;
        }

        const cv_bridge::CvImagePtr img_ros2 = cv_bridge::toCvCopy(img_ros, img_ros->encoding);
        img = color_if_depthmap(img_ros2->image, img_ros->encoding);

        for (const auto& hyp_msg : hyps_msg.hypotheses)
        {
          const bool is_main = hyp_msg.id == hyps_msg.main_hypothesis_id;
          if (show_all_hyps || is_main)
          {
            const size_t n_hist = hyp_msg.positions.size();
            if (hyp_msg.position_sources.size() != n_hist)
            {
              ROS_ERROR("[%s]: Number of position sources is not equal to number of positions, skipping!", ros::this_node::getName().c_str());
              continue;
            }

            cv::Point prev_pt2d;
            geometry_msgs::Point pt3d_global;
            cv::Point3d pt3d;
            cv::Scalar color;
            for (size_t it = 0; it < n_hist; it++)
            {
              if (!show_history && it != n_hist-1)
                continue;
              geometry_msgs::Point point_transformed;
              tf2::doTransform(hyp_msg.positions[it], point_transformed, transform);

              pt3d_global = hyp_msg.positions[it];

              pt3d.x = point_transformed.x;
              pt3d.y = point_transformed.y;
              pt3d.z = point_transformed.z;
              const cv::Point pt2d = camera_model.project3dToPixel(pt3d);

              if (hyp_msg.position_sources[it] > possible_sources.size())
                ROS_WARN("[%s]: INVALID HYPOTHESIS SOURCE: %u!", ros::this_node::getName().c_str(), hyp_msg.position_sources[it]);

              /* color = get_color(hyp_msg.position_sources[it]); */
              color = cv::Scalar(255, 0, 0);
              /* const int thickness = is_main ? 2 : 1; */
              const int thickness = 1;
              const int size = is_main ? 8 : 3;

              cv::circle(img, pt2d, size, color, thickness);
              if (show_history && it > 1)
                cv::line(img, prev_pt2d, pt2d, color);

              prev_pt2d = pt2d;
            } // for (size_t it = 0; it < n_hist; it++)

            if (is_main)
            {
              cv::circle(img, prev_pt2d, 30, color, 1);
              cv::line(img, cv::Point(prev_pt2d.x - 15, prev_pt2d.y), cv::Point(prev_pt2d.x + 15, prev_pt2d.y), Scalar(0, 0, 220));
              cv::line(img, cv::Point(prev_pt2d.x, prev_pt2d.y - 15), cv::Point(prev_pt2d.x, prev_pt2d.y + 15), Scalar(0, 0, 220));
            }

            bool show_error = false;
            double cur_err = std::numeric_limits<double>::quiet_NaN();
            if (sh_gt.hasMsg())
            {
              nav_msgs::Odometry gt = *sh_gt.getMsg();
              const auto hx = pt3d_global.x;
              const auto hy = pt3d_global.y;
              const auto hz = pt3d_global.z;
              const auto gx = gt.pose.pose.position.x + x_offset;
              const auto gy = gt.pose.pose.position.y + y_offset;
              const auto gz = gt.pose.pose.position.z + z_offset;
              cur_err = sqrt( (hx-gx)*(hx-gx) + (hy-gy)*(hy-gy) + (hz-gz)*(hz-gz) );
              show_error = true;

              geometry_msgs::TransformStamped transform;
              try
              {
                transform = tf_buffer.lookupTransform(img_ros->header.frame_id, gt.header.frame_id, hyps_msg.header.stamp, ros::Duration(1.0));
              } catch (tf2::TransformException& ex)
              {
                ROS_WARN("Error during transform from \"%s\" frame to \"%s\" frame.\n\tMSG: %s", hyps_msg.header.frame_id.c_str(), img_ros->header.frame_id.c_str(), ex.what());
                continue;
              }
              geometry_msgs::Point gt_bearing_world; gt_bearing_world.x = gx; gt_bearing_world.y = gy; gt_bearing_world.z = gz;
              geometry_msgs::Point gt_dir_vec;
              tf2::doTransform(gt_bearing_world, gt_dir_vec, transform);
              Eigen::Vector3d gt_dir = Eigen::Vector3d(gt_dir_vec.x, gt_dir_vec.y, gt_dir_vec.z).normalized();
              Eigen::Vector3d est_dir = Eigen::Vector3d(pt3d.x, pt3d.y, pt3d.z).normalized();
              std::cout << "ground-truth dir. vec.: " << gt_dir.transpose() << std::endl;
              std::cout << "estimated    dir. vec.: " << est_dir.transpose() << std::endl;
              const double gt_gamma1 = atan2(gt_dir.x(), gt_dir.z());
              const double gt_gamma2 = atan2(gt_dir.y(), gt_dir.z());

              const double est_gamma1 = atan2(est_dir.x(), est_dir.z());
              const double est_gamma2 = atan2(est_dir.y(), est_dir.z());

              std::cout << "ground-truth bearing: " << gt_gamma1 << ", " << gt_gamma2 << std::endl;
              std::cout << "estimated    bearing: " << est_gamma1 << ", " << est_gamma2 << std::endl;
            }

            // display info
            {
              const double est_dist = sqrt(pt3d.x*pt3d.x + pt3d.y*pt3d.y + pt3d.z*pt3d.z);
              double gt_dist = std::numeric_limits<double>::quiet_NaN();
              if (sh_pos.hasMsg() && sh_gt.hasMsg())
              {
                auto pos = *sh_pos.getMsg();
                auto gt = *sh_gt.getMsg();
                const auto px = pos.pose.pose.position.x;
                const auto py = pos.pose.pose.position.y;
                const auto pz = pos.pose.pose.position.z;
                const auto gx = gt.pose.pose.position.x;
                const auto gy = gt.pose.pose.position.y;
                const auto gz = gt.pose.pose.position.z;
                gt_dist = sqrt( (px-gx)*(px-gx) + (py-gy)*(py-gy) + (pz-gz)*(pz-gz) );
              }
              int li = 0;        // line iterator
              const int ls = 15; // line step
              const cv::Point lo = prev_pt2d + cv::Point(45, -45);
              if (show_distance)
              {
                cv::putText(img, "est.dist: " + to_str_prec(est_dist) + "m", lo+cv::Point(0, li++*ls), FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                cv::putText(img, "gt. dist: " + to_str_prec(gt_dist) + "m", lo+cv::Point(0, li++*ls), FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
              }
              if (show_ID)
                cv::putText(img, "ID: " + std::to_string(hyp_msg.id), lo+cv::Point(0, li++*ls), FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
              if (show_n_corrections)
                cv::putText(img, "n. corrections: " + std::to_string(hyp_msg.n_corrections), lo+cv::Point(0, li++*ls), FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
              if (show_correction_delay)
              {
                const int delay = round((hyps_msg.header.stamp-hyp_msg.last_correction_stamp).toSec()*1000);
                cv::putText(img, "correction delay: " + std::to_string(delay) + "ms", lo+cv::Point(0, li++*ls), FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
              }
              if (show_error)
                cv::putText(img, "error: " + std::to_string(cur_err), lo+cv::Point(0, li++*ls), FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

              /* if (show_error) */
              /*   ofile << hyps_msg.header.stamp.toSec() << "," << gt_dist << "," << est_dist << "," << cur_err << std::endl; */
            }
          } // if (show_all_hyps || is_main)
        } // for (const auto& hyp_msg : hyps_msg.hypotheses)
      } else // if (sh_hyps.newMsg())
      {
        sensor_msgs::ImageConstPtr img_ros = img_buffer.back();
        cv_bridge::CvImagePtr img_ros2 = cv_bridge::toCvCopy(img_ros, img_ros->encoding);
        img = color_if_depthmap(img_ros2->image, img_ros->encoding);
      }

      /* const string state = eliminating ? "locked, firing net" : "detected, intercepting"; */
      /* cv::putText(img, "intruder " + state, cv::Point(50, 50), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 2); */
      /* cv::putText(img, "image frequency: " + to_str_prec(img_freq) + "Hz", cv::Point(50, img.rows - 75), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 0), 1); */
      /* cv::putText(img, "detection frequency: " + to_str_prec(det_freq) + "Hz", cv::Point(50, img.rows - 50), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 0), 1); */
      /* cv::putText(img, "localization frequency: " + to_str_prec(loc_freq) + "Hz", cv::Point(50, img.rows - 25), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 0), 1); */
      /* double no_detection = (ros::Time::now() - last_valid_hypothesis_stamp).toSec(); */
      /* if (abs(no_detection) > detection_timeout) */
      /*   cv::putText(img, "no detection for " + std::to_string(int(round(no_detection*1000))) + "ms", cv::Point(img.rows - 130, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2); */
      /* draw_legend(img); */
      cv::imshow(window_name, img);
      /* writer.write(img); */
      eval_keypress(cv::waitKey(1));
    }

    r.sleep();
  }
  /* ROS_INFO("[%s]: Closing video writer", ros::this_node::getName().c_str()); */
  /* writer.release(); */
}

//}
