#include "main.h"
#include "display_utils.h"

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
void draw_legend(cv::Mat& img)
{
  static const std::vector<Source> possible_sources =
  {
    {uav_localize::LocalizationHypothesis::SOURCE_DEPTH_DETECTION, "depth detection"},
    {uav_localize::LocalizationHypothesis::SOURCE_RGB_TRACKING, "RGB tracking"}
  };
  int offset = 1;
  cv::putText(img, "last source:", cv::Point(25, 30*offset++), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
  for (auto source : possible_sources)
  {
    cv::Scalar color = get_color(source.id);
    cv::putText(img, source.name, cv::Point(35, 30*offset++), FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
  }
}

bool show_distance = true;
bool show_ID = false;
bool show_all_hyps = false;
bool show_n_corrections = false;
bool show_correction_delay = false;
struct Option {const int key; bool& option; const std::string txt, op1, op2;};
static const std::vector<Option> options =
{
  {'d', show_distance, "showing distance", "not ", ""},
  {'i', show_ID, "showing ID", "not ", ""},
  {'a', show_all_hyps, "showing all hypotheses", "not ", ""},
  {'c', show_n_corrections, "showing number of corrections", "not ", ""},
  {'t', show_correction_delay, "showing time since last correction", "not ", ""},
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

int main(int argc, char** argv)
{
  ros::init(argc, argv, "backproject_display_node");
  ROS_INFO("Node initialized.");

  ros::NodeHandle nh = ros::NodeHandle("~");

  std::cout << "Waiting for valid time..." << std::endl;
  while (!ros::Time::isValid())
    ros::spinOnce();

  mrs_lib::SubscribeHandlerPtr<uav_localize::LocalizationHypotheses> sh_hyps;
  mrs_lib::SubscribeHandlerPtr<sensor_msgs::ImageConstPtr> sh_img;
  mrs_lib::SubscribeHandlerPtr<sensor_msgs::CameraInfo> sh_cinfo;

  double detection_timeout;
  nh.param("detection_timeout", detection_timeout, 0.5);
  ROS_INFO("detection_timeout: %f", detection_timeout);

  mrs_lib::SubscribeMgr smgr(nh);
  sh_hyps = smgr.create_handler_threadsafe<uav_localize::LocalizationHypotheses>("dbg_hypotheses", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
  sh_img = smgr.create_handler_threadsafe<sensor_msgs::ImageConstPtr>("image_rect", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
  sh_cinfo = smgr.create_handler_threadsafe<sensor_msgs::CameraInfo>("camera_info", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));

  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener(tf_buffer);

  if (!smgr.loaded_successfully())
  {
    ROS_ERROR("Failed to subscribe some nodes");
    ros::shutdown();
  }

  print_options();

  int window_flags = WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_NORMAL;
  std::string window_name = "backprojected_localization";
  cv::namedWindow(window_name, window_flags);
  image_geometry::PinholeCameraModel camera_model;
  ros::Rate r(30);

  std::list<sensor_msgs::ImageConstPtr> img_buffer;
  ros::Time last_valid_hypothesis_stamp = ros::Time::now();

  while (ros::ok())
  {
    ros::spinOnce();

    if (sh_cinfo->has_data() && !sh_cinfo->used_data())
    {
      camera_model.fromCameraInfo(sh_cinfo->get_data());
    }

    if (sh_img->new_data())
      add_to_buffer(sh_img->get_data(), img_buffer);

    if (sh_img->has_data() && sh_cinfo->used_data())
    {
      cv::Mat img;
      if (sh_hyps->new_data())
      {
        uav_localize::LocalizationHypotheses hyps_msg = sh_hyps->get_data();
        if (hyps_msg.main_hypothesis_id >= 0)
          last_valid_hypothesis_stamp = hyps_msg.header.stamp;
        sensor_msgs::ImageConstPtr img_ros = find_closest(hyps_msg.header.stamp, img_buffer);

        geometry_msgs::TransformStamped transform;
        try
        {
          transform = tf_buffer.lookupTransform(img_ros->header.frame_id, hyps_msg.header.frame_id, hyps_msg.header.stamp, ros::Duration(1.0));
        } catch (tf2::TransformException& ex)
        {
          ROS_WARN("Error during transform from \"%s\" frame to \"%s\" frame.\n\tMSG: %s", hyps_msg.header.frame_id.c_str(), img_ros->header.frame_id.c_str(), ex.what());
          continue;
        }

        const cv_bridge::CvImagePtr img_ros2 = cv_bridge::toCvCopy(img_ros, "bgr8");
        img = img_ros2->image;

        for (const auto& hyp_msg : hyps_msg.hypotheses)
        {
          const bool is_main = hyp_msg.id == hyps_msg.main_hypothesis_id;
          if (show_all_hyps || is_main)
          {
            geometry_msgs::Point point_transformed;
            tf2::doTransform(hyp_msg.position, point_transformed, transform);

            cv::Point3d pt3d;
            pt3d.x = point_transformed.x;
            pt3d.y = point_transformed.y;
            pt3d.z = point_transformed.z;
            const double dist = sqrt(pt3d.x*pt3d.x + pt3d.y*pt3d.y + pt3d.z*pt3d.z);
            const cv::Point pt2d = camera_model.project3dToPixel(pt3d);

            const cv::Scalar color = get_color(hyp_msg.last_correction_source);
            const int thickness = is_main ? 3 : 1;
            
            cv::circle(img, pt2d, 40, color, thickness);
            cv::line(img, cv::Point(pt2d.x - 15, pt2d.y), cv::Point(pt2d.x + 15, pt2d.y), Scalar(0, 0, 220));
            cv::line(img, cv::Point(pt2d.x, pt2d.y - 15), cv::Point(pt2d.x, pt2d.y + 15), Scalar(0, 0, 220));

            // display info
            {
              int li = 0;        // line iterator
              const int ls = 15; // line step
              const cv::Point lo = pt2d + cv::Point(45, -45);
              if (show_distance)
                cv::putText(img, "distance: " + std::to_string(dist), lo+cv::Point(0, li++*ls), FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
              if (show_ID)
                cv::putText(img, "ID: " + std::to_string(hyp_msg.id), lo+cv::Point(0, li++*ls), FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
              if (show_n_corrections)
                cv::putText(img, "n. cors.: " + std::to_string(hyp_msg.n_corrections), lo+cv::Point(0, li++*ls), FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
              if (show_correction_delay)
              {
                const int delay = round((hyps_msg.header.stamp-hyp_msg.last_correction_stamp).toSec()*1000);
                cv::putText(img, "cor. delay: " + std::to_string(delay) + "ms", lo+cv::Point(0, li++*ls), FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
              }
            }
          } // if (show_all_hyps || is_main)
        } // for (const auto& hyp_msg : hyps_msg.hypotheses)
      } else // if (sh_hyps->new_data())
      {
        sensor_msgs::ImageConstPtr img_ros = img_buffer.back();
        cv_bridge::CvImagePtr img_ros2 = cv_bridge::toCvCopy(img_ros, "bgr8");
        img = img_ros2->image;
      }

      double no_detection = (ros::Time::now() - last_valid_hypothesis_stamp).toSec();
      if (abs(no_detection) > detection_timeout)
        cv::putText(img, "no detection for " + std::to_string(int(round(no_detection*1000))) + "ms", cv::Point(img.rows - 130, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
      draw_legend(img);
      cv::imshow(window_name, img);
      eval_keypress(cv::waitKey(1));
    }

    r.sleep();
  }
}
