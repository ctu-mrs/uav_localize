#include "main.h"
#include "display_utils.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <uav_localize/LocalizedUAV.h>

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
    {uav_localize::LocalizedUAV::SOURCE_DEPTH_DETECTION, "depth detection"},
    {uav_localize::LocalizedUAV::SOURCE_RGB_TRACKING, "rgb tracking"}
  };
  int offset = 1;
  cv::putText(img, "sources:", cv::Point(25, 30*offset++), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
  for (auto source : possible_sources)
  {
    cv::Scalar color = get_color(source.id);
    cv::putText(img, source.name, cv::Point(35, 30*offset++), FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
  }
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "backproject_display_node");
  ROS_INFO("Node initialized.");

  ros::NodeHandle nh = ros::NodeHandle("~");

  mrs_lib::SubscribeHandlerPtr<uav_localize::LocalizedUAV> sh_pose;
  mrs_lib::SubscribeHandlerPtr<sensor_msgs::ImageConstPtr> sh_img;
  mrs_lib::SubscribeHandlerPtr<sensor_msgs::CameraInfo> sh_cinfo;

  double detection_timeout;
  nh.param("detection_timeout", detection_timeout, 0.5);
  ROS_INFO("[%s]: detection_timeout: %f", ros::this_node::getName().c_str(), detection_timeout);

  mrs_lib::SubscribeMgr smgr(nh);
  sh_pose = smgr.create_handler_threadsafe<uav_localize::LocalizedUAV>("dbg_localized_uav", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
  sh_img = smgr.create_handler_threadsafe<sensor_msgs::ImageConstPtr>("image_rect", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));
  sh_cinfo = smgr.create_handler_threadsafe<sensor_msgs::CameraInfo>("camera_info", 1, ros::TransportHints().tcpNoDelay(), ros::Duration(5.0));

  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener(tf_buffer);

  if (!smgr.loaded_successfully())
  {
    ROS_ERROR("[%s]: Failed to subscribe some nodes", ros::this_node::getName().c_str());
    ros::shutdown();
  }

  int window_flags = WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_NORMAL;
  std::string window_name = "backprojected_localization";
  cv::namedWindow(window_name, window_flags);
  image_geometry::PinholeCameraModel camera_model;
  ros::Rate r(30);

  std::list<sensor_msgs::ImageConstPtr> img_buffer;
  ros::Time last_pose_stamp = ros::Time::now();

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
      if (sh_pose->new_data())
      {
        uav_localize::LocalizedUAV loc_uav = sh_pose->get_data();
        last_pose_stamp = loc_uav.header.stamp;
        sensor_msgs::ImageConstPtr img_ros = find_closest(last_pose_stamp, img_buffer);

        geometry_msgs::Point point_transformed;
        try
        {
          geometry_msgs::TransformStamped transform = tf_buffer.lookupTransform(img_ros->header.frame_id, loc_uav.header.frame_id, loc_uav.header.stamp, ros::Duration(1.0));
          tf2::doTransform(loc_uav.position, point_transformed, transform);
        } catch (tf2::TransformException& ex)
        {
          ROS_WARN("Error during transform from \"%s\" frame to \"%s\" frame.\n\tMSG: %s", loc_uav.header.frame_id.c_str(), img_ros->header.frame_id.c_str(), ex.what());
        }

        cv::Point3d pt3d;
        pt3d.x = point_transformed.x;
        pt3d.y = point_transformed.y;
        pt3d.z = point_transformed.z;
        double dist = sqrt(pt3d.x*pt3d.x + pt3d.y*pt3d.y + pt3d.z*pt3d.z);
        
        cv::Point pt2d = camera_model.project3dToPixel(pt3d);
      
        cv_bridge::CvImagePtr img_ros2 = cv_bridge::toCvCopy(img_ros, "bgr8");
        cv::Mat img = img_ros2->image;

        cv::Scalar color = get_color(loc_uav.last_source);

        draw_legend(img);
        cv::circle(img, pt2d, 40, color, 2);
        cv::line(img, cv::Point(pt2d.x - 15, pt2d.y), cv::Point(pt2d.x + 15, pt2d.y), Scalar(0, 0, 220));
        cv::line(img, cv::Point(pt2d.x, pt2d.y - 15), cv::Point(pt2d.x, pt2d.y + 15), Scalar(0, 0, 220));
        cv::putText(img, "distance: " + std::to_string(dist), cv::Point(pt2d.x + 35, pt2d.y + 30), FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        cv::putText(img, "ID: " + std::to_string(loc_uav.hyp_id), cv::Point(pt2d.x + 35, pt2d.y - 30), FONT_HERSHEY_SIMPLEX, 0.5, color, 2);

        cv::imshow(window_name, img);
        cv::waitKey(1);
      } else
      {
        sensor_msgs::ImageConstPtr img_ros = img_buffer.back();
        if (abs((img_ros->header.stamp - last_pose_stamp).toSec()) > detection_timeout)
        {
          cv_bridge::CvImagePtr img_ros2 = cv_bridge::toCvCopy(img_ros, "bgr8");
          cv::Mat img = img_ros2->image;
          cv::putText(img, "no detection", cv::Point(35, 50), FONT_HERSHEY_SIMPLEX, 2.5, Scalar(0, 0, 255), 2);
          cv::imshow(window_name, img);
          cv::waitKey(1);
        }
      }
    }

    r.sleep();
  }
}
