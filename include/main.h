#ifndef MAIN_H
#define MAIN_H

#include <ros/package.h>
#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <image_geometry/pinhole_camera_model.h>
#include <image_transport/image_transport.h>

#include <list>

#include <uav_detect/Detections.h>
#include <uav_track/Trackings.h>
#include <mrs_lib/ParamLoader.h>
#include <mrs_lib/DynamicReconfigureMgr.h>
#include <mrs_lib/subscribe_handler.h>

#include <boost/circular_buffer.hpp>

#endif //  MAIN_H
