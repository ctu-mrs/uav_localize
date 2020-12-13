# MAV filtering and localization using multiple detection methods

Implements the multi-target tracking and filtering algorithm, described in the papers [1] and [2].

Rudimentary knowledge of the ROS system is assumed.
This package is intended to be used with the `uav_detect` package.

## The working principle

Target of this algorithm is to filter out false detections from the detection algorithms and to provide a 3D position estimate of the current position of the target (a **single valid target is presumed** for the purpose of autonomous interception, but the algorithm may easily be modified to output multiple targets).
A set of active tracks is kept and a prediction update is made for each track in this set periodically.
The set is initially empty and tracks are added to it with new detections.
Each track represents a hypothesis about the 3D position and velocity of a target.
A Kalman filter is used for the correction and prediction steps.
If a new set of detections comes in, the following procedure is executed:
 1) The detections are recalculated to 3D locations in the world coordinate system based on the camera projection matrices and the ROS transformation from sensor to world at the time of taking the image in which the blobs were detected.
 2) Covariance of each 3D location is calculated based on the set parameters (`xy_covariance_coeff` and `z_covariance_coeff`) and transformed to the world coordinate system (so that it is properly rotated).
 3) For each track in the set of currently active tracks the following steps are taken:
   1) The measurement with the highest likelihood from the set of the latest measurements (3D positions with covariances) is associated to the track and used for a correction update unless it is further than `max_gating_distance`.
   2) The uncertainty of the track is calculated (currently a determinant of the state covariance matrix) and if it is higher than `max_hyp_uncertainty`, the track is kicked out from the pool of currently active tracks and is not considered further.
 4) The track with the highest number of correction updates is found. If two track have the same number of correction updates, the one with the lower uncertainty is picked. If the number of correction updates of the resulting track is higher than `min_corrs_to_consider` then the position estimation of this track is published as the current position of the detected target.
 5) For each measurement which was not associated to any track, a new track is instantiated using the parameters `lkf_process_noise_pos` and `lkf_process_noise_vel` to initialize the process noise matrix and `init_vel_cov` to initialize covariance of the velocity states. The initial position estimate and its covariance are initialized based on the measurement. Initial velocity estimate is set to zero.

For a more thorough description and evaluation of the algorithm implemented in this repository, see the paper [1].

## Description of the provided interface and other info

### The following launchfiles are provided:
 * **localize_single.launch**: Starts 3D localization and filtering of a single UAV position from the detected blobs using Kalman Filters.
 * **backproject_location.launch**: Starts visualization of the detected UAV 3D location. The location is backprojected to the RGB image and displayed in an OpenCV window.
 * **simulation.launch**: Starts the simulation world with some trees and a grass pane. Also starts static transform broadcasters for the Realsense camera coordinate frames! *Note:* Don't use this launchfile to start the simulation manually. Instead, use the prepared tmux scripts.
 * **localization_pipeline.launch**: Starts the sensor driver nodelet, detection nodelet and localization nodelet under a single nodelet manager (see ROS nodelets for explanation).

### The following config files are used by the nodes:
 * **localization_params_common.yaml:** Contains common parameters for the UAV localization parameter. Parameters are documented in the file itself.
 * **localization_params_realsense.yaml:** Contains parameters for the UAV localization, tuned to detect drones on real-world data with the Realsense sensor. Parameters are documented in the file itself.
Most parameters (those which make sense) from the above files are dynamically reconfigurable.

### To launch simulation, detection, localization and visualization:
 1) In folder **tmux_scripts/simulation** launch the tmuxinator session to start simulation with two drones: `uav1` and `uav2`. `uav1` is the interceptor with the Realsense sensor and `uav2` is an intruder with a random flier controlling it.
 2) In folder **tmux_scripts/detection_and_visualization** launch the tmuxinator session to start detection, localization and visualization nodes.
 3) You can adjust the detection and localization parameters according to Rviz or according to the OpenCV visualization using the **rqt_reconfigure** (which is automatically launched in the **detection_and_visualization** session).

----
References:

 * [1]: M. Vrba, D. Heřt and M. Saska, "Onboard Marker-Less Detection and Localization of Non-Cooperating Drones for Their Safe Interception by an Autonomous Aerial System," in IEEE Robotics and Automation Letters, vol. 4, no. 4, pp. 3402-3409, Oct. 2019, doi: 10.1109/LRA.2019.2927130.
 * [2]: M. Vrba and M. Saska, "Marker-Less Micro Aerial Vehicle Detection and Localization Using Convolutional Neural Networks," in IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 2459-2466, April 2020, doi: 10.1109/LRA.2020.2972819.
