# This message is intended primarily for debugging and visualisation purposes.
# It contains data from detected UAV localization.

std_msgs/Header     header
geometry_msgs/Point position
uint32              hyp_id
uint8               last_source
uint8               SOURCE_DEPTH_DETECTION=0
uint8               SOURCE_RGB_TRACKING=1
