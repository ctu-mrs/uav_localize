# This message is intended primarily for debugging and visualisation purposes.
# It contains data about a UAV location hypothesis.

int32               id
geometry_msgs/Point position
int64               n_corrections
time                last_correction_stamp
uint8               last_correction_source

uint8               SOURCE_DEPTH_DETECTION=0
uint8               SOURCE_RGB_TRACKING=1
