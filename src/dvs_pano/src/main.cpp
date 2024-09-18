#ifdef __INTELLISENSE__
#define __CUDACC__
#endif

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "dvs_pano.h"
#include <sstream>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <vector>

using namespace std::chrono;

int main(int argc, char **argv)
{

    ros::init(argc, argv, "dvs_pano");
    ros::NodeHandle nh;
    DVSPano dvs_pano(nh);
    
    ros::spin();

    return 0;
}
