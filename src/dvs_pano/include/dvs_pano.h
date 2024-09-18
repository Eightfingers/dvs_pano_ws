#include "ros/ros.h"
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PoseStamped.h>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
#include <Eigen/Dense>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <opencv2/core/core.hpp>
#include "pinhole_camera.h"

class DVSPano
{
public:
    DVSPano(ros::NodeHandle &nh);
    ~DVSPano();

private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    ros::Subscriber event_sub_, cam_info_sub_, pose_sub_;
    image_transport::ImageTransport it_;
    image_transport::Publisher pano_img_pub_;
    image_transport::Publisher event_packet_img_pub_;
    // image_transport::Publisher event_packet_projected_img_pub_;

    void eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg);
    void camInfoCallback(const sensor_msgs::CameraInfo::ConstPtr &camera_info);
    void poseCallback(const geometry_msgs::PoseStamped::ConstPtr &msg);

    void projectPixelToCylindircal(int x, int y, float theta);
    void publishPacketFOV();
    void undistort();
    void calculateNormalizationFactor();
    void calculateMapFunction();

    // Parameters
    bool got_cam_info_ = false;
    int num_of_events_per_packet_ = 1500;
    Eigen::Vector2f map_center_ = {1024, 256};
    Eigen::Vector2f pano_size_ = {2048, 512};
    cv::Mat pano_frame_ = cv::Mat::zeros(pano_size_(1), pano_size_(0), CV_32FC1);

    
    pinHoleCamera cam_;

    Eigen::Matrix3f current_rot_matrix_;
    Eigen::Vector3f current_translation_vector_;

    // Events
    std::vector<dvs_msgs::Event> event_vector_;    
};