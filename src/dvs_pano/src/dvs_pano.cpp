#include "dvs_pano.h"
#include <geometry_msgs/Quaternion.h>
#include <tf2/LinearMath/Quaternion.h>
#include <Eigen/Geometry>

DVSPano::DVSPano(ros::NodeHandle &nh)
    : nh_(nh), pnh_("~"), it_(nh)

{
    const std::string events_topic = pnh_.param<std::string>("events_topic", "/dvs/events");
    const std::string camera_info_topic = pnh_.param<std::string>("camera_info_topic", "/dvs/camera_info");
    const std::string pose_topic = pnh_.param<std::string>("pose_topic", "/optitrack/davis");

    // Set up subscribers
    event_sub_ = nh_.subscribe(events_topic, 0, &DVSPano::eventsCallback, this);
    cam_info_sub_ = nh_.subscribe(camera_info_topic, 0, &DVSPano::camInfoCallback, this);
    pose_sub_ = nh_.subscribe(pose_topic, 0, &DVSPano::poseCallback, this);
    got_cam_info_ = false;

    // Publish images
    pano_img_pub_ = it_.advertise("camera/pano_image", 1);
    event_packet_img_pub_ = it_.advertise("camera/raw_event_image", 1);
    num_of_events_per_packet_ = pnh_.param<int>("events_per_packet", 0);
}

DVSPano::~DVSPano()
{
    ROS_INFO("Closing!");
};

void DVSPano::camInfoCallback(const sensor_msgs::CameraInfo::ConstPtr &camera_info)
{
    if (!got_cam_info_)
    {
        got_cam_info_ = true;

        ROS_INFO("Camera info got");
        Eigen::Matrix3f camera_intrinsics;
        camera_intrinsics << camera_info->K[0], camera_info->K[1], camera_info->K[2],
            camera_info->K[3], camera_info->K[4], camera_info->K[5],
            camera_info->K[6], camera_info->K[7], camera_info->K[8];

        cam_.setCameraIntrinsics(camera_intrinsics);
        cam_.setCameraSize(camera_info->width, camera_info->height);
        cam_.preCalculate_K_Inv_Q(); // precalculate camera coordinates

        ROS_INFO("Camera camera_intrinsics_ defined as: ");
        std::cout << cam_.getCameraIntrinsics() << std::endl;
        ROS_INFO(" ");
        cam_info_sub_.shutdown(); // no need to listen to this topic any more
    }
}

void DVSPano::poseCallback(const geometry_msgs::PoseStamped::ConstPtr &msg)
{
    tf2::Quaternion tf2_quaternion;
    tf2_quaternion.setX(msg->pose.orientation.x);
    tf2_quaternion.setY(msg->pose.orientation.y);
    tf2_quaternion.setZ(msg->pose.orientation.z);
    tf2_quaternion.setW(msg->pose.orientation.x);
    Eigen::Quaternionf eigen_quaternion(tf2_quaternion.w(), tf2_quaternion.x(), tf2_quaternion.y(), tf2_quaternion.z());

    current_rot_matrix_ = eigen_quaternion.toRotationMatrix();
}

//
void DVSPano::projectPixelToCylindircal(int x, int y, float theta)
{

    int idx = x * cam_.getCameraHeight() + y;
    std::vector<Eigen::Vector3f> *camera_coordinates_ptr = cam_.getCameraCoordinatesPtr();

    try 
    {
        Eigen::Vector3f camera_coordinates = camera_coordinates_ptr->at(idx); // K^-1 * (x,y,1)
        Eigen::Vector3f world_coordinates = current_rot_matrix_ * camera_coordinates;
        float X = world_coordinates(0);
        float Y = world_coordinates(1);
        float Z = world_coordinates(2);

        // Convert Cylindrical Coordinate System
        float projectedX = map_center_(0) * (1 + atan2(X, Z) / M_PI);
        float projectedY = map_center_(1) * (1 + Y / sqrt(Z * Z + X * X));

        // Unwrap the cylinder 
        // int projectedX = r * cos(theta);
        // int projectedY = r * sin(theta);
        if ((projectedX > 0 && projectedX < pano_size_(0) - 2) && (projectedY < pano_size_(1) -2 && projectedY > 0))
        {
            pano_frame_.at<float>(projectedY, projectedX) = 1;
        }
        else
        {
            if (projectedX > pano_size_(0))
            {
                std::cout << "X IS TOO BIG: " << projectedX << std::endl;
            }
            if (projectedY > pano_size_(1))
            {
                std::cout << "Y IS TOO BIG: " << projectedY << std::endl;
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        std::cerr << idx << std::endl;
    }

    // std::cout << ideal_homogeneous_coords << std::endl;
    // std::cout << rotatedCoords << std::endl;
    // std::cout << map_center_ << std::endl;
    // std::cout << projectedX << std::endl;
    // std::cout << projectedY << std::endl;
}

void DVSPano::eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg)
{
    static int num_events = 0;

    cv::Mat frame_ = cv::Mat::zeros(cam_.getCameraWidth(), cam_.getCameraHeight(), CV_32FC1);
    for (int i = 0; i < event_vector_.size(); i++)
    {
        int x = event_vector_[i].x;
        int y = event_vector_[i].y;
        frame_.at<float>(x, y) = 1.0;
    }
    sensor_msgs::ImagePtr event_image_pub_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", frame_).toImageMsg();
    event_image_pub_msg->header = msg->header;
    event_image_pub_msg->header.frame_id = "map";
    event_packet_img_pub_.publish(event_image_pub_msg);

    if (!got_cam_info_)
    {
        ROS_ERROR("Received events but camera info is still missing");
        return;
    }

    for (auto ev = msg->events.begin(); ev < msg->events.end(); ev += 1)
    {
        // Push events into the frontend, which will pass them to the backend then.
        num_events++;
        event_vector_.push_back(*ev);

        if (num_events > num_of_events_per_packet_)
        {
            pano_frame_.setTo(0);
            cv::Mat frame_ = cv::Mat::zeros(cam_.getCameraHeight(), cam_.getCameraWidth(), CV_32FC1);
            for (int i = 0; i < event_vector_.size(); i++)
            {
                int x = event_vector_[i].x;
                int y = event_vector_[i].y;
                projectPixelToCylindircal(x, y, 0.5);
                frame_.at<float>(y, x) = 1;
            }

            sensor_msgs::ImagePtr pano_pub_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", pano_frame_).toImageMsg();
            pano_pub_msg->header = msg->header;
            pano_pub_msg->header.frame_id = "map";
            // Publish the image message
            pano_img_pub_.publish(pano_pub_msg);
            num_events = 0;
            event_vector_.clear();
        }
    }
}
