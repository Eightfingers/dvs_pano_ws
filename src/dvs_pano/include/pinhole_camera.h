#include <Eigen/Dense>
#include <vector>
// Contain camera parameters
//

class pinHoleCamera
{
private:
    Eigen::Matrix3f camera_intrinsics_;
    Eigen::Matrix<float, 3, 4> camera_extrinsics_;

    Eigen::Matrix3f current_rot_matrix_;
    Eigen::Vector3f current_translation_vector_;
    Eigen::Matrix<float, 3, 4> camera_matrix_;

    int camera_width_;
    int camera_height_;
    std::vector<Eigen::Vector3f> camera_coordinates_;

public:
    void setCameraIntrinsics(Eigen::Matrix3f camera_intrinsics) { camera_intrinsics_ = camera_intrinsics; };
    void setCameraExtrinsics(Eigen::Matrix<float, 3, 4> camera_extrinsics) { camera_extrinsics_ = camera_extrinsics; };
    void setCameraSize(int width, int height)
    {
        camera_width_ = width;
        camera_height_ = height;
    };

    Eigen::Matrix3f getCameraIntrinsics() { return camera_intrinsics_; };
    Eigen::Matrix<float, 3, 4> getCameraExtrinsics() { return camera_extrinsics_; };
    Eigen::Matrix3f getRotMatrix() { return current_rot_matrix_; };
    Eigen::Vector3f getTranslationVector() { return current_translation_vector_; };
    int getCameraHeight(){return camera_height_;};
    int getCameraWidth(){return camera_width_;};
    std::vector<Eigen::Vector3f>* getCameraCoordinatesPtr(){return &camera_coordinates_;};

    void preCalculate_K_Inv_Q(); // camera coordinates
};