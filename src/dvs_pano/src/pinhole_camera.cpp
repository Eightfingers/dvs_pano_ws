#include "pinhole_camera.h"

void pinHoleCamera::preCalculate_K_Inv_Q()
{
    for (int i = 0; i < camera_height_; i++)
    {
        for (int j = 0; j < camera_width_; j++)
        {
            int idx = i * camera_height_ + j;
            Eigen::Vector3f homogenous_pixel_coordinates = {(float) j, (float) i, 1.0f};
            Eigen::Vector3f camera_coordinate = camera_intrinsics_ * homogenous_pixel_coordinates;
            camera_coordinates_.push_back(camera_coordinate);
        }
    }
}
