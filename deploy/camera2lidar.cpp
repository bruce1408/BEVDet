

/*

point_cloud_range = [-51.2, -51.2, -10.0, 51.2, 51.2, 10.0]
voxel_size = [0.8, 0.8, 20.0]
grid_size = [128, 128, 1]

bev_channels = 64
D=59
H=128
W=128
score_thresh=0.1

这个代码是libtorch版本的，把at::Tensor换成指针传入就变为纯C++版本；
bev_features是6路拼接的输出BEV地址，
depth是camera_feat_encoder输出的前59个通道
camera_features是camera_feat_encoder输出的后64个通道

frustum是一组固定的数据，不会变化

*/





at::Tensor camera2lidar(
  at::Tensor bev_features_, at::Tensor depth_, at::Tensor camera_features_, at::Tensor frustum_, at::Tensor calib_pixel2camera_,
  at::Tensor calib_camera2lidar_,
  std::vector<float> point_cloud_range, std::vector<float> voxel_size,
  std::vector<int> grid_size, int D, int H, int W, int bev_channels, float score_thresh
) {

  float* bev_features = bev_features_.data_ptr<float>();
  const float* depth = depth_.data_ptr<float>();
  const float* camera_features = camera_features_.data_ptr<float>();
  const float* frustum = frustum_.data_ptr<float>();

  const float* calib_pixel2camera = calib_pixel2camera_.data_ptr<float>();
  const float* calib_camera2lidar = calib_camera2lidar_.data_ptr<float>();

  const int offset = H * W;
  const int grid_xy = grid_size[0] * grid_size[1];
  const int grid_xyz = grid_size[0] * grid_size[1] * grid_size[2];

  float camera_x, camera_y, camera_z;
  float lidar_x, lidar_y, lidar_z;
  int coor_x, coor_y, coor_z;
  std::vector<float> exp_list(D, 0.0);
  std::vector<float> calib_camera2lidar_fused(12, 0.0);
  calib_camera2lidar_fused[0] = calib_camera2lidar[0] / voxel_size[0];
  calib_camera2lidar_fused[1] = calib_camera2lidar[1] / voxel_size[0];
  calib_camera2lidar_fused[2] = calib_camera2lidar[2] / voxel_size[0];
  calib_camera2lidar_fused[3] = (calib_camera2lidar[3] - point_cloud_range[0]) / voxel_size[0];

  calib_camera2lidar_fused[4] = calib_camera2lidar[4] / voxel_size[1];
  calib_camera2lidar_fused[5] = calib_camera2lidar[5] / voxel_size[1];
  calib_camera2lidar_fused[6] = calib_camera2lidar[6] / voxel_size[1];
  calib_camera2lidar_fused[7] = (calib_camera2lidar[7] - point_cloud_range[1] / voxel_size[1];

  calib_camera2lidar_fused[8] = calib_camera2lidar[8] / voxel_size[2];
  calib_camera2lidar_fused[9] = calib_camera2lidar[9] / voxel_size[2];
  calib_camera2lidar_fused[10] = calib_camera2lidar[10] / voxel_size[2];
  calib_camera2lidar_fused[11] = (calib_camera2lidar[11] - point_cloud_range[2]) / voxel_size[2];


  for(int h = 0; h < H; ++h){
    for(int w = 0; w < W; ++w){
        const float* cur_pixel_depth = depth + h * W + w;
        float sum_softmax = 0.0;
        for(int d = 0; d < D; ++d){
            exp_list[d] = exp(cur_pixel_depth[d * offset]);
            sum_softmax += exp_list[d];
        }
        sum_softmax = 1.0 / sum_softmax;
        for(int d = 0; d < D; ++d){
            float score = exp_list[d] * sum_softmax;
            if(score < score_thresh){
                continue;
            }

            const float* cur_frustum = frustum + d * offset * 3 + h * W * 3 + w * 3;
            float x = cur_frustum[0] - calib_pixel2camera[3];
            float y = cur_frustum[1] - calib_pixel2camera[7];
            float z = cur_frustum[2] - calib_pixel2camera[11];

            camera_x = x * calib_pixel2camera[0] + y * calib_pixel2camera[1] + z * calib_pixel2camera[2];
            camera_y = x * calib_pixel2camera[4] + y * calib_pixel2camera[5] + z * calib_pixel2camera[6];
            camera_z = x * calib_pixel2camera[8] + y * calib_pixel2camera[9] + z * calib_pixel2camera[10];
            camera_x *= camera_z;
            camera_y *= camera_z;

            coor_x = camera_x * calib_camera2lidar_fused[0] + camera_y * calib_camera2lidar_fused[1] + camera_z * calib_camera2lidar_fused[2] + calib_camera2lidar_fused[3];
            coor_y = camera_x * calib_camera2lidar_fused[4] + camera_y * calib_camera2lidar_fused[5] + camera_z * calib_camera2lidar_fused[6] + calib_camera2lidar_fused[7];
            coor_z = camera_x * calib_camera2lidar_fused[8] + camera_y * calib_camera2lidar_fused[9] + camera_z * calib_camera2lidar_fused[10] + calib_camera2lidar_fused[11];

            if((coor_x < 0) || (coor_x >= grid_size[0]) || (coor_y < 0) || (coor_y >= grid_size[1]) || (coor_z < 0) || (coor_z >= grid_size[2])){
                continue;
            }

            for(int c = 0; c < bev_channels; ++c){
                bev_features[c * grid_xyz + coor_z * grid_xy + coor_y * grid_size[0] + coor_x] += camera_features[c * offset + h * W + w] * score;
            }
        }
    }
  }

  return bev_features_;
}