#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, pcast_i16_to_f32
from diff_gaussian_rasterization._C import calculate_colours_variance, kmeans_cuda
import numpy as np
import math
from collections import OrderedDict
from scene.cameras import Camera
from utils.sh_utils import eval_sh
import matplotlib.pyplot as plt

class Codebook():
    def __init__(self, ids, centers):
        self.ids = ids
        self.centers = centers
    
    def evaluate(self):
        return self.centers[self.ids.flatten().long()]

def generate_codebook(values, inverse_activation_fn=lambda x: x, num_clusters=256, tol=0.0001):
    shape = values.shape
    values = values.flatten().view(-1, 1)
    centers = values[torch.randint(values.shape[0], (num_clusters, 1), device="cuda").squeeze()].view(-1,1)

    ids, centers = kmeans_cuda(values, centers.squeeze(), tol, 500)
    ids = ids.byte().squeeze().view(shape)
    centers = centers.view(-1,1)

    return Codebook(ids.cuda(), inverse_activation_fn(centers.cuda()))

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance) # 提取下三角部分
            return symm
        
        def build_covariance_from_scaling_rotation_mat(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            # symm = strip_symmetric(actual_covariance) # 提取下三角部分
            return actual_covariance

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.covariance_activation_mat = build_covariance_from_scaling_rotation_mat

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, variable_sh_bands : bool = False):
        self.active_sh_degree = 0
        self.variable_sh_bands = variable_sh_bands
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        if variable_sh_bands:
            # List that stores the individual non 0 band SH features
            # For implementation reasons, the first tensor will always have a shape of Nx0x3
            self._features_rest = list(torch.empty(0)) * (self.max_sh_degree + 1)
        else:
            self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self._degrees = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._codebook_dict = None
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self._degrees,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self._degrees) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def per_band_count(self):
        result = list()
        if self.variable_sh_bands:
            for tensor in self._features_rest:
                result.append(tensor.shape[0])
        return result

    @property
    def num_primitives(self):
        return self._xyz.shape[0]

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        if self.variable_sh_bands:
            features = list()
            index_start = 0
            for idx, sh_tensor in enumerate(self._features_rest):
                index_end = index_start + self.per_band_count[idx]
                features.append(torch.cat((self._features_dc[index_start: index_end], sh_tensor), dim=1))
                index_start = index_end
        else:
            features = torch.cat((self._features_dc, self._features_rest), dim=1)
        return features

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_color(self, view_point):
        shs_view = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
        dir_pp = (self.get_xyz - view_point.camera_center.repeat(self.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
        color_at_view = torch.clamp_min(sh2rgb + 0.5, 0.0)
        return  color_at_view
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def get_covariance_mat(self, scaling_modifier = 1):
        return self.covariance_activation_mat(self.get_scaling, scaling_modifier, self._rotation)
    
    def get_cov2D(self, viewpoint_cam, scaling_modifier=1.0):
        """
        根据相机参数计算 2D 协方差矩阵
        Args:
            viewpoint_cam: 相机对象，包含投影矩阵 (full_proj_transform)
            scaling_modifier: 缩放因子（例如用于各向异性缩放）
        Returns:
            cov2D: [N, 2, 2] 的 2D 协方差矩阵
        """
        # 1. 获取 3D 协方差矩阵 [N, 3, 3]
        cov3D = self.get_covariance_mat(scaling_modifier)  # 使用现有的 3D 协方差方法
        print(f"cov3D shape: {cov3D.shape}") 
        # 2. 提取相机投影矩阵的线性部分（忽略平移）
        # viewpoint_cam.full_proj_transform 形状为 [4, 4]
        proj_matrix = viewpoint_cam.full_proj_transform[:3, :3]  # [3, 3]
        
        # 3. 计算投影后的协方差矩阵（公式合并投影和视图变换）
        # 注意: 如果视图变换 (world_view_transform) 需要单独处理，需调整此处
        view_proj_matrix = proj_matrix @ viewpoint_cam.world_view_transform[:3, :3]  # [3,3]
        
        # 4. 计算协方差的投影变换: Σ' = M Σ M^T
        cov = view_proj_matrix @ cov3D @ view_proj_matrix.transpose(-1, -2)  # [N,3,3]
        
        # 5. 提取 XY 平面部分并添加正则化项（避免奇异矩阵）
        cov2D = cov[:, :2, :2]  # [N, 2, 2]
        cov2D[:, 0, 0] += 1e-6  # 正则化 X 方向
        cov2D[:, 1, 1] += 1e-6  # 正则化 Y 方向
        
        return cov2D
    
    def compute_uv(self, viewpoint_cam):
        """
        根据相机参数将 3D 坐标投影到 2D 图像平面
        Args:
            viewpoint_cam: 相机对象，包含投影矩阵 (full_proj_transform)
        Returns:
            uv: 投影后的归一化坐标 [N,2]，范围 [0,1]
        """
        # 将 3D 坐标转换为齐次坐标 [N,4]
        xyz_hom = torch.cat([self._xyz, torch.ones_like(self._xyz[:, :1])], dim=-1)
        
        # 应用投影矩阵
        proj_matrix = viewpoint_cam.full_proj_transform.T.to(self._xyz.device)  # [4,4]
        camera_coords = (xyz_hom @ proj_matrix)  # [N,4]
        
        # 透视除法并归一化到 [0,1]
        uv = camera_coords[:, :2] / camera_coords[:, 3:4]  # [N,2]
        uv = (uv + 1) / 2  # 从 [-1,1] 转换到 [0,1]
        return uv
    
    def get_radius(self):
        """
        根据缩放参数计算高斯点的半径
        Returns:
            radius: [N]
        """
        scaling = self.get_scaling  # [N,3]
        return torch.max(scaling, dim=1).values  # 取各向异性缩放的最大值作为半径

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            self._degrees += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        dist2 = torch.sqrt(dist2)
        scales = torch.log(dist2)[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.num_primitives), device="cuda")
        self._degrees = torch.zeros((self.num_primitives, 1), device="cuda", dtype=torch.int32)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.num_primitives, 1), device="cuda")
        self.denom = torch.zeros((self.num_primitives, 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self, rest_coeffs=45):
        return ['x', 'y', 'z',
                'f_dc_0','f_dc_1','f_dc_2',
                *[f"f_rest_{i}" for i in range(rest_coeffs)],
                'opacity',
                'scale_0','scale_1','scale_2',
                'rot_0','rot_1','rot_2','rot_3']

    def save_ply(self, path, quantised=False, half_float=False):
        float_type = 'int16' if half_float else 'f4'
        attribute_type = 'u1' if quantised else float_type
        max_sh_coeffs = (self.max_sh_degree + 1) ** 2 - 1
        mkdir_p(os.path.dirname(path))
        elements_list = []

        if quantised:
            # Read codebook dict to extract ids and centers
            if self._codebook_dict is None:
                print("Clustering codebook missing. Returning without saving")
                return

            opacity = self._codebook_dict["opacity"].ids
            scaling = self._codebook_dict["scaling"].ids
            rot = torch.cat((self._codebook_dict["rotation_re"].ids,
                            self._codebook_dict["rotation_im"].ids),
                            dim=1)
            features_dc = self._codebook_dict["features_dc"].ids
            features_rest = torch.stack([self._codebook_dict[f"features_rest_{i}"].ids
                                        for i in range(max_sh_coeffs)
                                        ], dim=1).squeeze()

            dtype_full = [(k, float_type) for k in self._codebook_dict.keys()]
            codebooks = np.empty(256, dtype=dtype_full)

            centers_numpy_list = [v.centers.detach().cpu().numpy() for v in self._codebook_dict.values()]

            if half_float:
                # No float 16 for plydata, so we just pointer cast everything to int16
                for i in range(len(centers_numpy_list)):
                    centers_numpy_list[i] = np.cast[np.float16](centers_numpy_list[i]).view(dtype=np.int16)
                
            codebooks[:] = list(map(tuple, np.concatenate([ar for ar in centers_numpy_list], axis=1)))
                
        else:
            opacity = self._opacity
            scaling = self._scaling
            rot = self._rotation
            features_dc = self._features_dc
            features_rest = self._features_rest

        for sh_degree in range(self.max_sh_degree + 1):
            coeffs_num = (sh_degree+1)**2 - 1
            degrees_mask = (self._degrees == sh_degree).squeeze()

            #  Position is not quantised
            if half_float:
                xyz = self._xyz[degrees_mask].detach().cpu().half().view(dtype=torch.int16).numpy()
            else:
                xyz = self._xyz[degrees_mask].detach().cpu().numpy()

            f_dc = features_dc[degrees_mask].detach().contiguous().cpu().view(-1,3).numpy()
            # Transpose so that to save rest featrues as rrr ggg bbb instead of rgb rgb rgb
            if self.variable_sh_bands:
                f_rest = features_rest[sh_degree][:, :coeffs_num].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            else:
                f_rest = features_rest[degrees_mask][:, :coeffs_num].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = opacity[degrees_mask].detach().cpu().numpy()
            scale = scaling[degrees_mask].detach().cpu().numpy()
            rotation = rot[degrees_mask].detach().cpu().numpy()

            dtype_full = [(attribute, float_type) 
                          if attribute in ['x', 'y', 'z'] else (attribute, attribute_type) 
                          for attribute in self.construct_list_of_attributes(coeffs_num * 3)]
            elements = np.empty(degrees_mask.sum(), dtype=dtype_full)

            attributes = np.concatenate((xyz, f_dc, f_rest, opacities, scale, rotation), axis=1)
            elements[:] = list(map(tuple, attributes))
            elements_list.append(PlyElement.describe(elements, f'vertex_{sh_degree}'))
        if quantised:
            elements_list.append(PlyElement.describe(codebooks, f'codebook_centers'))
        PlyData(elements_list).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def _parse_vertex_group(self,
                            vertex_group,
                            sh_degree,
                            float_type,
                            attribute_type,
                            max_coeffs_num,
                            quantised,
                            half_precision,                            
                            codebook_centers_torch=None):
        coeffs_num = (sh_degree+1)**2 - 1
        num_primitives = vertex_group.count

        xyz = np.stack((np.asarray(vertex_group["x"], dtype=float_type),
                        np.asarray(vertex_group["y"], dtype=float_type),
                        np.asarray(vertex_group["z"], dtype=float_type)), axis=1)

        opacity = np.asarray(vertex_group["opacity"], dtype=attribute_type)[..., np.newaxis]
    
        # Stacks the separate components of a vector attribute into a joint numpy array
        # Defined just to avoid visual clutter
        def stack_vector_attribute(name, count):
            return np.stack([np.asarray(vertex_group[f"{name}_{i}"], dtype=attribute_type)
                            for i in range(count)], axis=1)

        features_dc = stack_vector_attribute("f_dc", 3).reshape(-1, 1, 3)
        scaling = stack_vector_attribute("scale", 3)
        rotation = stack_vector_attribute("rot", 4)
        
        # Take care of error when trying to stack 0 arrays
        if sh_degree > 0:
            features_rest = stack_vector_attribute("f_rest", coeffs_num*3).reshape((num_primitives, 3, coeffs_num))
        else:
            features_rest = np.empty((num_primitives, 3, 0), dtype=attribute_type)

        if not self.variable_sh_bands:
            # Using full tensors (P x 15) even for points that don't require it
            features_rest = np.concatenate(
                (features_rest,
                    np.zeros((num_primitives, 3, max_coeffs_num - coeffs_num), dtype=attribute_type)), axis=2)

        degrees = np.ones(num_primitives, dtype=np.int32)[..., np.newaxis] * sh_degree

        xyz = torch.from_numpy(xyz).cuda()
        if half_precision:
            xyz = pcast_i16_to_f32(xyz)
        features_dc = torch.from_numpy(features_dc).contiguous().cuda()
        features_rest = torch.from_numpy(features_rest).contiguous().cuda()
        opacity = torch.from_numpy(opacity).cuda()
        scaling = torch.from_numpy(scaling).cuda()
        rotation = torch.from_numpy(rotation).cuda()
        degrees = torch.from_numpy(degrees).cuda()

        # If quantisation has been used, it is needed to index the centers
        if quantised:
            features_dc = codebook_centers_torch['features_dc'][features_dc.view(-1).long()].view(-1, 1, 3)

            # This is needed as we might have padded the features_rest tensor with zeros before
            reshape_channels = coeffs_num if self.variable_sh_bands else max_coeffs_num            
            # The gather operation indexes a 256x15 tensor with a (P*3)features_rest index tensor,
            # in a column-wise fashion
            # Basically this is equivalent to indexing a single codebook with a P*3 index
            # features_rest times inside a loop
            features_rest = codebook_centers_torch['features_rest'].gather(0, features_rest.view(num_primitives*3, reshape_channels).long()).view(num_primitives, 3, reshape_channels)
            opacity = codebook_centers_torch['opacity'][opacity.long()]
            scaling = codebook_centers_torch['scaling'][scaling.view(num_primitives*3).long()].view(num_primitives, 3)
            # Index the real and imaginary part separately
            rotation = torch.cat((
                codebook_centers_torch['rotation_re'][rotation[:, 0:1].long()],
                codebook_centers_torch['rotation_im'][rotation[:, 1:].reshape(num_primitives*3).long()].view(num_primitives,3)
                ), dim=1)

        return {'xyz': xyz,
                'opacity': opacity,
                'features_dc': features_dc,
                'features_rest': features_rest,
                'scaling': scaling,
                'rotation': rotation,
                'degrees': degrees
        }

    def load_ply(self, path, half_float=False, quantised=False):
        plydata = PlyData.read(path)

        xyz_list = []
        features_dc_list = []
        features_rest_list = []
        opacity_list = []
        scaling_list = []
        rotation_list = []
        degrees_list = []

        float_type = 'int16' if half_float else 'f4'
        attribute_type = 'u1' if quantised else float_type
        max_coeffs_num = (self.max_sh_degree+1)**2 - 1

        codebook_centers_torch = None
        if quantised:
            # Parse the codebooks.
            # The layout is 256 x 20, where 256 is the number of centers and 20 number of codebooks
            # In the future we could have different number of centers
            codebook_centers = plydata.elements[-1]

            codebook_centers_torch = OrderedDict()
            codebook_centers_torch['features_dc'] = torch.from_numpy(np.asarray(codebook_centers['features_dc'], dtype=float_type)).cuda()
            codebook_centers_torch['features_rest'] = torch.from_numpy(np.concatenate(
                [
                    np.asarray(codebook_centers[f'features_rest_{i}'], dtype=float_type)[..., np.newaxis]
                    for i in range(max_coeffs_num)
                ], axis=1)).cuda()
            codebook_centers_torch['opacity'] = torch.from_numpy(np.asarray(codebook_centers['opacity'], dtype=float_type)).cuda()
            codebook_centers_torch['scaling'] = torch.from_numpy(np.asarray(codebook_centers['scaling'], dtype=float_type)).cuda()
            codebook_centers_torch['rotation_re'] = torch.from_numpy(np.asarray(codebook_centers['rotation_re'], dtype=float_type)).cuda()
            codebook_centers_torch['rotation_im'] = torch.from_numpy(np.asarray(codebook_centers['rotation_im'], dtype=float_type)).cuda()

            # If use half precision then we have to pointer cast the int16 to float16
            # and then cast them to floats, as that's the format that our renderer accepts
            if half_float:
                for k, v in codebook_centers_torch.items():
                    codebook_centers_torch[k] = pcast_i16_to_f32(v)

        # Iterate over the point clouds that are stored on top level of plyfile
        # to get the various fields values 
        for sh_degree in range(0, self.max_sh_degree+1):
            attribues_dict = self._parse_vertex_group(plydata.elements[sh_degree],
                                                      sh_degree,
                                                      float_type,
                                                      attribute_type,
                                                      max_coeffs_num,
                                                      quantised,
                                                      half_float,
                                                      codebook_centers_torch)

            xyz_list.append(attribues_dict['xyz'])
            features_dc_list.append(attribues_dict['features_dc'])
            features_rest_list.append(attribues_dict['features_rest'].transpose(1,2))
            opacity_list.append(attribues_dict['opacity'])
            scaling_list.append(attribues_dict['scaling'])
            rotation_list.append(attribues_dict['rotation'])
            degrees_list.append(attribues_dict['degrees'])

        # Concatenate the tensors into one, to be used in our program
        # TODO: allow multiple PCDs to be rendered/optimise and skip this step
        xyz = torch.cat((xyz_list), dim=0)
        features_dc = torch.cat((features_dc_list), dim=0)
        if not self.variable_sh_bands:
            features_rest = torch.cat((features_rest_list), dim=0)
        else:
            features_rest = features_rest_list
        opacity = torch.cat((opacity_list), dim=0)
        scaling = torch.cat((scaling_list), dim=0)
        rotation = torch.cat((rotation_list), dim=0)
        
        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        if not self.variable_sh_bands:
            self._features_rest = nn.Parameter(features_rest.requires_grad_(True))
        else:
            for tensor in features_rest_list:
                tensor.requires_grad_(True)
            self._features_rest = features_rest_list
        self._opacity = nn.Parameter(opacity.requires_grad_(True))
        self._scaling = nn.Parameter(scaling.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))
        self._degrees = torch.cat((degrees_list), dim=0)

        self.active_sh_degree = self.max_sh_degree

    # Replaces a parameter group's tensor defined by name with a given tensor
    # and zeroes-out the internal optimiser state
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask, store_grads=False):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                if store_grads:
                    grad = group["params"][0].grad[mask]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                if store_grads:
                    group["params"][0].grad = grad
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def mercy_points(self, densification_statistics_dict, lambda_mercy=2, mercy_minimum=2, mercy_type='redundancy_opacity'):
        mean = self._splatted_num_accum.squeeze().float().mean(dim=0, keepdim=True)
        std =  self._splatted_num_accum.squeeze().float().var(dim=0, keepdim=True).sqrt()

        threshold = max((mean + lambda_mercy*std).item(), mercy_minimum)

        mask = (self._splatted_num_accum > threshold).squeeze()

        if mercy_type == 'redundancy_opacity':
            # Prune redundant points based on 50% lowest opacity
            mask[mask.clone()] = self.get_opacity[mask].squeeze() < self.get_opacity[mask].median()
        elif mercy_type == 'redundancy_random':
            # Prune 50% redundant points at random
            mask[mask.clone()] = torch.rand(mask[mask].shape, device="cuda").squeeze() < 0.5
        elif mercy_type == 'opacity':
            # Prune based just on opacity
            threshold = self.get_opacity.quantile(0.045)
            mask = (self.get_opacity < threshold).squeeze()
        elif mercy_type == 'redundancy_opacity_opacity':
            # Prune based on opacity and on redundancy + opacity (options 1 and 3)
            mask[mask.clone()] = self.get_opacity[mask].squeeze() < self.get_opacity[mask].median()
            threshold = torch.min(self.get_opacity.quantile(0.03), torch.tensor([0.05], device="cuda"))
            mask = torch.logical_or(mask, (self.get_opacity < threshold).squeeze())
            
        self.prune_points(mask)
        densification_statistics_dict["n_points_mercied"] = mask.sum()
        densification_statistics_dict["redundancy_threshold"] = mean + lambda_mercy*std
        densification_statistics_dict["opacity_threshold"] = threshold if mercy_type in ['redundancy_opacity_opacity', 'opacity'] else 0

    def project_points_to_depth(self,  viewpoint_cam: Camera) -> torch.Tensor:
        """
        将3D点投影到当前相机视角,计算每个点的理论深度
        参数:
            viewpoint_cam: 当前相机参数
        返回:
            proj_depth: 每个点在当前视角下的理论深度 [N]
        """
        # # 获取点云的世界坐标
        # world_points = self.get_xyz  # [N,3]
        # # 转换为相机坐标系
        # camera_center = viewpoint_cam.camera_center.to(world_points.device)
        # view_dir = world_points - camera_center
        # device = world_points.device

        # # 同样处理 viewpoint_cam.R 和 viewpoint_cam.T
        # # R = viewpoint_cam.R
        # # T = viewpoint_cam.T
        # R = torch.from_numpy(viewpoint_cam.R).to(device)
        # T = torch.from_numpy(viewpoint_cam.T).to(device)
        # R = R.to(torch.float32)
        # T = T.to(torch.float32)
        # camera_coords = (view_dir @ R.T) + T  # [N,3]
        # # camera_coords = (view_dir @ viewpoint_cam.R.T) + viewpoint_cam.T  # [N,3]
        # # 提取深度值（相机坐标系Z轴）
        # proj_depth = camera_coords[:, 2]  # [N]
        # return proj_depth

        # world_points = self.get_xyz
        # # === 使用world_view_transform进行坐标变换 ===
        # # 获取世界到相机的4x4变换矩阵
        # world_view_transform = viewpoint_cam.world_view_transform.to(world_points.device)  # [4,4]
        
        # # 转换为齐次坐标 [N,4]
        # homogeneous_coords = torch.cat([world_points, torch.ones_like(world_points[:, :1])], dim=1)
        
        # # 应用变换矩阵: [N,4] @ [4,4] -> [N,4]
        # camera_coords_homogeneous = homogeneous_coords @ world_view_transform.T
        # camera_coords = camera_coords_homogeneous[:, :3]  # [N,3]
        
        # # 提取深度值（相机坐标系Z轴）
        # proj_depth = camera_coords[:, 2]  # [N]
        # return proj_depth

        world_points = self.get_xyz  # [N,3]
        world_view_transform = viewpoint_cam.world_view_transform.to(world_points.device)
        
        homogeneous_coords = torch.cat([world_points, torch.ones_like(world_points[:, :1])], dim=1)
        camera_coords_homogeneous = homogeneous_coords @ world_view_transform.T
        camera_coords = camera_coords_homogeneous[:, :3]
        
        return camera_coords[:, 2]  # 直接返回相机坐标系Z轴值
    
    def project_to_image(self,  viewpoint_cam: Camera):
        world_points = self.get_xyz
        device = world_points.device
        H = viewpoint_cam.image_height
        W = viewpoint_cam.image_width
        # === 1. 转换到相机坐标系 ===
        # R = viewpoint_cam.R.to(device)  # [3,3]
        R = torch.from_numpy(viewpoint_cam.R).to(device)
        # T = viewpoint_cam.T.to(device)  # [3]
        T = torch.from_numpy(viewpoint_cam.T).to(device)
        camera_center = viewpoint_cam.camera_center.to(device)
        
        # 计算相对位置
        view_dir = world_points - camera_center  # [N,3]
        # view_dir = view_dir.to(torch.float32)
        R = R.to(torch.float32)
        T = T.to(torch.float32)
        # 应用旋转和平移
        camera_coords = (view_dir @ R.T) + T  # [N,3]
        
        # === 2. 投影到图像平面 ===
        # 获取内参（假设相机参数类包含这些属性）
        fx = W / (2.0 * math.tan(viewpoint_cam.FoVx * 0.5))  # 水平焦距
        fy = H / (2.0 * math.tan(viewpoint_cam.FoVy * 0.5))  # 垂直焦距
        cx = W / 2  # 光心x（假设图像中心）
        cy = H / 2  # 光心y
        
        # 归一化坐标
        z = camera_coords[:, 2]  # [N]
        x = (camera_coords[:, 0] / z) * fx + cx  # [N]
        y = (camera_coords[:, 1] / z) * fy + cy  # [N]
        
        uv = torch.stack([x, y], dim=1)  # [N,2]
        
        # === 3. 有效性判断 ===
        # 深度正值判断
        valid_z = z > 0  # [N]
        
        # 图像边界判断
        valid_x = (uv[:, 0] >= 0) & (uv[:, 0] < W)
        valid_y = (uv[:, 1] >= 0) & (uv[:, 1] < H)
        
        valid_mask = valid_z & valid_x & valid_y  # [N]
        valid_mask = valid_z
        # print("valid_z:",valid_z,"valid_x:",valid_x,"valid_y:",valid_y)
        return uv, valid_mask

    
    def depth_aware_prune(self, depth_map, viewpoint_cam: Camera, threshold: float = 10.0, opacity_thresh: float = 1/255) -> None:
        """
        深度感知的点云修剪
        参数:
            threshold: 深度误差阈值（米）
            opacity_thresh: 不透明度阈值
        """
        # 计算每个点的理论投影深度
        proj_depth = self.project_points_to_depth(viewpoint_cam)
        
        # 获取点云可见性信息
        uv, valid_mask = self.project_to_image(viewpoint_cam)

        # 提取深度图对应位置的深度值（双线性插值）
        H, W = depth_map.shape[0], depth_map.shape[1]
        uv_normalized = torch.stack([
            2 * (uv[:, 0].float() / (W - 1)) - 1,
            2 * (uv[:, 1].float() / (H - 1)) - 1
        ], dim=1).view(1, 1, -1, 2)  # 调整形状适配grid_sample
        
        depth_values = torch.nn.functional.grid_sample(
            depth_map.unsqueeze(0).unsqueeze(0).float(),  # [1,1,H,W]
            uv_normalized,                                # [1,1,N,2]
            mode='bilinear',
            align_corners=True
        ).squeeze()  # [N]
        depth_values = depth_values.view(-1) 
        # 初始化误差张量（默认无穷大表示不可见点）
        depth_error = torch.full_like(proj_depth, float('inf'))
        # depth_error = torch.zeros_like(proj_depth)
        print("proj_depth:",proj_depth.shape)
        # print("proj_depth:",proj_depth)
        print("depth_map:",depth_map.shape)
        depth_error[valid_mask] = (proj_depth[valid_mask] - depth_values[valid_mask]).abs()
        # print("uv:",uv)
        print("valid_mask:",valid_mask)
        print("valid_mask:",valid_mask.shape)
        print("depth_error:",depth_error)
        print("depth_error:",depth_error.shape)
        # 创建修剪掩码
        # print('depth_error:',depth_error)
        prune_mask = (depth_error > threshold) | (self.get_opacity.squeeze() < opacity_thresh)
        # prune_mask = (depth_error > threshold)
        
        # 执行修剪
        self.prune_points(prune_mask)

    def multi_view_color_based_pruning(self, viewpoints, low_threshold=1e-10):
        pos = self.get_xyz  # (N, 3)
        num_views = len(viewpoints)
        num_points = pos.shape[0]
        
        # 初始化一个存储每个点颜色变化的张量
        color_changes = torch.zeros(num_points, device=pos.device)

        # 初始化颜色存储
        all_colors = torch.zeros((num_points, num_views, 3), device=pos.device)

        # 对每个点，计算其在多个视角下的颜色变化        
        for j, view_point in enumerate(viewpoints):
            
            # 在每个视角下渲染该点（这里假设我们有一个渲染函数）
            shs_view = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
            dir_pp = (self.get_xyz - view_point.camera_center.repeat(self.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
            color_at_view = torch.clamp_min(sh2rgb + 0.5, 0.0)
            
            all_colors[:, j, :] = color_at_view  # 记录所有视角的颜色
                
        # 计算颜色的方差
        color_variance = torch.var(all_colors, dim=1)  # (N, 3)，在视角维度计算方差
        # 计算颜色熵
        eps = 1e-8
        prob = all_colors.mean(dim=1)
        prob_sum = prob.sum(dim=1, keepdim=True) + eps  # 避免除 0
        prob = prob / prob_sum
        color_entropy = -torch.sum(prob * torch.log(prob + eps), dim=1)  # 计算熵

        # 计算三通道 RGB 方差的均值，得到最终的变化度量
        color_changes = color_variance.mean(dim=1)  # (N,)
        
        # 使用颜色变化的度量进行剪枝：变化小的点被删除
        high_threshold = torch.quantile(color_changes, 0.95)  # 取 98% 分位数
        # prune_mask = (color_changes < low_threshold) | (color_changes > high_threshold)
        threshold=0.01
        prune_mask = (color_changes < low_threshold) | ((color_changes > high_threshold) & (color_entropy < threshold))
        print("num_points:",num_points)
        self.prune_points(prune_mask)

    def multi_view_entropy_based_pruning(self, viewpoints, threshold=0.01):
        pos = self.get_xyz  # (N, 3) 高斯点坐标
        num_views = len(viewpoints)  # 视角数量
        num_points = pos.shape[0]  # 高斯点数量

        # 初始化颜色存储
        all_colors = torch.zeros((num_points, num_views, 3), device=pos.device)

        for j, view_point in enumerate(viewpoints):
            # 在每个视角下渲染该点
            shs_view = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
            dir_pp = (self.get_xyz - view_point.camera_center.repeat(self.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
            color_at_view = torch.clamp_min(sh2rgb + 0.5, 0.0)

            all_colors[:, j, :] = color_at_view  # 记录所有视角的颜色

        # 计算颜色熵
        eps = 1e-8
        prob = all_colors.mean(dim=1)
        prob_sum = prob.sum(dim=1, keepdim=True) + eps  # 避免除 0
        prob = prob / prob_sum
        color_entropy = -torch.sum(prob * torch.log(prob + eps), dim=1)  # 计算熵

        # 低熵点（颜色变化小）被剪枝
        prune_mask = color_entropy < threshold
        print("num_points:", num_points)
        self.prune_points(prune_mask)


    def prune_points(self, mask, store_grads=False):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask, store_grads=store_grads)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._degrees = self._degrees[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict, store_grads=False):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                if store_grads:
                    grad = torch.cat((group["params"][0].grad, torch.zeros_like(extension_tensor)), dim=0)
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                if store_grads:
                    group["params"][0].grad = grad
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                if store_grads:
                    grad = torch.cat((group["params"][0].grad, torch.zeros_like(extension_tensor)), dim=0)
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                if store_grads:
                    group["params"][0].grad = grad
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_degrees, store_grads=False):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d, store_grads=store_grads)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._degrees = torch.cat((self._degrees, new_degrees), dim=0)

        self.xyz_gradient_accum = torch.zeros((self.num_primitives, 1), device="cuda")
        self.density_gradient_accum = torch.zeros((self.num_primitives, 1), device="cuda")
        self.denom = torch.zeros((self.num_primitives, 1), device="cuda")
        self.max_radii2D = torch.zeros((self.num_primitives), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, store_grads=False):
        # The pads are used to ignore the densification that happened before (densify_and_clone)
        n_init_points = self.num_primitives
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = padded_grad >= grad_threshold
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        n_points_split = selected_pts_mask.sum().item()

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_degrees = self._degrees[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_degrees, store_grads=store_grads)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter, store_grads)
        return n_points_split

    def densify_and_clone(self, grads, grad_threshold, scene_extent, store_grads=False):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = grads.squeeze() >= grad_threshold
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        n_points_cloned = selected_pts_mask.sum().item()
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_degrees = self._degrees[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_degrees, store_grads=store_grads)
        return n_points_cloned

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, densification_statistics_dict, store_grads=False):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        n_points_cloned = self.densify_and_clone(grads, max_grad, extent, store_grads=store_grads)
        n_points_split = self.densify_and_split(grads, max_grad, extent, store_grads=store_grads)

        self.prune(min_opacity, extent, max_screen_size, densification_statistics_dict, store_grads=store_grads)

        torch.cuda.empty_cache()

        densification_statistics_dict["n_points_cloned"] = n_points_cloned
        densification_statistics_dict["n_points_split"] = n_points_split

    def prune(self, min_opacity, extent, max_screen_size, densification_statistics_dict, store_grads=False):
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        densification_statistics_dict["n_points_pruned"] = prune_mask.sum()
        self.prune_points(prune_mask, store_grads=store_grads)

    def prune_mean(self, min_opacity, extent, max_screen_size, densification_statistics_dict, store_grads=False):

        # 获取不透明度张量并计算符合条件的均值
        opacity_values = self.get_opacity
        valid_opacity_mask = opacity_values < min_opacity
        valid_opacity_values = opacity_values[valid_opacity_mask]
        
        # 计算符合条件的不透明度均值（处理可能为空的情况）
        valid_opacity_mean = valid_opacity_values.mean().item() if valid_opacity_values.numel() > 0 else 0.0
        
        # 写入日志文件（如果指定）
        log_file = "./opacity_mean_values.txt"
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"Prune Opacity Mean: {valid_opacity_mean:.6f}\n")
    

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        densification_statistics_dict["n_points_pruned"] = prune_mask.sum()
        self.prune_points(prune_mask, store_grads=store_grads)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum += torch.norm(viewspace_point_tensor.grad[:, :2], dim=-1, keepdim=True)
        self.denom += update_filter.unsqueeze(1)

    def _low_variance_colour_culling(self, threshold, weighted_variance, weighted_mean):
        original_degrees = torch.zeros_like(self._degrees)
        original_degrees.copy_(self._degrees)

        # Uniform colour culling
        weighted_colour_std = weighted_variance.sqrt()
        weighted_colour_std[weighted_colour_std.isnan()] = 0
        weighted_colour_std = weighted_colour_std.mean(dim=2).squeeze()

        std_mask = weighted_colour_std < threshold
        print('std_mask:',std_mask)
        self._features_dc[std_mask] = (weighted_mean[std_mask] - 0.5) / 0.28209479177387814
        self._degrees[std_mask] = 0
        self._features_rest[std_mask] = 0

    def _low_distance_colour_culling(self, threshold, colour_distances):
        colour_distances[colour_distances.isnan()] = 0

        # Loop from active_sh_degree - 1 to 0, since the comparisons
        # are always done based on the max band that corresponds to active_sh_degree
        for sh_degree in range(self.active_sh_degree - 1, 0, -1):
            coeffs_num = (sh_degree+1)**2 - 1
            mask = colour_distances[:, sh_degree] < threshold
            print('mask:',mask)
            self._degrees[mask] = torch.min(
                    torch.tensor([sh_degree], device="cuda", dtype=int),
                    self._degrees[mask]
                ).int()
            
            # Zero-out the associated SH coefficients for clarity,
            # as they won't be used in rasterisation due to the degrees field
            self._features_rest[mask, coeffs_num:] = 0

    def cull_sh_bands(self, cameras, threshold=0*np.sqrt(3)/255, std_threshold=0.):
        camera_positions = torch.stack([cam.camera_center for cam in cameras], dim=0)
        camera_viewmatrices = torch.stack([cam.world_view_transform for cam in cameras], dim=0)
        camera_projmatrices = torch.stack([cam.full_proj_transform for cam in cameras], dim=0)
        camera_fovx = torch.tensor([camera.FoVx for camera in cameras], device="cuda", dtype=torch.float32)
        camera_fovy = torch.tensor([camera.FoVy for camera in cameras], device="cuda", dtype=torch.float32)
        image_height = torch.tensor([camera.image_height for camera in cameras], device="cuda", dtype=torch.int32)
        image_width = torch.tensor([camera.image_width for camera in cameras], device="cuda", dtype=torch.int32)

        # Wrapping in a function since it's called with the same parameters twice
        def run_calculate_colours_variance():
            return calculate_colours_variance(
                camera_positions,
                self.get_xyz,
                self._opacity,
                self.get_scaling,
                self.get_rotation,
                camera_viewmatrices,
                camera_projmatrices,
                torch.tan(camera_fovx*0.5),
                torch.tan(camera_fovy*0.5),
                image_height,
                image_width,
                self.get_features,
                self._degrees,
                self.active_sh_degree)
        
        _, weighted_variance, weighted_mean = run_calculate_colours_variance()
        # print('weighted_variance:',weighted_variance)
        # print('weighted_mean:',weighted_mean)
        self._low_variance_colour_culling(std_threshold, weighted_variance, weighted_mean)

        # Recalculate to account for the changed values
        colour_distances, _, _ = run_calculate_colours_variance()
        # print('colour_distances:',colour_distances)
        self._low_distance_colour_culling(threshold, colour_distances)

    def produce_clusters(self, num_clusters=256, store_dict_path=None):
        max_coeffs_num = (self.max_sh_degree + 1)**2 - 1
        codebook_dict = OrderedDict({})

        codebook_dict["features_dc"] = generate_codebook(self._features_dc.detach()[:, 0],
                                                         num_clusters=num_clusters, tol=0.001)
        for sh_degree in range(max_coeffs_num):
                codebook_dict[f"features_rest_{sh_degree}"] = generate_codebook(
                    self._features_rest.detach()[:, sh_degree], num_clusters=num_clusters)

        codebook_dict["opacity"] = generate_codebook(self.get_opacity.detach(),
                                                     self.inverse_opacity_activation, num_clusters=num_clusters)
        codebook_dict["scaling"] = generate_codebook(self.get_scaling.detach(),
                                                     self.scaling_inverse_activation, num_clusters=num_clusters)
        codebook_dict["rotation_re"] = generate_codebook(self.get_rotation.detach()[:, 0:1],
                                                         num_clusters=num_clusters)
        codebook_dict["rotation_im"] = generate_codebook(self.get_rotation.detach()[:, 1:],
                                                         num_clusters=num_clusters)
        if store_dict_path is not None:
            torch.save(codebook_dict, os.path.join(store_dict_path, 'codebook.pt'))
        
        self._codebook_dict = codebook_dict

    def apply_clustering(self, codebook_dict=None):
        max_coeffs_num = (self.max_sh_degree + 1)**2 - 1
        if codebook_dict is None:
            return

        opacity = codebook_dict["opacity"].evaluate().requires_grad_(True)
        scaling = codebook_dict["scaling"].evaluate().view(-1, 3).requires_grad_(True)
        rotation = torch.cat((codebook_dict["rotation_re"].evaluate(),
                            codebook_dict["rotation_im"].evaluate().view(-1, 3)),
                            dim=1).squeeze().requires_grad_(True)
        features_dc = codebook_dict["features_dc"].evaluate().view(-1, 1, 3).requires_grad_(True)
        features_rest = []
        for sh_degree in range(max_coeffs_num):
            features_rest.append(codebook_dict[f"features_rest_{sh_degree}"].evaluate().view(-1, 3))

        features_rest = torch.stack([*features_rest], dim=1).squeeze().requires_grad_(True)

        with torch.no_grad():
            self._opacity = opacity
            self._scaling = scaling
            self._rotation = rotation
            self._features_dc = features_dc
            self._features_rest = features_rest