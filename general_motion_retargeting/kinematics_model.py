import xml.etree.ElementTree as ET

import numpy as np
import torch
from scipy.signal import savgol_filter
from scipy.interpolate import PchipInterpolator, CubicSpline

from . import torch_utils


class Joint:
    def __init__(self, name, dof_dim, axis):
        self._name = name
        self._dof_dim = dof_dim
        self._axis = axis
        self._dof_idx = -1 # indicate the start index of dof in the whole dof vector, -1 for root or no dof joint
        
    def set_dof_idx(self, dof_idx):
        if self._dof_dim == 0:
            raise ValueError('Joint {} has no dof'.format(self._name))
        self._dof_idx = dof_idx
        
    def dof_to_rot(self, dof):
        # Input dof shape: [..., dof_dim]
        # Output rot shape: [..., 4]
        # Function: convert 1-dim or 3-dim dof to quaternion
        rot_shape = list(dof.shape[:-1]) + [4]
        ret_rot = torch.zeros(rot_shape, dtype=dof.dtype, device=dof.device)
        if self._dof_dim == 0:
            ret_rot[..., -1] = 1.0
        elif self._dof_dim == 1:
            axis = self._axis # shape: [3]
            axis = torch.broadcast_to(axis, ret_rot[..., 0:3].shape)
            ret_rot[:] = torch_utils.axis_angle_to_quat(axis, dof.squeeze(-1))
        elif self._dof_dim == 3:
            ret_rot[:] = torch_utils.exp_map_to_quat(dof)
            
        return ret_rot
    
    def rot_to_dof(self, rot):
        # Input rot shape: [..., 4]
        # Output dof shape: [..., dof_dim]
        # Function: convert quaternion to 1-dim or 3-dim dof
        dof_shape = list(rot.shape[:-1]) + [self._dof_dim]
        ret_dof = torch.zeros(dof_shape, dtype=rot.dtype, device=rot.device)
        if self._dof_dim == 1:
            axis = self._axis
            axis, angle = torch_utils.quat_to_axis_angle(rot)
            dot_axis = torch.sum(axis * self._axis, dim=-1)
            angle[dot_axis < 0] *= -1
            ret_dof[:] = angle.unsqueeze(-1)
        elif self._dof_dim == 3:
            ret_dof[:] = torch_utils.quat_to_exp_map(rot)
            
        return ret_dof
    
    @property
    def dof_dim(self):
        return self._dof_dim
    
    @property
    def name(self):
        return self._name
    
    @property
    def dof_idx(self):
        return self._dof_idx
    
    
class KinematicsModel:
    def __init__(self, file_path, device):
        self._device = device
        self._file_path = file_path
        
        self._build_kinematics_model()
        self._set_dof_indices()
        
    def _build_kinematics_model(self):
        self._body_names = []
        self._dof_names = []
        self._parent_indices = []
        self._local_translation = []
        self._local_rotation = []
        self._joints = []
        self._dof_size = []
        self._dof_upper_limits = []
        self._dof_lower_limits = []
        
        if self._file_path.endswith('.xml'):
            self._parse_xml()
        else:
            raise NotImplementedError('File type not supported')
        
        self._parent_indices = torch.tensor(self._parent_indices, dtype=torch.long, device=self._device)
        self._local_translation = torch.tensor(np.array(self._local_translation), dtype=torch.float, device=self._device)
        self._local_rotation = torch.tensor(np.array(self._local_rotation), dtype=torch.float, device=self._device)
        self._num_dof = sum(self._dof_size)
        self._dof_lower_limits = torch.tensor(self._dof_lower_limits, dtype=torch.float, device=self._device)
        self._dof_upper_limits = torch.tensor(self._dof_upper_limits, dtype=torch.float, device=self._device)
        if self._rot_unit == "degree":
            self._dof_lower_limits = torch.deg2rad(self._dof_lower_limits)
            self._dof_upper_limits = torch.deg2rad(self._dof_upper_limits)

    def _find_local_maxima_indices(self, z, resolution_factor=100):
        if len(z) < 3:
            raise ValueError("Input array must have at least 3 elements")
        
        t = len(z)
        x_original = np.arange(t)
        cs = CubicSpline(x_original, z)
        
        def cs_derivative(x):
            return cs(x, 1)
        
        def cs_second_derivative(x):
            return cs(x, 2)
        
        potential_maxima = []
        for i in range(t - 1):
            x_fine = np.linspace(i, i + 1, resolution_factor)
            derivative_values = cs_derivative(x_fine)
            
            for j in range(len(x_fine) - 1):
                if derivative_values[j] * derivative_values[j + 1] <= 0:
                    if derivative_values[j] == 0:
                        candidate = x_fine[j]
                    elif derivative_values[j + 1] == 0:
                        candidate = x_fine[j + 1]
                    else:
                        left = x_fine[j]
                        right = x_fine[j + 1]
                        for _ in range(10):
                            mid = (left + right) / 2
                            if cs_derivative(left) * cs_derivative(mid) <= 0:
                                right = mid
                            else:
                                left = mid
                        candidate = (left + right) / 2
                    
                    if cs_second_derivative(candidate) < 0:
                        potential_maxima.append(candidate)
        
        if potential_maxima:
            potential_maxima = np.array(potential_maxima)
            unique_maxima = []
            threshold = 0.5
            
            sorted_indices = np.argsort(potential_maxima)
            for idx in sorted_indices:
                x = potential_maxima[idx]
                if not unique_maxima or abs(x - unique_maxima[-1]) > threshold:
                    unique_maxima.append(x)
            
            maxima_indices = []
            for x_max in unique_maxima:
                nearest_idx = np.argmin(np.abs(x_original - x_max))
                if nearest_idx == 0:
                    if z[nearest_idx] > z[nearest_idx + 1]:
                        maxima_indices.append(nearest_idx)
                elif nearest_idx == t - 1:
                    if z[nearest_idx] > z[nearest_idx - 1]:
                        maxima_indices.append(nearest_idx)
                else:
                    if (z[nearest_idx] > z[nearest_idx - 1] and 
                        z[nearest_idx] > z[nearest_idx + 1]):
                        maxima_indices.append(nearest_idx)
            
            maxima_indices = sorted(set(maxima_indices))
            return maxima_indices
        else:
            return []

    def _find_local_minima_indices(self, z, resolution_factor=100):
        if len(z) < 3:
            raise ValueError("Input array must have at least 3 elements")
        
        t = len(z)
        x_original = np.arange(t)
        cs = CubicSpline(x_original, z)
        
        def cs_derivative(x):
            return cs(x, 1)
        
        def cs_second_derivative(x):
            return cs(x, 2)
        
        potential_minima = []
        for i in range(t - 1):
            x_fine = np.linspace(i, i + 1, resolution_factor)
            derivative_values = cs_derivative(x_fine)
            
            for j in range(len(x_fine) - 1):
                if derivative_values[j] * derivative_values[j + 1] <= 0:
                    if derivative_values[j] == 0:
                        candidate = x_fine[j]
                    elif derivative_values[j + 1] == 0:
                        candidate = x_fine[j + 1]
                    else:
                        left = x_fine[j]
                        right = x_fine[j + 1]
                        for _ in range(10):
                            mid = (left + right) / 2
                            if cs_derivative(left) * cs_derivative(mid) <= 0:
                                right = mid
                            else:
                                left = mid
                        candidate = (left + right) / 2
                    
                    if cs_second_derivative(candidate) > 0:
                        potential_minima.append(candidate)
        
        if potential_minima:
            potential_minima = np.array(potential_minima)
            unique_minima = []
            threshold = 0.5
            
            sorted_indices = np.argsort(potential_minima)
            for idx in sorted_indices:
                x = potential_minima[idx]
                if not unique_minima or abs(x - unique_minima[-1]) > threshold:
                    unique_minima.append(x)
            
            minima_indices = []
            for x_min in unique_minima:
                nearest_idx = np.argmin(np.abs(x_original - x_min))
                if nearest_idx == 0:
                    if z[nearest_idx] < z[nearest_idx + 1]:
                        minima_indices.append(nearest_idx)
                elif nearest_idx == t - 1:
                    if z[nearest_idx] < z[nearest_idx - 1]:
                        minima_indices.append(nearest_idx)
                else:
                    if (z[nearest_idx] < z[nearest_idx - 1] and 
                        z[nearest_idx] < z[nearest_idx + 1]):
                        minima_indices.append(nearest_idx)
            
            minima_indices = sorted(set(minima_indices))
            return minima_indices
        else:
            return []

    def _get_min_body_z_from_qpos(self, qpos):
        # Forward kinematics to get body positions
        root_pos = qpos[:3]
        dof_pos = qpos[3:]
        body_pos, _ = self.forward_kinematics(
            torch.from_numpy(root_pos).unsqueeze(0).to(device=self._device, dtype=torch.float),
            torch.zeros(1, 4).to(device=self._device, dtype=torch.float),  # We ignore rotation for this calculation
            torch.from_numpy(dof_pos).unsqueeze(0).to(device=self._device, dtype=torch.float)
        )
        
        # Get minimum z value across all body parts
        z_vals = body_pos[0, :, 2].cpu().numpy()  # Take z coordinates of all bodies
        min_idx = int(np.argmin(z_vals))
        min_z = float(z_vals[min_idx])
        body_id = min_idx
        body_name = self._body_names[min_idx]

        return min_z, body_id, body_name

    def adjust_root_z(self, qpos, window=1, vel_thresh=5e-4):
        """
        Adjusts the root z position to ensure the robot's lowest body part stays above ground.
        """
        import copy
        qpos = copy.deepcopy(qpos)
        T, N = qpos.shape

        root_z = qpos[:, 2]
        root_vel = np.zeros_like(root_z)
        root_vel[:-1] = root_z[1:] - root_z[:-1]

        rebuild_root_z = np.zeros_like(root_z)
        minidxes = self._find_local_minima_indices(qpos[:,2])
        maxidxes = self._find_local_maxima_indices(qpos[:,2])
        rebuild_root_z[0] = root_z[0] - self._get_min_body_z_from_qpos(qpos[0])[0]
        
        skip = [minidxes[i] for i in [-3]] if len(minidxes) >= 3 else []

        for t in range(1,T):
            if t in minidxes and t not in skip:
                new_z = root_z[t] - self._get_min_body_z_from_qpos(qpos[t])[0]
                new_qpos_t = qpos[t].copy()
                new_qpos_t[2] = new_z
                if self._get_min_body_z_from_qpos(new_qpos_t)[0] >= 0:
                    rebuild_root_z[t] = new_z
                    continue
            rebuild_root_z[t] = rebuild_root_z[t-1] + root_vel[t-1]

        for i in maxidxes:
            for j in minidxes:
                if j in skip:
                    break
                if j > i and rebuild_root_z[i] > rebuild_root_z[j]:
                    rebuild_root_z[i+1:j] = self._fit_parabola_and_sample(rebuild_root_z[i],rebuild_root_z[j], j - i - 1)[-1]
                    break
        
        qpos[:,2] = rebuild_root_z

        for t in range(T):
            minz = self._get_min_body_z_from_qpos(qpos[t])[0]
            if minz < 0:
                rebuild_root_z[t] = root_z[t] - self._get_min_body_z_from_qpos(qpos[t])[0]

        qpos[:,2] = rebuild_root_z
        return qpos, minidxes, maxidxes

    def _fit_parabola_and_sample(self, y0, y1, N, g=9.81):
        T = np.sqrt(2.0 * (y0 - y1) / g)
        t_all = np.linspace(0.0, T, N + 2)
        y_all = y0 - 0.5 * g * t_all**2
        t_mid = t_all[1:-1]
        y_mid = y_all[1:-1]
        return t_all, y_all, t_mid, y_mid

    def piecewise_savgol(self, qpos, window=7, poly=3, deriv_thresh=4.0):
        """
        Applies Savitzky-Golay filter piecewise to smooth the motion data.
        """
        qpos_f = qpos.copy()
        T, D = qpos.shape

        for d in range(D):
            x = qpos[:, d]
            dx = np.diff(x, prepend=x[0])
            ddx = np.abs(np.diff(dx, prepend=dx[0]))

            turning_points = np.where(ddx > deriv_thresh * np.std(ddx))[0]
            segments = np.split(np.arange(T), turning_points)

            for seg in segments:
                if len(seg) < window:
                    continue
                qpos_f[seg, d] = savgol_filter(
                    x[seg], window_length=window, polyorder=poly
                )

        return qpos_f
    
    def _parse_xml(self):
        tree = ET.parse(self._file_path)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        assert xml_world_body is not None, "worldbody not found"
        
        xml_body_root = xml_world_body.find("body")
        assert xml_body_root is not None, "body not found"
        
        compiler_data = xml_doc_root.find("compiler")
        self._rot_unit = compiler_data.attrib.get("angle", "degree")
        assert self._rot_unit in ["degree", "radian"], f"Invalid rotation unit: {self._rot_unit}"
        
        def _add_xml_body(xml_node, parent_index, body_index):
            body_name = xml_node.attrib.get("name")
            pos_data = xml_node.attrib.get("pos", "0 0 0")
            pos = np.fromstring(pos_data, dtype=float, sep=" ")
            
            rot_data = xml_node.attrib.get("quat", "1 0 0 0")
            rot = np.fromstring(rot_data, dtype=float, sep=" ")
            rot_w = rot[..., 0].copy()
            rot[..., 0:3] = rot[..., 1:]
            rot[..., 3] = rot_w
            
            # 初始化joint相关变量
            dof_names = []

            if body_index == 0:
                curr_joint = Joint(name=body_name, dof_dim=0, axis=None) # root
                # dof_names.append("floating_base_joint")  # 根据XML中的定义
            else:
                curr_joints = xml_node.findall("joint")
                num_joints = len(curr_joints)
                if num_joints == 0:
                    curr_joint = Joint(name=body_name, dof_dim=0, axis=None)
                elif num_joints == 1:

                    dof_name = curr_joints[0].attrib.get("name")
                    dof_names.append(dof_name)

                    _axis = np.fromstring(curr_joints[0].attrib.get("axis"), dtype=float, sep=" ")
                    axis = torch.from_numpy(_axis).to(self._device)
                    curr_joint = Joint(name=body_name, dof_dim=1, axis=axis)
                    _dof_limits = np.fromstring(curr_joints[0].attrib.get("range"), dtype=float, sep=" ")
                    self._dof_lower_limits.append(_dof_limits[0])
                    self._dof_upper_limits.append(_dof_limits[1])
                elif num_joints == 3:
                    axis = None
                    curr_joint = Joint(name=body_name, dof_dim=3, axis=axis)
                    for joint in curr_joints:
                        dof_name = joint.attrib.get("name")
                        dof_names.append(dof_name)
                        _dof_limits = np.fromstring(joint.attrib.get("range"), dtype=float, sep=" ")
                        self._dof_lower_limits.append(_dof_limits[0])
                        self._dof_upper_limits.append(_dof_limits[1])
                else:
                    raise ValueError(f"Invalid number of joints: {num_joints} of body: {body_name}")
            
            self._body_names.append(body_name)

            for dof_name in dof_names:
                self._dof_names.append(dof_name)

            self._parent_indices.append(parent_index)
            self._local_rotation.append(rot)
            self._local_translation.append(pos)
            self._joints.append(curr_joint)
            self._dof_size.append(curr_joint.dof_dim)
            
            curr_index = body_index
            body_index += 1
            for child in xml_node.findall("body"):
                body_index = _add_xml_body(child, curr_index, body_index)
                
            return body_index
        
        _add_xml_body(xml_body_root, -1, 0)
        
    def _set_dof_indices(self):
        curr_dof_idx = 0
        for joint in self._joints:
            if joint.dof_dim > 0:
                joint.set_dof_idx(curr_dof_idx)
                curr_dof_idx += joint.dof_dim
                
    def dof_to_rot(self, dof):
        rot_shape = list(dof.shape[:-1]) + [self.num_joint-1, 4]
        joint_rot = torch.zeros(rot_shape, dtype=dof.dtype, device=dof.device)
        
        for j in range(1, self.num_joint):
            joint = self._joints[j]
            if joint.dof_idx == -1:
                joint_rot[..., j-1, -1] = 1.0
            else:
                joint_rot[..., j-1, :] = joint.dof_to_rot(dof[..., joint.dof_idx:joint.dof_idx+joint.dof_dim])
        return joint_rot
    
    def rot_to_dof(self, rot):
        dof_shape = list(rot.shape[:-2]) + [self.num_dof]
        dof = torch.zeros(dof_shape, dtype=rot.dtype, device=rot.device)
        
        for j in range(1, self.num_joint):
            joint = self._joints[j]
            if joint.dof_dim == 0:
                continue
            joint_rot = rot[..., j-1, :]
            dof[..., joint.dof_idx:joint.dof_idx+joint.dof_dim] = joint.rot_to_dof(joint_rot)
        
        dof = torch.clamp(dof, self._dof_lower_limits, self._dof_upper_limits)
            
        return dof
    
    def convert_local_rot_to_global(self, local_rot):
        # Input local_rot shape: [..., num_joint, 4] first row is root rotation
        # local rotation shape: [num_joint-1, 4]
        global_rot = torch.zeros_like(local_rot)
        global_rot[..., 0, :] = local_rot[..., 0, :]
        
        for j in range(1, self.num_joint):
            parent_idx = self._parent_indices[j]
            parent_rot = global_rot[..., parent_idx, :]
            local_rot_j = local_rot[..., j, :]
            global_rot[..., j, :] = torch_utils.quat_mul(parent_rot, local_rot_j)
        
        return global_rot
    
    def forward_kinematics(self, root_pos, root_rot, dof_pos, fitted_shape=None):
        joint_rot = self.dof_to_rot(dof_pos)
        
        body_pos = [None] * self.num_joint
        body_rot = [None] * self.num_joint
        
        body_pos[0] = root_pos
        body_rot[0] = root_rot
        
        for j in range(1, self.num_joint):
            j_rot = joint_rot[..., j-1, :]
            local_trans = self._local_translation[j] if fitted_shape is None else self._local_translation[j] * fitted_shape[j]
            local_rot = self._local_rotation[j]
            parent_idx = self._parent_indices[j]
            
            parent_pos = body_pos[parent_idx]
            parent_rot = body_rot[parent_idx]
            
            local_trans_broadcast = torch.broadcast_to(local_trans, parent_pos.shape)
            local_rot_broadcast = torch.broadcast_to(local_rot, parent_rot.shape)
            
            world_trans = torch_utils.quat_rotate(parent_rot, local_trans_broadcast)
            
            curr_pos = parent_pos + world_trans
            curr_rot = torch_utils.quat_mul(local_rot_broadcast, j_rot)
            curr_rot = torch_utils.quat_mul(parent_rot, curr_rot)
            
            body_pos[j] = curr_pos
            body_rot[j] = curr_rot
        
        body_pos = torch.stack(body_pos, dim=-2)
        body_rot = torch.stack(body_rot, dim=-2)
        
        return body_pos, body_rot
    
    def get_body_idx(self, body_name):
        return self._body_names.index(body_name)
    
    def get_joint_idx(self, joint_name):
        return self._dof_names.index(joint_name)

    @property
    def body_names(self):
        return self._body_names
    

    @property
    def dof_names(self):
        return self._dof_names

    @property
    def num_dof(self):
        return self._num_dof
    
    @property
    def num_joint(self):
        return len(self._joints)
    
    @property
    def joint_dof_idx(self):
        dof_indices = []
        for joint in self._joints:
            dof_indices.append(joint.dof_idx)
        return dof_indices
    
    @property
    def parent_indices(self):
        return self._parent_indices
    
    def get_parent_idx(self, idx):
        return self._parent_indices[idx]
    
    def get_dof_limits(self):
        return self._dof_lower_limits, self._dof_upper_limits
