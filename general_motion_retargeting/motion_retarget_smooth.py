
import mink
import mujoco as mj
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print
from scipy.spatial.transform import Slerp


def quat_normalize_wxyz(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n

def quat_slerp_wxyz(q0, q1, alpha):
    """
    q0, q1: [qw,qx,qy,qz] (wxyz)
    alpha: 0..1, 越小越稳、越大越跟手
    """
    q0 = quat_normalize_wxyz(q0)
    q1 = quat_normalize_wxyz(q1)

    # 1) 符号连续（取最短弧）
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    # 2) 非常接近时用 lerp（数值更稳）
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return quat_normalize_wxyz(q)

    # 3) 标准 slerp
    theta0 = np.arccos(dot)          # 0..pi
    sin0 = np.sin(theta0)
    theta = theta0 * alpha
    s0 = np.sin(theta0 - theta) / sin0
    s1 = np.sin(theta) / sin0
    q = s0 * q0 + s1 * q1
    return quat_normalize_wxyz(q)


class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str="daqp", # change from "quadprog" to "daqp".
        damping: float=5e-1, # change from 1e-1 to 1e-2.
        verbose: bool=True,
        use_velocity_limit: bool=False,
    ) -> None:

        # load the robot model
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        
        # Print DoF names in order
        print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")
            
            
        print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")
        
        print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # Load the IK config
        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])
        
        # compute the scale ratio based on given human height and the assumption in the IK config
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
            
        # adjust the human scale table
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio
    

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        self.max_iter = 10

        self.solver = solver
        self.damping = damping

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        self.task_errors1 = {}
        self.task_errors2 = {}

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            VELOCITY_LIMITS = {k: 3*np.pi for k in self.robot_motor_names.keys()}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS)) 
            
        self.setup_retarget_configuration()

        # --- unwrap state (avoid +pi <-> -pi jumps) ---
        self._prev_qpos = None
        self._unwrap_idx = []

        # Only unwrap hinge joints (1D angle in qpos). Don't unwrap free/ball joints.
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE:
                adr = int(self.model.jnt_qposadr[j])  # qpos index for this joint angle
                self._unwrap_idx.append(adr)
        
        self.ground_offset = 0.0

        # DoF 0: left_hip_pitch_joint
        # DoF 1: left_hip_roll_joint
        # DoF 2: left_hip_yaw_joint
        # DoF 3: left_knee_joint
        # DoF 4: left_ankle_pitch_joint
        # DoF 5: left_ankle_roll_joint
        # DoF 6: right_hip_pitch_joint
        # DoF 7: right_hip_roll_joint
        # DoF 8: right_hip_yaw_joint
        # DoF 9: right_knee_joint
        # DoF 10: right_ankle_pitch_joint
        # DoF 11: right_ankle_roll_joint
        # DoF 12: waist_roll_joint
        # DoF 13: waist_yaw_joint
        # DoF 14: left_shoulder_pitch_joint
        # DoF 15: left_shoulder_roll_joint
        # DoF 16: left_shoulder_yaw_joint
        # DoF 17: left_elbow_joint
        # DoF 18: left_wrist_yaw_joint
        # DoF 19: right_shoulder_pitch_joint
        # DoF 20: right_shoulder_roll_joint
        # DoF 21: right_shoulder_yaw_joint
        # DoF 22: right_elbow_joint
        # DoF 23: right_wrist_yaw_joint

        ### 打印前后两帧的输出检查是否有跳变，如果有的话是哪个关节在跳变
        self.frame_idx = 0
        self.jump_maxabs_thresh = 0.7
        self.jump_norm_thresh = 2.0
        self.debug = True
        print("start smooth")

        ### 平滑IK的结果
        self.root_alpha_pos_mask = True # root position，True为开启对root position的平滑
        self.root_alpha_rot_mask = True # root rotation，True为开启对root rotation的平滑
        self.joint_smooth_mask = np.ones(self.model.nq - 7, dtype=bool) # joint，分别对应是否开启22个关节的平滑
        # self.joint_smooth_mask[:] = False

        self.prev_qpos = None
        self.root_alpha_pos = 0.25 # root position平滑系数，越小越稳，越大根的越快

        self.prev_root_quat = None
        self.root_alpha_rot = 0.25   # root rotation平滑系数，越小越稳，越大跟随越快

        self.joint_alpha = np.ones(self.model.nq - 7, dtype=float) # joint 平滑系数，越小越稳，越大跟随越快
        self.joint_alpha[:] = 0.25

        ### 平滑Target
        self.enable_target_smoothing = False # target，True为开启对human data的平滑
        self.target_alpha_pos = 0.2 # human data position 平滑系数
        self.target_alpha_rot = 0.2 # human data rotation 平滑系数
        self._tgt_pos_f = {}
        self._tgt_quat_f = {}


    def _smooth_target(self, body_name: str, pos_raw: np.ndarray, quat_raw_wxyz: np.ndarray):
        """
        Smooth target in input space:
          - position: first-order low-pass
          - rotation: quaternion slerp (with sign fix)
        quat format: wxyz (scalar_first=True)
        """
        pos_raw = np.asarray(pos_raw, dtype=np.float64).reshape(3)
        quat_raw = np.asarray(quat_raw_wxyz, dtype=np.float64).reshape(4)

        # safety: if NaN appears, just fall back to raw (avoid propagating NaNs)
        if not (np.isfinite(pos_raw).all() and np.isfinite(quat_raw).all()):
            return pos_raw, quat_raw

        # init cache on first use
        if body_name not in self._tgt_pos_f:
            self._tgt_pos_f[body_name] = pos_raw.copy()
            self._tgt_quat_f[body_name] = quat_raw.copy()
            return pos_raw, quat_raw

        # --- position low-pass ---
        pos_prev = self._tgt_pos_f[body_name]
        pos_f = (1.0 - self.target_alpha_pos) * pos_prev + self.target_alpha_pos * pos_raw

        # --- rotation slerp ---
        quat_prev = self._tgt_quat_f[body_name]

        # sign fix: q and -q represent same rotation, but slerp needs consistent hemisphere
        if float(np.dot(quat_prev, quat_raw)) < 0.0:
            quat_raw = -quat_raw

        # We keep scalar_first=True => wxyz
        r_prev = R.from_quat(quat_prev, scalar_first=True)
        r_raw = R.from_quat(quat_raw, scalar_first=True)

        # interpolate: prev -> raw by alpha_rot
        key_times = [0.0, 1.0]
        key_rots = R.from_quat([quat_prev, quat_raw], scalar_first=True)
        slerp = Slerp(key_times, key_rots)
        r_f = slerp([self.target_alpha_rot])[0]
        quat_f = r_f.as_quat(scalar_first=True)

        # cache
        self._tgt_pos_f[body_name] = pos_f
        self._tgt_quat_f[body_name] = quat_f

        return pos_f, quat_f

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
    
        self.tasks1 = []
        self.tasks2 = []
        
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets1[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks1.append(task)
                self.task_errors1[task] = []
        
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks2.append(task)
                self.task_errors2[task] = []

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)
    
        self.tasks1 = []
        self.tasks2 = []
        
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets1[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks1.append(task)
                self.task_errors1[task] = []
        
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks2.append(task)
                self.task_errors2[task] = []

  
    def update_targets(self, human_data, offset_to_ground=False):
        
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)
        human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
        human_data = self.apply_ground_offset(human_data)
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)
        self.scaled_human_data = human_data



        if self.use_ik_match_table1:
            for body_name in self.human_body_to_task1.keys():
                task = self.human_body_to_task1[body_name]
                # pos, rot = human_data[body_name]
                # task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
                pos, rot = human_data[body_name]
                if self.enable_target_smoothing:
                    pos, rot = self._smooth_target(body_name, pos, rot)

                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

        
        if self.use_ik_match_table2:
            for body_name in self.human_body_to_task2.keys():
                task = self.human_body_to_task2[body_name]
                # pos, rot = human_data[body_name]
                # task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
                pos, rot = human_data[body_name]
                if self.enable_target_smoothing:
                    pos, rot = self._smooth_target(body_name, pos, rot)

                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

    def ensure_valid_configuration(self):
        """确保机器人的初始配置符合关节限位"""
        # 获取当前配置
        qpos = self.configuration.data.qpos.copy()
        
        # 遍历所有关节，检查并修正限位违规
        for i in range(self.model.njnt):
            # joint_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, i)
            joint_type = self.model.jnt_type[i]
            
            # 只处理可动关节
            if joint_type != mj.mjtJoint.mjJNT_FREE and joint_type != mj.mjtJoint.mjJNT_BALL:
                # 获取关节限位
                jnt_id = self.model.jnt_qposadr[i]
                if self.model.jnt_limited[i]:
                    lower = self.model.jnt_range[i, 0]
                    upper = self.model.jnt_range[i, 1]
                    
                    # 检查当前值是否在限位内
                    current_val = qpos[jnt_id]
                    if current_val < lower or current_val > upper:
                        # 选择更近的限位值
                        if current_val < lower:
                            safe_val = lower
                        else:
                            safe_val = upper
                        
                        qpos[jnt_id] = safe_val
                        # print(f"调整关节 {joint_name} 的值从 {current_val} 到 {safe_val} 以符合限位 [{lower}, {upper}]")
        
        # 更新配置
        self.configuration.data.qpos[:] = qpos
        self.configuration.update()
            
    def retarget(self, human_data, offset_to_ground=False):
        # 在开始重定向前，确保配置符合关节限位
        self.ensure_valid_configuration()

        ### trial 1
        num_iter1, err1 = -1, float("nan")
        num_iter2, err2 = -1, float("nan")


        # Update the task targets
        self.update_targets(human_data, offset_to_ground)

        if self.use_ik_match_table1:
            # Solve the IK problem
            curr_error = self.error1()
            dt = self.configuration.model.opt.timestep
            # dt = 1/30
            vel1 = mink.solve_ik(
                self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self.error1()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel1 = mink.solve_ik(
                    self.configuration, self.tasks1, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self.error1()
                num_iter += 1
            
            ### tiral 1
            num_iter1 = num_iter
            err1 = next_error

        if self.use_ik_match_table2:
            curr_error = self.error2()
            dt = self.configuration.model.opt.timestep
            vel2 = mink.solve_ik(
                self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
            )
            self.configuration.integrate_inplace(vel2, dt)
            next_error = self.error2()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                # Solve the IK problem with the second task
                dt = self.configuration.model.opt.timestep
                vel2 = mink.solve_ik(
                    self.configuration, self.tasks2, dt, self.solver, self.damping, self.ik_limits
                )
                self.configuration.integrate_inplace(vel2, dt)
                
                next_error = self.error2()
                num_iter += 1
            
            ### trial 1
            num_iter2 = num_iter
            err2 = next_error
                
            
        # return self.configuration.data.qpos.copy()
        qpos = self.configuration.data.qpos.copy()

        if self.prev_qpos is not None:
            if self.root_alpha_pos_mask:
                qpos[:3] = self.root_alpha_pos * qpos[:3] + (1 - self.root_alpha_pos) * self.prev_qpos[:3]

            if self.root_alpha_rot_mask:
                q_new = qpos[3:7]  # wxyz
                if self.prev_root_quat is None:
                    q_smoothed = quat_normalize_wxyz(q_new)
                else:
                    q_smoothed = quat_slerp_wxyz(self.prev_root_quat, q_new, self.root_alpha_rot)

                qpos[3:7] = q_smoothed
                self.prev_root_quat = q_smoothed.copy()


            for i in range(len(self.joint_smooth_mask)):
                if self.joint_smooth_mask[i]:
                    qpos[i+7] = self.joint_alpha[i] * qpos[i+7] + (1 - self.joint_alpha[i]) * self.prev_qpos[i+7]                    

            dq = qpos - self.prev_qpos
            maxabs = np.max(np.abs(dq))
            # print(maxabs)
            norm = np.linalg.norm(dq)

            if maxabs > self.jump_maxabs_thresh or norm > self.jump_norm_thresh:
                j = int(np.argmax(np.abs(dq)))
                print(f"[JUMP] frame={self.frame_idx} maxabs={maxabs:.3f} norm={norm:.3f} idx={j} dq={dq}")
                print(f"[IK] it1={num_iter1}/{self.max_iter} err1={err1:.4f} it2={num_iter2}/{self.max_iter} err2={err2:.4f}")

                jnt_id = int(self.model.jnt_qposadr.searchsorted(j, side="right") - 1)
                jnt_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, jnt_id)
                print(f"[JUMP_joint] qpos_idx={j} belongs to joint={jnt_name} (jnt_id={jnt_id})")
        
            
        self.prev_qpos = qpos
        self.frame_idx += 1

        return qpos



    def error1(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )
    
    def error2(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]
        
        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos
        
        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
            
        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global
    
    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            # apply rotation offset first
            updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
            offset_human_data[body_name][1] = updated_quat
            
            local_offset = pos_offsets[body_name]
            # compute the global position offset using the updated rotation
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)
            
            offset_human_data[body_name][0] = pos + global_pos_offset
           
        return offset_human_data
            
    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data
