import argparse
import pathlib
import os
import mujoco as mj
import numpy as np
from tqdm import tqdm
import torch
import pickle

from general_motion_retargeting.utils.lafan1 import load_bvh_file
from general_motion_retargeting.kinematics_model import KinematicsModel
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from rich import print


def check_nan_in_frames(frames):
    """
    Check if any frame contains NaN values in position or orientation.
    Returns: (has_nan, frame_idx, bone_name, field_type)
        - has_nan: bool, whether NaN was found
        - frame_idx: int, index of the frame with NaN (-1 if not found)
        - bone_name: str, name of the bone with NaN ("" if not found)
        - field_type: str, "position", "orientation", or "" if not found
    """
    for frame_idx, frame in enumerate(frames):
        for bone_name, (pos, quat) in frame.items():
            pos_arr = np.asarray(pos)
            quat_arr = np.asarray(quat)
            if np.any(np.isnan(pos_arr)):
                return True, frame_idx, bone_name, "position"
            if np.any(np.isnan(quat_arr)):
                return True, frame_idx, bone_name, "orientation"
            # Also check for zero-norm quaternion which causes scipy error
            quat_norm = np.linalg.norm(quat_arr)
            if quat_norm < 1e-8:
                return True, frame_idx, bone_name, "zero_norm_quat"
    return False, -1, "", ""


if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_folder",
        help="Folder containing BVH motion files to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--tgt_folder",
        help="Folder to save the retargeted motion files.",
        default="../../qingtong/q1_motion_0821"
    )
    
    parser.add_argument(
        "--robot",
        default="q1",
    )
    
    parser.add_argument(
        "--format",
        choices=["lafan1", "nokov", "soma", "qingtong"],
        default="lafan1",
    )
    parser.add_argument(
        "--override",
        default=False,
        action="store_true",
    )
    
    parser.add_argument(
        "--target_fps",
        default=30,
        type=int,
    )

    args = parser.parse_args()
    
    src_folder = args.src_folder
    tgt_folder = args.tgt_folder

    # Statistics for NaN files
    nan_files = []
    total_files = 0
    skipped_nan = 0

    # walk over all files in src_folder
    for dirpath, _, filenames in os.walk(src_folder):
        for filename in tqdm(sorted(filenames), desc="Retargeting files"):
            if not filename.endswith(".bvh"):
                continue
                
            # get the bvh file path
            bvh_file_path = os.path.join(dirpath, filename)
            
            # get the target file path
            tgt_file_path = bvh_file_path.replace(src_folder, tgt_folder).replace(".bvh", ".pkl")

            if os.path.exists(tgt_file_path) and not args.override:
                print(f"Skipping {bvh_file_path} because {tgt_file_path} exists")
                continue
            
            total_files += 1

            # Load LAFAN1 trajectory
            try:
                lafan1_data_frames, actual_human_height, bvh_fps = load_bvh_file(bvh_file_path, format=args.format)
                src_fps = bvh_fps
            except Exception as e:
                print(f"Error loading {bvh_file_path}: {e}")
                continue

            # Check for NaN values in loaded BVH data
            has_nan, frame_idx, bone_name, field_type = check_nan_in_frames(lafan1_data_frames)
            if has_nan:
                print(f"[yellow]Skipping {bvh_file_path}: NaN detected in {field_type} of bone '{bone_name}' at frame {frame_idx}[/yellow]")
                nan_files.append((bvh_file_path, frame_idx, bone_name, field_type))
                skipped_nan += 1
                continue

            # Initialize the retargeting system
            retarget = GMR(
                src_human=f"bvh_{args.format}", 
                tgt_robot=args.robot,
                actual_human_height=actual_human_height,
            )
            model = mj.MjModel.from_xml_path(retarget.xml_file)
            data = mj.MjData(model)

            # retarget to get all qpos
            qpos_list = []
            retarget_failed = False
            for curr_frame in range(len(lafan1_data_frames)):
                smplx_data = lafan1_data_frames[curr_frame]
                
                # Retarget till convergence
                try:
                    qpos = retarget.retarget(smplx_data)
                except Exception as e:
                    print(f"[red]Retarget failed for {bvh_file_path} at frame {curr_frame}: {e}[/red]")
                    nan_files.append((bvh_file_path, curr_frame, "", "retarget_error"))
                    skipped_nan += 1
                    retarget_failed = True
                    break
                
                qpos_list.append(qpos.copy())
            
            if retarget_failed:
                continue
            
            qpos_list = np.array(qpos_list)

            # Initialize the forward kinematics
            device = "cuda:0"
            kinematics_model = KinematicsModel(retarget.xml_file, device=device)
            
            root_pos = qpos_list[:, :3]
            root_rot = qpos_list[:, 3:7]
            root_rot[:, [0, 1, 2, 3]] = root_rot[:, [1, 2, 3, 0]]
            dof_pos = qpos_list[:, 7:]
            num_frames = root_pos.shape[0]
            
            # obtain local body pos
            identity_root_pos = torch.zeros((num_frames, 3), device=device)
            identity_root_rot = torch.zeros((num_frames, 4), device=device)
            identity_root_rot[:, -1] = 1.0
            local_body_pos, _ = kinematics_model.forward_kinematics(
                identity_root_pos, 
                identity_root_rot, 
                torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
            )
            body_names = kinematics_model.body_names

            HEIGHT_ADJUST = False
            PERFRAME_ADJUST = False
            if HEIGHT_ADJUST:
                body_pos, _ = kinematics_model.forward_kinematics(
                    torch.from_numpy(root_pos).to(device=device, dtype=torch.float),
                    torch.from_numpy(root_rot).to(device=device, dtype=torch.float),
                    torch.from_numpy(dof_pos).to(device=device, dtype=torch.float)
                )
                ground_offset = 0.00
                if not PERFRAME_ADJUST:
                    lowest_height = torch.min(body_pos[..., 2]).item()
                    root_pos[:, 2] = root_pos[:, 2] - lowest_height + ground_offset
                else:
                    for i in range(root_pos.shape[0]):
                        lowest_body_part = torch.min(body_pos[i, :, 2])
                        root_pos[i, 2] = root_pos[i, 2] - lowest_body_part + ground_offset

            motion_data = {
                "root_pos": root_pos,
                "root_rot": root_rot,
                "dof_pos": dof_pos,
                "local_body_pos": local_body_pos.detach().cpu().numpy(),
                "fps": src_fps,
                "link_body_list": body_names,
            }
            
            os.makedirs(os.path.dirname(tgt_file_path), exist_ok=True)
            with open(tgt_file_path, "wb") as f:
                pickle.dump(motion_data, f)

    # Print NaN statistics
    print("\n" + "=" * 60)
    print(f"[bold]Processing Summary[/bold]")
    print(f"Total BVH files processed: {total_files}")
    print(f"Files skipped due to NaN: {skipped_nan}")
    if nan_files:
        print(f"\n[bold yellow]Files with NaN values ({len(nan_files)}):[/bold yellow]")
        for fpath, frame_idx, bone_name, field_type in nan_files:
            rel_path = os.path.relpath(fpath, src_folder)
            if bone_name:
                print(f"  - {rel_path}  (frame={frame_idx}, bone={bone_name}, field={field_type})")
            else:
                print(f"  - {rel_path}  (frame={frame_idx}, {field_type})")
    print("=" * 60)
    print(f"Done. saved to {tgt_folder}")