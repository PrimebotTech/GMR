#!/usr/bin/env python3
"""
批量转换NPZ格式数据脚本
用法: python batch_convert_npz_folder.py --input_dir /path/to/npz/files --robot q1 --output_dir data/converted
"""

import os
import sys
import subprocess
import argparse
import glob
from pathlib import Path
import time

def find_npz_files(input_dir, recursive=True):
    """查找指定目录下的所有.npz文件"""
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        return []
    
    pattern = "**/*.npz" if recursive else "*.npz"
    npz_files = glob.glob(os.path.join(input_dir, pattern), recursive=recursive)
    
    print(f"在 {input_dir} 中找到 {len(npz_files)} 个.npz文件")
    return npz_files

def convert_single_file(npz_file_path, robot_type="q1", output_dir="data/converted", 
                       save_as_pkl=True, save_as_npz=False, save_as_txt=False, save_as_csv=False,
                       slice_motion=None, verbose=True):
    """转换单个NPZ文件"""
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建输出文件名（基于输入文件名）
    base_name = os.path.splitext(os.path.basename(npz_file_path))[0]
    
    # 构建命令参数
    args = [
        sys.executable,  # 使用当前Python解释器
        "scripts/smplx_to_robot.py",
        "--smplx_file", npz_file_path,
        "--robot", robot_type,
        "--save_path", output_dir,
    ]
    
    # 添加保存格式参数
    if save_as_pkl:
        args.append("--save_as_pkl")
        args.append("True")
    
    if save_as_npz:
        args.append("--save_as_npz")
        args.append("True")
        
    if save_as_txt:
        args.append("--save_as_txt")
        args.append("True")
        
    if save_as_csv:
        args.append("--save_as_csv")
        args.append("True")
    
    # 添加切片参数
    if slice_motion and isinstance(slice_motion, (list, tuple)) and len(slice_motion) == 2:
        args.append("--slice_motion_start_end")
        args.append(f"[{slice_motion[0]}, {slice_motion[1]}]")
        args.append("--save_slice")
        args.append("True")
    
    if verbose:
        print(f"转换: {npz_file_path} -> {output_dir}/{base_name}")
    
    # 执行转换
    start_time = time.time()
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=300)  # 5分钟超时
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            if verbose:
                print(f"✓ 成功: {base_name} (耗时: {elapsed_time:.2f}s)")
            return True, elapsed_time
        else:
            if verbose:
                print(f"✗ 失败: {base_name}")
                print(f"错误信息: {result.stderr}")
            return False, elapsed_time
    except subprocess.TimeoutExpired:
        if verbose:
            print(f"✗ 超时: {base_name} (超过5分钟)")
        return False, 300
    except Exception as e:
        if verbose:
            print(f"✗ 执行错误: {base_name} - {e}")
        return False, 0

def batch_convert_npz_folder(input_dir, robot_type="q1", output_dir=None, 
                           save_formats=["pkl"], recursive=True, max_workers=1,
                           slice_motion=None, dry_run=False, verbose=True):
    """批量转换文件夹中的NPZ文件"""
    
    # 设置默认输出目录
    if output_dir is None:
        output_dir = f"data/converted/{robot_type}_gmr"
    
    # 查找所有NPZ文件
    npz_files = find_npz_files(input_dir, recursive)
    
    if not npz_files:
        print("未找到任何.npz文件")
        return 0, 0, 0
    
    # 设置保存格式
    save_as_pkl = "pkl" in save_formats
    save_as_npz = "npz" in save_formats
    save_as_txt = "txt" in save_formats
    save_as_csv = "csv" in save_formats
    
    if dry_run:
        print(f"[DRY RUN] 将要转换的文件:")
        for i, npz_file in enumerate(npz_files, 1):
            print(f"  {i}. {npz_file}")
        print(f"总计: {len(npz_files)} 个文件")
        return len(npz_files), 0, 0
    
    print(f"开始批量转换 {len(npz_files)} 个文件...")
    print("-" * 70)
    
    total_files = len(npz_files)
    success_count = 0
    failed_files = []
    total_time = 0
    
    for i, npz_file in enumerate(npz_files, 1):
        print(f"[{i}/{total_files}] 处理中: {os.path.basename(npz_file)}")
        
        success, elapsed_time = convert_single_file(
            npz_file, 
            robot_type=robot_type,
            output_dir=output_dir,
            save_as_pkl=save_as_pkl,
            save_as_npz=save_as_npz,
            save_as_txt=save_as_txt,
            save_as_csv=save_as_csv,
            slice_motion=slice_motion,
            verbose=verbose
        )
        
        total_time += elapsed_time
        
        if success:
            success_count += 1
        else:
            failed_files.append(npz_file)
    
    # 输出统计信息
    print("-" * 70)
    print(f"转换完成!")
    print(f"总计: {total_files} 个文件")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {len(failed_files)} 个文件")
    print(f"总耗时: {total_time:.2f}s")
    print(f"平均耗时: {total_time/total_files:.2f}s/文件")
    
    if failed_files:
        print(f"\n失败的文件:")
        for i, failed_file in enumerate(failed_files, 1):
            print(f"  {i}. {failed_file}")
    
    print(f"\n输出目录: {output_dir}")
    
    return success_count, len(failed_files), total_time

def main():
    parser = argparse.ArgumentParser(description="批量转换NPZ格式数据")
    parser.add_argument("--input_dir", required=True, 
                       help="输入NPZ文件目录")
    parser.add_argument("--robot", default="q1", 
                       choices=["unitree_g1", "unitree_g1_with_hands", "q1", "unitree_h1_2",
                                "booster_t1", "booster_t1_29dof", "stanford_toddy", "fourier_n1", 
                                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", 
                                "berkeley_humanoid_lite", "booster_k1", "pnd_adam_lite", 
                                "openlong", "roboparty_atom01", "roboparty_atom01_long_base_link"],
                       help="机器人类型 (默认: q1)")
    parser.add_argument("--output_dir", 
                       help="输出目录 (默认: data/converted/{robot}_gmr)")
    parser.add_argument("--save_formats", nargs="+", default=["pkl"],
                       choices=["pkl", "npz", "txt", "csv"],
                       help="保存格式 (默认: pkl)")
    parser.add_argument("--recursive", action="store_true", default=True,
                       help="递归搜索子目录 (默认: True)")
    parser.add_argument("--no_recursive", dest="recursive", action="store_false",
                       help="不递归搜索子目录")
    parser.add_argument("--slice_motion", nargs=2, type=int, metavar=("START", "END"),
                       help="截取动作片段 [开始帧, 结束帧]")
    parser.add_argument("--dry_run", action="store_true",
                       help="只显示将要处理的文件，不实际转换")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="详细输出")
    parser.add_argument("--quiet", dest="verbose", action="store_false",
                       help="静默模式")
    
    args = parser.parse_args()
    
    # 验证输入目录
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return 1
    
    if not os.path.isdir(args.input_dir):
        print(f"错误: 输入路径不是目录: {args.input_dir}")
        return 1
    
    # 执行批量转换
    success, failed, total_time = batch_convert_npz_folder(
        input_dir=args.input_dir,
        robot_type=args.robot,
        output_dir=args.output_dir,
        save_formats=args.save_formats,
        recursive=args.recursive,
        slice_motion=args.slice_motion,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    # 返回状态码
    if args.dry_run:
        return 0
    elif failed == 0:
        return 0
    elif success > 0:
        return 0  # 部分成功也算成功
    else:
        return 1  # 全部失败

if __name__ == "__main__":
    sys.exit(main())