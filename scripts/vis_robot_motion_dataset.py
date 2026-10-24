"""Visualize one or more saved robot motions.

``--robot_motion_path`` accepts either a motion pickle file or a directory
containing motion pickle files.  When a directory is supplied, motions are
played in filename order and, if video recording is enabled, one video is
written per motion.
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from general_motion_retargeting import RobotMotionViewer, load_robot_motion


MOTION_SUFFIXES = {".pkl", ".pickle"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


def find_motion_files(motion_path):
    """Return motion files for a file or directory input.

    Only pickle files are selected from a directory because
    :func:`load_robot_motion` expects pickle data.
    """
    motion_path = Path(motion_path)

    if not motion_path.exists():
        raise FileNotFoundError(f"Motion path {motion_path} not found")
    if motion_path.is_file():
        return [motion_path]
    if not motion_path.is_dir():
        raise ValueError(f"Motion path {motion_path} is neither a file nor a directory")

    motion_files = sorted(
        path
        for path in motion_path.iterdir()
        if path.is_file() and path.suffix.lower() in MOTION_SUFFIXES
    )
    if not motion_files:
        suffixes = ", ".join(sorted(MOTION_SUFFIXES))
        raise FileNotFoundError(f"No motion files ({suffixes}) found in {motion_path}")
    return motion_files


def get_video_path(video_path, motion_path, batch_mode):
    """Resolve the output path, creating the parent directory if needed."""
    raw_video_path = str(video_path)
    video_path = Path(video_path)

    # A directory input always produces one video per source motion.  For a
    # single source, a path ending in '/' (or an existing directory) is also
    # treated as an output directory for convenience.
    output_is_dir = (
        batch_mode
        or video_path.is_dir()
        or raw_video_path.endswith(("/", "\\"))
        or video_path.suffix == ""
    )
    if output_is_dir:
        video_path.mkdir(parents=True, exist_ok=True)
        video_path = video_path / f"{motion_path.stem}.mp4"
    elif video_path.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(
            f"Video output {video_path} must be an mp4 file or an output directory"
        )

    video_path.parent.mkdir(parents=True, exist_ok=True)
    return str(video_path)


def visualize_motion(robot_type, motion_path, record_video=False, video_path=None):
    """Play a single motion once and close its viewer/video writer."""
    (
        _motion_data,
        motion_fps,
        motion_root_pos,
        motion_root_rot,
        motion_dof_pos,
        _motion_local_body_pos,
        _motion_link_body_list,
    ) = load_robot_motion(str(motion_path))

    env = None
    try:
        env = RobotMotionViewer(
            robot_type=robot_type,
            motion_fps=motion_fps,
            camera_follow=False,
            record_video=record_video,
            video_path=video_path,
        )

        for frame_idx in range(len(motion_root_pos)):
            env.step(
                motion_root_pos[frame_idx],
                motion_root_rot[frame_idx],
                motion_dof_pos[frame_idx],
                rate_limit=True,
            )
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize one motion file or all .pkl files in a directory."
    )
    parser.add_argument("--robot", type=str, default="q1")
    parser.add_argument(
        "--robot_motion_path",
        type=str,
        required=True,
        help="Motion .pkl/.pickle file or directory containing motion files.",
    )
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/example.mp4",
        help="Video file for a single motion, or output directory for a batch.",
    )
    args = parser.parse_args()

    motion_files = find_motion_files(args.robot_motion_path)
    batch_mode = len(motion_files) > 1 or Path(args.robot_motion_path).is_dir()

    print(f"Found {len(motion_files)} motion file(s).")
    for motion_path in tqdm(motion_files, desc="Visualizing motions"):
        output_video_path = None
        if args.record_video:
            output_video_path = get_video_path(
                args.video_path, motion_path, batch_mode=batch_mode
            )
            print(f"Visualizing {motion_path} -> {output_video_path}")
        else:
            print(f"Visualizing {motion_path}")

        visualize_motion(
            robot_type=args.robot,
            motion_path=motion_path,
            record_video=args.record_video,
            video_path=output_video_path,
        )


if __name__ == "__main__":
    main()
