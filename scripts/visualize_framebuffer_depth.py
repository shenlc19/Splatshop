#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        header = file.readline().decode("ascii").strip()
        if header not in {"Pf", "PF"}:
            raise ValueError(f"{path} is not a PFM file")

        line = file.readline().decode("ascii").strip()
        while line.startswith("#"):
            line = file.readline().decode("ascii").strip()
        width, height = [int(v) for v in line.split()]

        scale = float(file.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        channels = 1 if header == "Pf" else 3
        data = np.fromfile(file, endian + "f4", width * height * channels)

    if channels == 1:
        return data.reshape((height, width))
    return data.reshape((height, width, channels))


def read_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is needed to read PNG colors") from exc

    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def resolve_paths(prefix_or_depth: Path, camera_path: Optional[Path], color_path: Optional[Path], mask_path: Optional[Path]):
    if prefix_or_depth.name.endswith("_depth.pfm"):
        prefix = prefix_or_depth.with_name(prefix_or_depth.name[:-len("_depth.pfm")])
        depth_path = prefix_or_depth
    else:
        prefix = prefix_or_depth
        depth_path = prefix.with_name(prefix.name + "_depth.pfm")

    camera_path = camera_path or prefix.with_name(prefix.name + "_camera.json")
    color_path = color_path or prefix.with_name(prefix.name + "_color.png")
    mask_path = mask_path or prefix.with_name(prefix.name + "_transparent_mask.pfm")
    return prefix, depth_path, camera_path, color_path, mask_path


def depth_to_point_map(depth: np.ndarray, camera: dict, *, camera_space: bool) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    projection = np.asarray(camera["projectionMatrix"], dtype=np.float64)
    proj00 = float(projection[0, 0])
    proj11 = float(projection[1, 1])

    x, y = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    ndc_x = (2.0 * (x + 0.5) / width) - 1.0
    ndc_y = 1.0 - (2.0 * (y + 0.5) / height)

    z = depth.astype(np.float64)
    valid = np.isfinite(z) & (z > 0.0)

    point_map = np.empty((height, width, 3), dtype=np.float64)
    point_map[..., 0] = ndc_x * z / proj00
    point_map[..., 1] = ndc_y * z / proj11
    point_map[..., 2] = -z
    point_map[~valid] = np.nan

    if not camera_space:
        camera_world = np.asarray(camera["cameraWorldMatrix"], dtype=np.float64)
        valid_points = point_map[valid]
        homogeneous = np.concatenate([valid_points, np.ones((valid_points.shape[0], 1), dtype=np.float64)], axis=1)
        point_map[valid] = (camera_world @ homogeneous.T).T[:, :3]

    return point_map, valid


def point_map_to_points(point_map: np.ndarray, valid: np.ndarray, *, stride: int) -> tuple[np.ndarray, np.ndarray]:
    sampled_points = point_map[::stride, ::stride]
    sampled_valid = valid[::stride, ::stride]
    return sampled_points[sampled_valid], sampled_valid


def write_pfm(path: Path, image: np.ndarray):
    if image.ndim not in {2, 3} or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError("PFM output must be HxW or HxWx3")

    path.parent.mkdir(parents=True, exist_ok=True)
    header = "PF" if image.ndim == 3 else "Pf"
    image = np.asarray(image, dtype="<f4")
    height, width = image.shape[:2]

    with path.open("wb") as file:
        file.write(f"{header}\n{width} {height}\n-1.0\n".encode("ascii"))
        image.tofile(file)


def write_point_map(path: Path, point_map: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".pfm":
        write_pfm(path, point_map)
    else:
        with path.open("wb") as file:
            np.save(file, point_map.astype(np.float32))


def write_ply(path: Path, points: np.ndarray, colors: Optional[np.ndarray]):
    path.parent.mkdir(parents=True, exist_ok=True)
    has_color = colors is not None
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        header.extend(["property uchar red", "property uchar green", "property uchar blue"])
    header.append("end_header")

    with path.open("wb") as file:
        file.write(("\n".join(header) + "\n").encode("ascii"))
        if has_color:
            data = np.empty(
                len(points),
                dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
            )
            data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
            data["red"], data["green"], data["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
            data.tofile(file)
        else:
            points.astype("<f4").tofile(file)


def show_points(points: np.ndarray, colors: Optional[np.ndarray], point_size: float):
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D is not installed; wrote the PLY but skipped interactive visualization.")
        return

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Splatshop framebuffer depth")
    vis.add_geometry(cloud)
    options = vis.get_render_option()
    options.point_size = point_size
    options.background_color = np.asarray([0.02, 0.02, 0.02])
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Convert a Splatshop framebuffer depth dump to a 3D point cloud.")
    parser.add_argument("dump", type=Path, help="Dump prefix like debug/framebuffer_0006, or the *_depth.pfm file")
    parser.add_argument("--camera", type=Path, help="Path to *_camera.json")
    parser.add_argument("--color", type=Path, help="Path to *_color.png")
    parser.add_argument("--mask", type=Path, help="Path to *_transparent_mask.pfm")
    parser.add_argument("--output", "-o", type=Path, help="Output PLY path")
    parser.add_argument("--point-map-output", type=Path, help="Output HxWx3 point map path (.npy by default, or .pfm)")
    parser.add_argument("--no-point-map", action="store_true", help="Only write the PLY, not the image-aligned point map")
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth pixel")
    parser.add_argument("--max-transparent", type=float, default=0.98, help="Reject pixels above this transparent-mask value")
    parser.add_argument("--camera-space", action="store_true", help="Leave points in camera coordinates instead of world coordinates")
    parser.add_argument("--no-view", action="store_true", help="Write PLY without opening an interactive viewer")
    parser.add_argument("--point-size", type=float, default=2.0)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be at least 1")

    prefix, depth_path, camera_path, color_path, mask_path = resolve_paths(args.dump, args.camera, args.color, args.mask)
    output_path = args.output or prefix.with_name(prefix.name + "_depth_points.ply")
    point_map_output_path = args.point_map_output or prefix.with_name(prefix.name + "_depth_point_map.npy")

    with camera_path.open("r", encoding="utf-8") as file:
        camera = json.load(file)

    depth = read_pfm(depth_path)
    point_map, valid = depth_to_point_map(depth, camera, camera_space=args.camera_space)

    if mask_path.exists():
        mask = read_pfm(mask_path)
        if mask.shape != depth.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match depth shape {depth.shape}")
        keep_pixels = mask <= args.max_transparent
        valid = valid & keep_pixels
        point_map[~valid] = np.nan

    if not args.no_point_map:
        write_point_map(point_map_output_path, point_map)
        print(f"Wrote point map {point_map.shape[1]}x{point_map.shape[0]}x3 to {point_map_output_path}")

    points, sampled_valid = point_map_to_points(point_map, valid, stride=args.stride)

    color_values = None
    if color_path.exists():
        color = read_image(color_path)
        if color.shape[:2] != depth.shape:
            raise ValueError(f"Color shape {color.shape[:2]} does not match depth shape {depth.shape}")
        if args.stride > 1:
            color = color[::args.stride, ::args.stride]
        color_values = color[sampled_valid]

    write_ply(output_path, points, color_values)
    print(f"Wrote {len(points):,} points to {output_path}")

    if not args.no_view:
        show_points(points, color_values, args.point_size)


if __name__ == "__main__":
    main()
