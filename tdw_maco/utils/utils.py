import numpy as np
from PIL import Image
import os
from tdw.tdw_utils import TDWUtils
import argparse
import json
import pickle
import re
from tqdm import tqdm

def convert_np_for_print(np_array):

    if isinstance(np_array, list):
        results = []
        for np_array_item in np_array:
            assert isinstance(np_array_item, dict), f"to_convert_object must be a list of dict, {type(np_array_item)} found"
            result = dict()
            for k, v in np_array_item.items():
                if isinstance(v, np.ndarray):
                    list_v = [float(x) if isinstance(x, np.float32) else x for x in v]
                    result[k] = list_v
                elif isinstance(v, dict):
                    result[k] = dict()
                    for k2, v2 in v.items():
                        if isinstance(v2, np.ndarray):
                            list_v2 = [float(x) if isinstance(x, np.float32) else x for x in v2]
                            result[k][k2] = list_v2
                        else:
                            result[k][k2] = v2
                else:
                    result[k] = v
            results.append(result)
        return results
    elif isinstance(np_array, dict):
        result = dict()
        for k, v in np_array.items():
            if isinstance(v, np.ndarray):
                list_v = [float(x) if isinstance(x, np.float32) else x for x in v]
                result[k] = list_v
            elif isinstance(v, dict):
                result[k] = dict()
                for k2, v2 in v.items():
                    if isinstance(v2, np.ndarray):
                        list_v2 = [float(x) if isinstance(x, np.float32) else x for x in v2]
                        result[k][k2] = list_v2
                    else:
                        result[k][k2] = v2
            else:
                result[k] = v
        return result
    else:
        raise ValueError(f"to_convert_object must be a list or dict, {type(np_array)} found")

def get_cook_prompt(agent_id, recipe):
    agents_name = ["Alice", "Bob"]
    recipe_strs = []
    for recipe_single in recipe:
        recipe_str = ", ".join(recipe_single)
        if recipe_str.startswith("bread_slice"):
            recipe_str = "sandwich: " + recipe_str
        else:
            recipe_str = "burger: " + recipe_str
        recipe_strs.append(recipe_str)
    agent_name = agents_name[agent_id]
    return f"""Two agents Alice and Bob are cooperating together to cook at a kitchen counter. Each agent can only operate within the region of one counter edge. The goal is to make a burger and a sandwich. Food items must be stacked on the plate following this order:
{recipe_strs[0]}
{recipe_strs[1]}"""

def get_game_prompt(agent_id, recipe=None):
    return f"""Four agents Alice, Bob, Charlie and David around a square table are cooperating together to solve a puzzle game. Each agent can only operate within the region of one table edge. The goal is to put all the pieces of the puzzle on the table into the correct puzzle box."""

def get_ego_topdown(task, camera_matrix_metadata, rgb_img, depth_img):
    import open3d as o3d
    if task == 'cook':
        extrinsic = np.array([
            [0, 0, 1, -0.2],
            [-1, 0, 0, -0.5],
            [0, -1, 0, 4.2],
            [0, 0, 0, 1]
        ])
    else:
        extrinsic = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 6],
            [0, 0, 0, 1]
        ])

    rgb_img = np.array(Image.open(rgb_img))
    depth_img = np.array(Image.open(depth_img))
    length = rgb_img.shape[0]

    vfov = 54.43222 / 180.0 * np.pi
    tan_half_vfov = np.tan(vfov / 2.0)
    tan_half_hfov = tan_half_vfov
    fx = length / 2.0 / tan_half_hfov  # focal length in pixel space
    fy = length / 2.0 / tan_half_vfov
    intrinsic = np.array([[fx, 0, length / 2.0],
                        [0, fy, length / 2.0],
                        [0, 0, 1]])

    # depth_correct = np.flip(depth_img, axis=0)
    # Image.fromarray(depth_correct).save(os.path.join(path, f"{agent_id}/depth_right_{frame_start:05}.png"))
    depth = np.array(TDWUtils.get_depth_values(depth_img, width = length, height = length), dtype = np.float32) # np.flip is already implemented in the tdw_utils
    camera_matrix = camera_matrix_metadata
    if task == "game" or task == "game_3" or task == "game_2":
        pcd_array = TDWUtils.get_point_cloud(depth, camera_matrix, vfov = 82).transpose((1,2,0)).reshape(-1, 3)
    else:
        pcd_array = TDWUtils.get_point_cloud(depth, camera_matrix, vfov = 75).transpose((1,2,0)).reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_array)
    colors = rgb_img.reshape((-1, 3)) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])
    # o3d.io.write_point_cloud(os.path.join(path, f"{agent_id}/pcd_{frame_start:05}.ply"), pcd)
    # get the top-down view from the partial pcd
    pcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
    # Create an RGBD image from the point cloud
    rgbd_image = pcd.project_to_rgbd_image(length, length, intrinsic, depth_scale=5000.0,
                            depth_max=10.0, extrinsics=extrinsic)

    # Extract the RGB image
    rgb_image = rgbd_image.color
    # Convert the RGB image to a numpy array
    rgb_image_np = np.asarray(rgb_image)
    rgb_image_uint8 = (rgb_image_np * 255).astype(np.uint8)
    rgb_image_uint8 = rgb_image_uint8[::-1]
    return rgb_image_uint8

def get_overlay_ego_topdown(imgs):
    top_down = None
    for cur_top_down in imgs:
        if top_down is None:
            top_down = cur_top_down
        else:
            top_down = top_down * (cur_top_down == 0) + cur_top_down
    return top_down


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="results/train/game/single")
    parser.add_argument("--task", type=str, default="game")
    parser.add_argument("--agents", nargs='+', type=str, default=("0", "1", "2", "3",))
    args = parser.parse_args()
    for episode in os.listdir(args.path):
        # try:
            if not os.path.isdir(os.path.join(args.path, episode)):
                continue
            path = os.path.join(args.path, episode)
            with open(os.path.join(path, "camera_matrix_metadata.pickle"), "rb") as f:
                camera = pickle.load(f)
            with open(os.path.join(path, "metadata.json"), "r") as f:
                metadata = json.load(f)
            imgs_name = sorted(filter(lambda x: x.startswith("img"), os.listdir(os.path.join(path, "top_down"))))
            imgs_id = [int(re.search(r"\d+", img).group(0)) for img in imgs_name]
            assert imgs_id == sorted(imgs_id), "imgs_id is not sorted"

            last_frame = 0
            for step, stepdata in enumerate(tqdm(metadata, desc=f"Episode {episode}")):
                frame_start = stepdata["frame_start"]
                frame_end = stepdata["frame_end"]
                first_frame = last_frame
                while imgs_id[last_frame] < frame_end:
                    last_frame += 1
                for agent in args.agents:
                    imgs = []
                    for i in imgs_id[first_frame:last_frame + 1]:
                        imgs.append(np.array(Image.open(os.path.join(args.path, episode, agent, "reconstructed_%05d.png" % i))))
                    if len(imgs) == 1:
                        continue
                    top_down = get_overlay_ego_topdown(imgs)
                    Image.fromarray(top_down).save(
                        os.path.join(args.path, episode, agent, f"overlay_{step:02}_{frame_end:05}.png"))