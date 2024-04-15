from typing import List, Tuple, Dict, Union
import os
from tdw.add_ons.logger import Logger
from tdw.add_ons.image_capture import ImageCapture

from envs.asset_cached_controller import AssetCachedController
from components.object_manager import AdvancedObjectManager
from components.avatar import *
from replicant.state import ChallengeState
from replicant import Replicant
import magnebot
from magnebot import Magnebot

class MacoController(AssetCachedController):
    def __init__(self, port: int = 1071, check_version: bool = True, launch_build: bool = True,
                 screen_width: int = 1024,
                 screen_height: int = 1024,
                 asset_cache_dir="tdw_asset_cache",
                 image_passes=None,
                 avatars: List[Avatar] = [TeaserAvatar(), TopDownAvatar()],
                 target_framerate: int = 250,
                 enable_collision_detection: bool = False,
                 logger_dir=None):
        
        """
        :param port: The socket port used for communicating with the build.
        :param check_version: If True, the controller will check the version of the build and print the result.
        :param launch_build: If True, automatically launch the build. If one doesn't exist, download and extract the correct version. Set this to False to use your own build, or (if you are a backend developer) to use Unity Editor.
        :param screen_width: The width of the screen in pixels.
        :param screen_height: The height of the screen in pixels.
        :param asset_cache_dir: The directory to cache the build and assets in. If None, don't cache anything.
        :param avatars: A list of avatars. If None, don't add any avatars.
        :param image_frequency: How often each Replicant will capture an image. `ImageFrequency.once` means once per action, at the end of the action. `ImageFrequency.always` means every communicate() call. `ImageFrequency.never` means never.
        :param image_passes: A list of image passes, such as `["_img"]`. If None, defaults to `["_img", "_id", "_depth"]` (i.e. image, segmentation colors, depth maps).
        :param target_framerate: The target framerate. It's possible to set a higher target framerate, but doing so can lead to a loss of precision in agent movement.
        :param logger_dir: The directory to save the log file to. If None, don't save a log file.
        """
        super().__init__(cache_dir=asset_cache_dir, port=port, check_version=check_version, launch_build=launch_build)
        self.logger_dir = logger_dir
        if logger_dir is not None:
            print(logger_dir)
            self.logger = Logger(path=os.path.join(logger_dir, "action_log.log"))
            self.add_ons.append(self.logger)
        else:
            self.logger = None
        self.state = ChallengeState()
        self.object_manager = AdvancedObjectManager(transforms=True, rigidbodies=False, bounds=True)
        self.add_ons.append(self.object_manager)
        self.enable_collision_detection = enable_collision_detection
        self.avatars = avatars
        self.screen_width = screen_width
        self.screen_height = screen_height
        self._image_passes: List[str] = ["_img"] if image_passes is None else image_passes
        self.target_framerate = target_framerate
        self.agents: List[Union[Replicant, Magnebot]] = []
    
    def setup(self, *args, **kwargs) -> None:
        raise NotImplementedError()
    
    def check_goal(self, *args, **kwargs) -> Tuple[int, int, bool]:
        raise NotImplementedError()
    
    def get_objects(self, id) -> List[Dict]:
        raise NotImplementedError()
    
    def _init_agents(self):
        self.add_ons.extend(self.agents)
        self.communicate([])
        for agent in self.agents:
            if isinstance(agent, Magnebot):
                agent.reset_arm(arms=[magnebot.Arm.left, magnebot.Arm.right], immediate=True)

        moving = True
        while moving:
            self.communicate([])
            moving = False
            for agent in self.agents:
                if isinstance(agent, Magnebot):
                    if agent.action.status == magnebot.ActionStatus.ongoing:
                        moving = True
                        break
    
    def _render(self):
        if self.avatars is not None:
            self.add_ons.extend(self.avatars)
            if self.logger_dir is not None:
                avatar_ids = [avatar.name for avatar in self.avatars]
                for agent in self.agents:
                    if isinstance(agent, Magnebot):
                        avatar_ids.append(str(agent.robot_id))
                    elif isinstance(agent, Replicant):
                        avatar_ids.append(str(agent.replicant_id))
                    else:
                        raise NotImplementedError(f"No avatar for type: {type(agent)}")
                # self.add_ons.append(ImageCapture(self.logger_dir, avatar_ids, True, self._image_passes))
        # Initialize the window and rendering.
        self.communicate([{"$type": "set_screen_size",
                           "width": self.screen_width,
                           "height": self.screen_height},
                          {"$type": "set_img_pass_encoding",
                           "value": True},  # png
                          {"$type": "set_render_quality",
                           "render_quality": 5}])
        self.communicate([])
