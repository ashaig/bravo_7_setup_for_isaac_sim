# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import numpy as np
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, GroundPlane
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
from isaacsim.core.utils import distance_metrics
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats, quats_to_rot_matrices
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot_motion.motion_generation import ArticulationMotionPolicy, RmpFlow
from isaacsim.robot_motion.motion_generation.interface_config_loader import load_supported_motion_policy_config
from isaacsim.storage.native import get_assets_root_path

# gotta convert bravo.urdf to bravo.usd so we can make a SingleArticulation for it
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.asset.importer.urdf import _urdf
import omni.kit.commands
import omni.usd
from isaacsim.core.utils.prims import is_prim_path_valid

import math
from pxr import UsdGeom, Gf
from omni.isaac.dynamic_control import _dynamic_control

class BravoRmpFlowExampleScript:
    def __init__(self):
        self._rmpflow = None
        self._articulation_rmpflow = None

        self._articulation = None
        self._target = None

        self._script_generator = None

    def load_example_assets(self):
        """Load assets onto the stage and return them so they can be registered with the
        core.World.

        This function is called from ui_builder._setup_scene()

        The position in which things are loaded is also the position to which
        they will be returned on reset.
        """

        assets_root_path = get_assets_root_path()
        print("assets_root_path = ", assets_root_path)

        robot_prim_path = "/World/bravo_manipulation"
        path_to_robot_usd = "/home/ashes/isaacsim/custom_robot_files/reach_bravo_7/usd_files/bravo_working_gripper.usd"

        robot_usd = add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = SingleArticulation(robot_prim_path)

        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._target = SingleXFormPrim(
            "/World/target",
            scale=[0.04, 0.04, 0.04],
            position=np.array([0.4, -0.2, 0.25]),
            orientation=euler_angles_to_quats([0, np.pi/2, 0])
        )

        self._obstacles = [
            FixedCuboid(
                name="ob1",
                prim_path="/World/obstacle_1",
                scale=np.array([0.03, 1.0, 0.3]),
                position=np.array([0.25, 0.0, 0.15]),
                color=np.array([0.0, 0.0, 1.0]),
            ),
            FixedCuboid(
                name="ob2",
                prim_path="/World/obstacle_2",
                scale=np.array([0.5, 0.03, 0.3]),
                position=np.array([0.5, 0.0, 0.15]),
                color=np.array([0.0, 0.0, 1.0]),
            ),
        ]

        self._goal_block = DynamicCuboid(
            name="Cube",
            position=np.array([0.375, -0.2, 0.025]),
            prim_path="/World/pick_cube",
            size=0.05,
            scale=np.array([2.0, 1.0, 1.0]),
            color=np.array([1, 0, 0]),
        )

        self._ground_plane = GroundPlane("/World/Ground")

        # Return assets that were added to the stage so that they can be registered with the core.World
        return self._articulation, self._target, *self._obstacles, self._goal_block, self._ground_plane

    def setup(self):
        """
        This function is called after assets have been loaded from ui_builder._setup_scenario().
        """
        # Set a camera view that looks good
        set_camera_view(eye=[1.4, 0.0, 0.5], target=[0, 0, 0.3], camera_prim_path="/OmniverseKit_Persp")

        # Loading RMPflow can be done quickly for supported robots
        rmp_config = load_supported_motion_policy_config("Bravo", "RMPflow")

        # Initialize an RmpFlow object
        self._rmpflow = RmpFlow(**rmp_config)

        for obstacle in self._obstacles:
            self._rmpflow.add_obstacle(obstacle)

        # Use the ArticulationMotionPolicy wrapper object to connect rmpflow to the Bravo robot articulation.
        self._articulation_rmpflow = ArticulationMotionPolicy(self._articulation, self._rmpflow)

        # Create a script generator to execute my_script().
        self._script_generator = self.my_script()

    def reset(self):
        """
        This function is called when the reset button is pressed.
        In this example the core.World takes care of all necessary resetting
        by putting everything back in the position it was in when loaded.

        In more complicated scripts, e.g. scripts that modify or create USD properties
        or attributes at runtime, the user will need to implement necessary resetting
        behavior to ensure their script runs deterministically.
        """
        # Start the script over by recreating the generator.
        self._script_generator = self.my_script()

    """
    The following two functions demonstrate the mechanics of running code in a script-like way
    from a UI-based extension.  This takes advantage of Python's yield/generator framework.  

    The update() function is tied to a physics subscription, which means that it will be called
    one time on every physics step (usually 60 frames per second).  Each time it is called, it
    queries the script generator using next().  This makes the script generator execute until it hits
    a yield().  In this case, no value need be yielded.  This behavior can be nested into subroutines
    using the "yield from" keywords.
    """

    def update(self, step: float):
        try:
            result = next(self._script_generator)
        except StopIteration:
            return True

    def my_script(self):
        translation_target, orientation_target = self._target.get_world_pose()

        yield from self.close_gripper_bravo(self._articulation)

        # Notice that subroutines can still use return statements to exit.  goto_position() returns a boolean to indicate success.
        success = yield from self.goto_position(
            translation_target, orientation_target, self._articulation, self._rmpflow, timeout=200
        )

        if not success:
            print("Could not reach target position")
            return
        
        #print("successfully reached first target position/orientation")

        yield from self.open_gripper_bravo(self._articulation)

        #print("successfully opened gripper")

        # Visualize the new target.
        lower_translation_target = np.array([0.4, -0.2, 0.155])
        self._target.set_world_pose(lower_translation_target, orientation_target)

        success = yield from self.goto_position(
            lower_translation_target, orientation_target, self._articulation, self._rmpflow, timeout=250
        )

        #print("successfully moved open gripper to position around target cube")

        yield from self.close_gripper_bravo(self._articulation, close_position=np.array([4.5]), atol=0.05)

        #print("successfully closed gripper on target cube")

        high_left_translation_target = np.array([0.4, -0.2, 0.55])
        high_left_orientation_target = euler_angles_to_quats([0, 0.36, -0.46])
        self._target.set_world_pose(high_left_translation_target, high_left_orientation_target)

        success = yield from self.goto_position(
            high_left_translation_target, high_left_orientation_target, self._articulation, self._rmpflow, timeout=200
        )

        high_right_translation_target = np.array([0.4, 0.2, 0.55])
        high_right_orientation_target = euler_angles_to_quats([0, 0.36, 0.47])
        self._target.set_world_pose(high_right_translation_target, high_right_orientation_target)

        success = yield from self.goto_position(
            high_right_translation_target, high_right_orientation_target, self._articulation, self._rmpflow, timeout=200
        )

        #next_translation_target = np.array([0.4, 0.2, 0.25])
        low_right_translation_target = np.array([0.4, 0.2, 0.2])
        self._target.set_world_pose(low_right_translation_target, orientation_target)

        success = yield from self.goto_position(
            low_right_translation_target, 
            orientation_target, 
            self._articulation, 
            self._rmpflow, 
            timeout=200, 
            num_done_frames_target = 50
        )

        yield from self.open_gripper_bravo(self._articulation)

    ################################### Functions

    def goto_position(
        self,
        translation_target,
        orientation_target,
        articulation,
        rmpflow,
        translation_thresh=0.01,
        orientation_thresh=0.1,
        timeout=500,
        num_done_frames_target = 10):
        """
        Use RMPflow to move a robot Articulation to a desired task-space position.
        Exit upon timeout or when end effector comes within the provided threshholds of the target pose.
        """

        articulation_motion_policy = ArticulationMotionPolicy(articulation, rmpflow, 1 / 60)
        rmpflow.set_end_effector_target(translation_target, orientation_target)
        num_done_frames = 0

        #for i in range(timeout):
        i = 0
        while i <= timeout:
            i += 1

            ee_trans, ee_rot = rmpflow.get_end_effector_pose(
                articulation_motion_policy.get_active_joints_subset().get_joint_positions()
            )

            trans_dist = distance_metrics.weighted_translational_distance(ee_trans, translation_target)
            rotation_target = quats_to_rot_matrices(orientation_target)
            rot_dist = distance_metrics.rotational_distance_angle(ee_rot, rotation_target)

            #print("ee_trans = ", ee_trans)
            #print("trans_dist = ", trans_dist)
            #print("rot_dist = ", rot_dist)
            #print("ee_rot = ")
            #print(ee_rot)
            #print("rotation_target = ")
            #print(rotation_target)

            done = trans_dist < translation_thresh and rot_dist < orientation_thresh

            if done:
                num_done_frames += 1
                timeout += 1
                #return True
            elif done == False and num_done_frames > 0: 
                num_done_frames -= 1
            
            #print("num_done_frames = ", num_done_frames)
            #print("timeout = ", timeout)
            #print("i = ", i)
            
            if num_done_frames >= num_done_frames_target:
                return True

            rmpflow.update_world()
            action = articulation_motion_policy.get_next_articulation_action(1 / 60)
            articulation.apply_action(action)

            # If not done on this frame, yield() to pause execution of this function until
            # the next frame.
            yield ()

        return False

    def open_gripper_bravo(self, articulation):
        #print("opening gripper")
        # NOTE: the joint index for bravo_axis_a is 6 (b/c it's the 7th joint); 6.5 is a good value to 
        # set the joint target to in order to open the gripper for this little cube target
        open_gripper_action = ArticulationAction(np.array([6.5]), joint_indices=np.array([6]))
        articulation.apply_action(open_gripper_action)

        # Check in once a frame until the gripper has been successfully opened.
        #while not np.allclose(articulation.get_joint_positions()[7:], np.array([0.04, 0.04]), atol=0.001):
        while not np.allclose(articulation.get_joint_positions()[6], np.array([6.5]), atol=0.05):
            #print("articulation.get_joint_positions() = ")
            #print(articulation.get_joint_positions())
            #print("gripper joint position - gripper joint target = ")
            #print(articulation.get_joint_positions()[6] - np.array([6.5]))
            yield ()

        return True

    #def close_gripper_bravo(self, articulation, close_position=np.array([0, 0]), atol=0.001):
    def close_gripper_bravo(self, articulation, close_position=np.array([0]), atol=0.05):
        # To close around the cube, different values are passed in for close_position and atol
        close_gripper_action = ArticulationAction(np.array(close_position), joint_indices=np.array([6]))
        articulation.apply_action(close_gripper_action)

        # Check in once a frame until the gripper has been successfully closed.
        while not np.allclose(articulation.get_joint_positions()[6], np.array(close_position), atol=atol):
            #print("articulation.get_joint_positions()[6] = ", articulation.get_joint_positions()[6])
            #print("close_position = ", close_position)
            #print("gripper joint position - gripper joint target = ")
            #print(articulation.get_joint_positions()[6] - np.array([4.4]))
            yield ()

        return True
