#!/usr/bin/env python3
"""
Infinite Corridor using Panda3D

This script creates an infinite corridor effect with user-controlled forward/backward movement.

Features:
- Configurable parameters loaded from JSON
- Infinite corridor effect
- User-controlled movement
- [real-time] Data logging (timestamp, distance, speed)

The corridor consists of left, right, ceiling, and floor segments.
It uses the Panda3D CardMaker API to generate flat geometry for the corridor's four faces.
An infinite corridor/hallway effect is simulated by recycling the front segments to the back when the player moves forward. 


Configuration parameters are loaded from a JSON file "conf.json".

Author: Jake Gronemeyer
Date: 2025-02-23
Version: 0.2
"""

import json
import sys
import csv
import os
import time
import serial
import random
import numpy as np
from typing import Any, Dict
from dataclasses import dataclass

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import CardMaker, NodePath, Texture, WindowProperties, Fog
from direct.showbase import DirectObject
from stopwatch import Stopwatch
from direct.fsm.FSM import FSM

# Generate 250 random samples from a normal distribution
gaussian_data = np.random.normal(loc=25, scale=5, size=250)
rounded_gaussian_data = np.round(gaussian_data)

# Generate 250 random samples from a normal distribution for stay_gaussian_data
stay_gaussian_data = []
while len(stay_gaussian_data) < 250:
    sample = np.random.normal(loc=7, scale=2)
    if sample >= 1:  # Only accept values >= 1
        stay_gaussian_data.append(sample)
rounded_stay_data = np.round(stay_gaussian_data)

# Generate 250 random samples from a normal distribution for go_gaussian_data
go_gaussian_data = []
while len(go_gaussian_data) < 250:
    sample = np.random.normal(loc=30, scale=2)
    if sample >= 1:  # Only accept values >= 1
        go_gaussian_data.append(sample)
rounded_go_data = np.round(go_gaussian_data)

# Create a global stopwatch instance
global_stopwatch = Stopwatch()

def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load configuration parameters from a JSON file.
    
    Parameters:
        config_file (str): Path to the configuration file.
        
    Returns:
        dict: Configuration parameters.
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading config file {config_file}: {e}")
        sys.exit(1)

@dataclass
class EncoderData:
    """ Represents a single encoder reading."""
    timestamp: int
    distance: float
    speed: float

    def __repr__(self):
        return (f"EncoderData(timestamp={self.timestamp}, "
                f"distance={self.distance:.3f} mm, speed={self.speed:.3f} mm/s)")

class DataLogger:
    """
    Logs movement data to a CSV file.
    """
    def __init__(self, filename):
        """
        Initialize the data logger.
        
        Args:
            filename (str): Path to the CSV file.
        """
        self.filename = filename
        self.fieldnames = ['timestamp', 'distance', 'speed']
        file_exists = os.path.isfile(self.filename)
        self.file = open(self.filename, 'a', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        if not file_exists:
            self.writer.writeheader()

    def log(self, data: EncoderData):
        self.writer.writerow({'timestamp': data.timestamp, 'distance': data.distance, 'speed': data.speed})
        self.file.flush()

    def close(self):
        self.file.close()

class Corridor:
    """
    Class for generating infinite corridor geometric rendering
    """
    def __init__(self, base: ShowBase, config: Dict[str, Any]) -> None:
        """
        Initialize the corridor by creating segments for each face.
        
        Parameters:
            base (ShowBase): The Panda3D base instance.
            config (dict): Configuration parameters.
        """
        self.base = base
        self.segment_length: float = config["segment_length"]
        self.corridor_width: float = config["corridor_width"]
        self.wall_height: float = config["wall_height"]
        self.num_segments: int = config["num_segments"]
        self.left_wall_texture: str = config["left_wall_texture"]
        self.right_wall_texture: str = config["right_wall_texture"]
        self.ceiling_texture: str = config["ceiling_texture"]
        self.floor_texture: str = config["floor_texture"]
        self.special_wall: str = config["special_wall"]
        self.alternative_wall_texture_1 = config["alternative_wall_texture_1"]
        self.alternative_wall_texture_2 = config["alternative_wall_texture_2"]
        self.trial_data = config["trial_data"]
        self.probe_onset = config["probe_onset"]
        self.probe_duration = config["probe_duration"]
        
        # Write the rounded Gaussian data to subject_data.py
        with open(self.trial_data, "a") as f:
            f.write(
                f"rounded_gaussian_data = {repr(rounded_gaussian_data.tolist())}\n"
                f"rounded_stay_data = {repr(rounded_stay_data.tolist())}\n"
                f"rounded_go_data = {repr(rounded_go_data.tolist())}\n"
            )
        
        # Create a parent node for all corridor segments.
        self.parent: NodePath = base.render.attachNewNode("corridor")
        
        # Separate lists for each face.
        self.left_segments: list[NodePath] = []
        self.right_segments: list[NodePath] = []
        self.ceiling_segments: list[NodePath] = []
        self.floor_segments: list[NodePath] = []
        
        self.build_segments()
        
        # Add a task to change textures at a random interval.
        self.schedule_texture_change()

        # Start the stopwatch when the corridor is initialized
        global_stopwatch.start()

        # Initialize attributes
        self.segments_until_revert = 0  # Ensure this attribute exists
        self.texture_change_scheduled = False  # Flag to track texture change scheduling

    def build_segments(self) -> None:
        """ 
        Build the initial corridor segments using CardMaker.
        """
        for i in range(-self.num_segments // 2, self.num_segments // 2):  # Adjust range to include negative indices
            segment_start: float = i * self.segment_length
            
            # ==== Left Wall:
            cm_left: CardMaker = CardMaker("left_wall")
            cm_left.setFrame(0, self.segment_length, 0, self.wall_height)
            left_node: NodePath = self.parent.attachNewNode(cm_left.generate())
            left_node.setPos(-self.corridor_width / 2, segment_start, 0)
            left_node.setHpr(90, 0, 0)
            self.apply_texture(left_node, self.left_wall_texture)
            self.left_segments.append(left_node)
            
            # ==== Right Wall:
            cm_right: CardMaker = CardMaker("right_wall")
            cm_right.setFrame(0, self.segment_length, 0, self.wall_height)
            right_node: NodePath = self.parent.attachNewNode(cm_right.generate())
            right_node.setPos(self.corridor_width / 2, segment_start, 0)
            right_node.setHpr(-90, 0, 0)
            self.apply_texture(right_node, self.right_wall_texture)
            self.right_segments.append(right_node)
            
            # ==== Ceiling (Top):
            cm_ceiling: CardMaker = CardMaker("ceiling")
            cm_ceiling.setFrame(-self.corridor_width / 2, self.corridor_width / 2, 0, self.segment_length)
            ceiling_node: NodePath = self.parent.attachNewNode(cm_ceiling.generate())
            ceiling_node.setPos(0, segment_start, self.wall_height)
            ceiling_node.setHpr(0, 90, 0)
            self.apply_texture(ceiling_node, self.ceiling_texture)
            self.ceiling_segments.append(ceiling_node)
            
            # ==== Floor (Bottom):
            cm_floor: CardMaker = CardMaker("floor")
            cm_floor.setFrame(-self.corridor_width / 2, self.corridor_width / 2, 0, self.segment_length)
            floor_node: NodePath = self.parent.attachNewNode(cm_floor.generate())
            floor_node.setPos(0, segment_start, 0)
            floor_node.setHpr(0, -90, 0)
            self.apply_texture(floor_node, self.floor_texture)
            self.floor_segments.append(floor_node)
            
    def apply_texture(self, node: NodePath, texture_path: str) -> None:
        """
        Load and apply the texture to a geometry node.
        
        Parameters:
            node (NodePath): The node to which the texture will be applied.
        """
        texture: Texture = self.base.loader.loadTexture(texture_path)
        node.setTexture(texture)
        
    def recycle_segment(self, direction: str) -> None:
        """
        Recycle the front segments by repositioning them to the end of the corridor.
        This is called when the player has advanced by one segment length.
        """
        if direction == "forward":
            # Calculate new base Y position from the last segment in the left wall.
            new_y: float = self.left_segments[-1].getY() + self.segment_length

            # Recycle left wall segment.
            left_seg: NodePath = self.left_segments.pop(0)
            left_seg.setY(new_y)
            self.left_segments.append(left_seg)

            # Recycle right wall segment.
            right_seg: NodePath = self.right_segments.pop(0)
            right_seg.setY(new_y)
            self.right_segments.append(right_seg)

            # Recycle ceiling segment.
            ceiling_seg: NodePath = self.ceiling_segments.pop(0)
            ceiling_seg.setY(new_y)
            self.ceiling_segments.append(ceiling_seg)

            # Recycle floor segment.
            floor_seg: NodePath = self.floor_segments.pop(0)
            floor_seg.setY(new_y)
            self.floor_segments.append(floor_seg)

        elif direction == "backward":
            # Calculate new base Y position from the first segment in the left wall.
            new_y: float = self.left_segments[0].getY() - self.segment_length

            # Recycle left wall segment.
            left_seg: NodePath = self.left_segments.pop(-1)
            left_seg.setY(new_y)
            self.left_segments.insert(0, left_seg)

            # Recycle right wall segment.
            right_seg: NodePath = self.right_segments.pop(-1)
            right_seg.setY(new_y)
            self.right_segments.insert(0, right_seg)

            # Recycle ceiling segment.
            ceiling_seg: NodePath = self.ceiling_segments.pop(-1)
            ceiling_seg.setY(new_y)
            self.ceiling_segments.insert(0, ceiling_seg)

            # Recycle floor segment.
            floor_seg: NodePath = self.floor_segments.pop(-1)
            floor_seg.setY(new_y)
            self.floor_segments.insert(0, floor_seg)
            
    def change_wall_textures(self, task: Task = None) -> Task:
        """
        Change the textures of the left and right walls to a randomly selected texture.
        
        Parameters:
            task (Task): The Panda3D task instance (optional).
            
        Returns:
            Task: Continuation signal for the task manager.
        """
        # Define a list of possible wall textures
        wall_textures = [
            self.special_wall,  # Texture 1
            self.alternative_wall_texture_2   # Texture 2
        ]
        
        # Randomly select a texture
        selected_texture = random.choice(wall_textures)
        
        # Write the selected texture to the subject_data.txt file
        with open(self.trial_data, "a") as f:
            f.write(f"Selected texture: {selected_texture}\n")
        
        # Apply the selected texture to the walls
        for left_node in self.left_segments:
            self.apply_texture(left_node, selected_texture)
        for right_node in self.right_segments:
            self.apply_texture(right_node, selected_texture)
        
        # Print the elapsed time since the corridor was initialized
        elapsed_time = global_stopwatch.get_elapsed_time()
        with open(self.trial_data, "a") as f:
            f.write(f"Wall texture changed. Elapsed time: {round(elapsed_time, 2)} seconds\n")
        
        # Determine the stay_or_go_data based on the selected texture
        if selected_texture == self.special_wall:
            stay_or_go_data = rounded_go_data
        else:
            stay_or_go_data = rounded_stay_data
        
        # Set the counter for segments to revert textures using a random value from stay_or_go_data
        self.segments_until_revert = int(random.choice(stay_or_go_data))
        
        # Write the segments_until_revert value to the trial_data file
        with open(self.trial_data, "a") as f:
            f.write(f"Segments until revert: {self.segments_until_revert}\n")
        
        # Return Task.done if task is None
        return Task.done if task is None else task.done

    def change_wall_textures_temporarily_once(self, task: Task = None) -> Task:
        """
        Temporarily change the wall textures for 1 second and then revert them back.
        This method ensures the temporary texture change happens only once.
        
        Parameters:
            task (Task): The Panda3D task instance (optional).
            
        Returns:
            Task: Continuation signal for the task manager.
        """
        # Define a list of possible wall textures
        temporary_wall_textures = [
            self.floor_texture,  # Texture 1
            self.alternative_wall_texture_1,   # Texture 2
            self.ceiling_texture,  # Texture 3
        ]

        # Randomly select a texture
        selected_temporary_texture = random.choice(temporary_wall_textures)

        # Write the selected texture to the subject_data.txt file
        with open(self.trial_data, "a") as f:
            f.write(f"Selected probe texture: {selected_temporary_texture}\n")

        ## Print the elapsed time since the corridor was initialized
        elapsed_time = global_stopwatch.get_elapsed_time() 
        with open(self.trial_data, "a") as f:
            f.write(f"Probe flashed. Elapsed time: {round(elapsed_time, 2)} seconds\n")

        # Apply the selected texture to the walls
        for left_node in self.left_segments:
            self.apply_texture(left_node, selected_temporary_texture)
        for right_node in self.right_segments:
            self.apply_texture(right_node, selected_temporary_texture)
        
        # Schedule a task to revert the textures back after 1 second
        self.base.taskMgr.doMethodLater(self.probe_duration, self.revert_temporary_textures, "RevertWallTextures")
        
        # Do not reset the texture_change_scheduled flag here to prevent repeated scheduling
        return Task.done if task is None else task.done
    
    def revert_temporary_textures(self, task: Task = None) -> Task:
        """
        Revert the temporary textures of the left and right walls back to their original textures.
        
        Parameters:
            task (Task): The Panda3D task instance (optional).
            
        Returns:
            Task: Continuation signal for the task manager.
        """
        # Reapply the original textures to the walls
        for left_node in self.left_segments:
            self.apply_texture(left_node, self.left_wall_texture)
        for right_node in self.right_segments:
            self.apply_texture(right_node, self.right_wall_texture)
        
        # Return Task.done if task is None
        return Task.done if task is None else task.done

    def revert_wall_textures(self, task: Task = None) -> Task:
        """
        Revert the textures of the left and right walls to their original textures.
        
        Parameters:
            task (Task): The Panda3D task instance (optional).
            
        Returns:
            Task: Continuation signal for the task manager.
        """
        # Reapply the original textures to the walls
        for left_node in self.left_segments:
            self.apply_texture(left_node, self.left_wall_texture)
        for right_node in self.right_segments:
            self.apply_texture(right_node, self.right_wall_texture)
        
        # Schedule a task to change the wall textures temporarily after reverting
        self.base.taskMgr.doMethodLater(self.probe_onset, self.change_wall_textures_temporarily_once, "ChangeWallTexturesTemporarilyOnce")
        
        # Return Task.done if task is None
        return Task.done if task is None else task.done

    def schedule_texture_change(self) -> None:
        """
        Schedule the next texture change after a random number of wall segments are recycled.
        """
        # Ensure segments_until_revert is initialized
        if not hasattr(self, 'segments_until_revert'):
            self.segments_until_revert = 0

        # Randomly determine the number of segments after which to change the texture
        segments_to_wait = random.choice(rounded_gaussian_data)
        
        # Write the selected number of segments to the subject_data.txt file
        with open(self.trial_data, "a") as f:
            f.write(f"Segments to wait for texture change: {int(segments_to_wait)}\n")
            f.write("\n")
        
        self.segments_until_texture_change = segments_to_wait + self.segments_until_revert

    def update_texture_change(self) -> None:
        """
        Check if the required number of segments has been recycled and change the texture if needed.
        """
        if self.segments_until_texture_change <= 0:
            self.change_wall_textures(None)  # Trigger the texture change
            self.schedule_texture_change()  # Schedule the next texture change
        
        # Check if textures need to be reverted
        if hasattr(self, 'segments_until_revert') and self.segments_until_revert > 0:
            self.segments_until_revert -= 1
            if self.segments_until_revert == 0:
                self.revert_wall_textures(None)  # Revert textures
            
class FogEffect:
    """
    Class to manage and apply fog to the scene.
    """
    def __init__(self, base: ShowBase, fog_color, density):
        """
        Initialize the fog effect.
        
        Parameters:
            base (ShowBase): The Panda3D base instance.
            fog_color (tuple): RGB color for the fog (default is white).
            near_distance (float): The near distance where the fog starts.
            far_distance (float): The far distance where the fog completely obscures the scene.
        """
        self.base = base
        self.fog = Fog("fog")
        base.setBackgroundColor(fog_color)
        
        # Set fog color.
        self.fog.setColor(*fog_color)
        
        # Set the density for the fog.
        self.fog.setExpDensity(density)
        
        # Attach the fog to the root node to affect the entire scene.
        render.setFog(self.fog)


class SerialInputManager(DirectObject.DirectObject):
    """
    Manages serial input via the pyserial interface.
    
    This class abstracts the serial connection and starts a thread that listens
    for serial data.
    """
    def __init__(self, serial_port: str, baudrate: int = 57600, messenger: DirectObject = None, test_mode: bool = False) -> None:
        self._port = serial_port
        self._baud = baudrate
        self.test_mode = test_mode
        self.test_file = None
        self.test_reader = None
        self.test_data = None

        if self.test_mode:
            try:
                self.test_file = open('test.csv', 'r')
                self.test_reader = csv.reader(self.test_file)
                next(self.test_reader)  # Skip header
            except Exception as e:
                print(f"Failed to open test.csv: {e}")
                raise
        else:
            try:
                self.serial = serial.Serial(self._port, self._baud, timeout=1)
            except serial.SerialException as e:
                print(f"{self.__class__}: I failed to open serial port {self._port}: {e}")
                raise

        self.accept('readSerial', self._store_data)
        self.data = EncoderData(0, 0.0, 0.0)
        self.messenger = messenger

    def _store_data(self, data: EncoderData):
        self.data = data

    def _read_serial(self, task: Task) -> Task:
        """Internal loop for continuously reading lines from the serial port or test.csv."""
        if self.test_mode:
            try:
                line = next(self.test_reader)
                if line:
                    data = self._parse_line_from_csv(line)
                    if data:
                        self.messenger.send("readSerial", [data])
            except StopIteration:
                # Restart the test file reading from the beginning
                self.test_file.seek(0)
                self.test_reader = csv.reader(self.test_file)
                next(self.test_reader)  # Skip header
        else:
            # Read a line from the Teensy board
            raw_line = self.serial.readline()
            # Decode and strip newline characters
            line = raw_line.decode('utf-8', errors='replace').strip()
            if line:
                data = self._parse_line(line)
                if data:
                    self.messenger.send("readSerial", [data])

        return Task.cont

    def _parse_line(self, line: str):
        """
        Parse a line of serial output.

        Expected line formats:
          - "timestamp,distance,speed"  or
          - "distance,speed"

        Args:
            line (str): A single line from the serial port.

        Returns:
            EncoderData: An instance with parsed values, or None if parsing fails.
        """
        parts = line.split(',')
        try:
            if len(parts) == 3:
                # Format: timestamp, distance, speed
                timestamp = int(parts[0].strip())
                distance = float(parts[1].strip())
                speed = float(parts[2].strip())
                return EncoderData(distance=distance, speed=speed, timestamp=timestamp)
            elif len(parts) == 2:
                # Format: distance, speed
                distance = float(parts[0].strip())
                speed = float(parts[1].strip())
                return EncoderData(distance=distance, speed=speed)
            else:
                # Likely a header or message line (non-data)
                return None
        except ValueError:
            # Non-numeric data (e.g., header info)
            return None

    def _parse_line_from_csv(self, line: list):
            """
            Parse a line from the test.csv file.
    
            Expected line format:
            - "timestamp,distance,speed"
    
            Args:
                line (list): A single line from the CSV file.
    
            Returns:
                EncoderData: An instance with parsed values, or None if parsing fails.
            """
            try:
                timestamp = int(line[0].strip())
                distance = float(line[1].strip())
                speed = float(line[2].strip())
                return EncoderData(distance=distance, speed=speed, timestamp=timestamp)
            except ValueError:
                # Non-numeric data (e.g., header info)
                return None
    
    def close(self):
        if self.test_mode and self.test_file:
            self.test_file.close()
        elif not self.test_mode and self.serial:
            self.serial.close()

class RewardOrPuff(FSM):
    """
    FSM to manage the reward or puff state.
    """
    def __init__(self, base: ShowBase, config: Dict[str, Any]) -> None:
        """
        Initialize the FSM with the base and configuration.

        Parameters:
            base (ShowBase): The Panda3D base instance.
            config (dict): Configuration parameters.
        """
        FSM.__init__(self, "RewardOrPuff")
        self.base = base
        self.config = config
        self.accept('puff-event', self.request, ['Puff'])
        self.accept('reward-event', self.request, ['Reward'])
        self.accept('neutral-event', self.request, ['Neutral'])

    def enterPuff(self):
        """
        Enter the Puff state.
        """
        print("Entering Puff state")
        self.base.taskMgr.doMethodLater(1.0, self._transitionToNeutral, 'return-to-neutral')

    def exitPuff(self):
        """
        Exit the Puff state.
        """
        print("Exiting Puff state")
        
    def enterReward(self):
        print("Entering Reward state: give reward (e.g. juice drop)")
        self.base.taskMgr.doMethodLater(1.0, self._transitionToNeutral, 'return-to-neutral')

    def exitReward(self):
        print("Exiting Reward state.")

    def enterNeutral(self):
        print("Entering Neutral state: waiting...")

    def exitNeutral(self):
        print("Exiting Neutral state.")

    def _transitionToNeutral(self, task):
        self.request('Neutral')
        return Task.done

class MousePortal(ShowBase):
    """
    Main application class for the infinite corridor simulation.
    """
    def __init__(self, config_file) -> None:
        """
        Initialize the application, load configuration, set up the camera, user input,
        corridor geometry, and add the update task.
        """
        ShowBase.__init__(self)
        
        # Load configuration from JSON
        with open(config_file, 'r') as f:
            self.cfg: Dict[str, Any] = load_config(config_file)

        # Set window properties
        wp = WindowProperties()
        wp.setSize(self.cfg["window_width"], self.cfg["window_height"])
        self.setFrameRateMeter(False)
        self.disableMouse()  # Disable default mouse-based camera control
        wp.setCursorHidden(True)
        wp.setFullscreen(True)
        wp.setUndecorated(True)
        self.win.requestProperties(wp)
        
        # Initialize camera parameters
        self.camera_position: float = 0.0
        self.camera_velocity: float = 0.0
        self.speed_scaling: float = self.cfg.get("speed_scaling", 5.0)
        self.camera_height: float = self.cfg.get("camera_height", 2.0)  
        self.camera.setPos(0, self.camera_position, self.camera_height)
        self.camera.setHpr(0, 0, 0)
        
        # Set up key mapping for keyboard input
        self.key_map: Dict[str, bool] = {"forward": False, "backward": False}
        self.accept("arrow_up", self.set_key, ["forward", True])
        self.accept("arrow_up-up", self.set_key, ["forward", False])
        self.accept("arrow_down", self.set_key, ["backward", True])
        self.accept("arrow_down-up", self.set_key, ["backward", False])
        self.accept('escape', self.userExit)

        # Set up treadmill input
        self.treadmill = SerialInputManager(
            serial_port=self.cfg["serial_port"],
            messenger=self.messenger,
            test_mode=self.cfg.get("test_mode", False)
        )

        # Create corridor geometry
        self.corridor: Corridor = Corridor(self, self.cfg)
        self.segment_length: float = self.cfg["segment_length"]
        
        # Initialize the RewardOrPuff FSM
        self.fsm = RewardOrPuff(self, self.cfg)
        
        # Variable to track movement since last recycling
        self.distance_since_recycle: float = 0.0
        
        # Movement speed (units per second)
        self.movement_speed: float = 10.0
        
        # Initialize data logger
        self.data_logger = DataLogger(self.cfg["data_logging_file"])

        # Add the update task
        self.taskMgr.add(self.update, "updateTask")
        
        # Initialize fog effect
        self.fog_effect = FogEffect(
            self,
            density=self.cfg["fog_density"],
            fog_color=(0.5, 0.5, 0.5)
        )
        
        # Set up task chain for serial input
        self.taskMgr.setupTaskChain(
            "serialInputDevice",
            numThreads=1,
            tickClock=None,
            threadPriority=None,
            frameBudget=None,
            frameSync=True,
            timeslicePriority=None
        )
        self.taskMgr.add(self.treadmill._read_serial, name="readSerial")

        # Enable verbose messaging
        #self.messenger.toggleVerbose()

    def set_key(self, key: str, value: bool) -> None:
        """
        Update the key state for the given key.
        
        Parameters:
            key (str): The key identifier.
            value (bool): True if pressed, False if released.
        """
        self.key_map[key] = value
        
    def update(self, task: Task) -> Task:
        """
        Update the camera's position based on user input and recycle corridor segments
        when the player moves forward beyond one segment.
        
        Parameters:
            task (Task): The Panda3D task instance.
            
        Returns:
            Task: Continuation signal for the task manager.
        """
        dt: float = globalClock.getDt()
        move_distance: float = 0.0
        
        # Update camera velocity based on key input
        if self.key_map["forward"]:
            self.camera_velocity = self.speed_scaling
        elif self.key_map["backward"]:
            self.camera_velocity = -self.speed_scaling
        else:
            self.camera_velocity = 0.0
        
        self.camera_velocity = (int(self.treadmill.data.speed) / self.cfg["treadmill_speed_scaling"])

        # Update camera position (movement along the Y axis)
        self.camera_position += self.camera_velocity * dt
        move_distance = self.camera_velocity * dt
        self.camera.setPos(0, self.camera_position, self.camera_height)
        
        # Recycle corridor segments when the camera moves beyond one segment length
        if move_distance > 0:
            self.distance_since_recycle += move_distance
            while self.distance_since_recycle >= self.segment_length:
                self.corridor.recycle_segment(direction="forward")
                self.distance_since_recycle -= self.segment_length
                self.corridor.segments_until_texture_change -= 1
                self.corridor.update_texture_change()
        elif move_distance < 0:
            self.distance_since_recycle += move_distance
            while self.distance_since_recycle <= -self.segment_length:
                self.corridor.recycle_segment(direction="backward")
                self.distance_since_recycle += self.segment_length

        # Log movement data (timestamp, distance, speed)
        self.data_logger.log(self.treadmill.data)

        # FSM state transition logic
        selected_texture = self.corridor.left_wall_texture

        if selected_texture == self.corridor.special_wall:
            if self.fsm.state != 'Reward':  # Only request if not already in the 'Reward' state
                print("Requesting Reward state")
                self.fsm.request('Reward')
        elif selected_texture == self.corridor.alternative_wall_texture_2:
            if self.fsm.state != 'Puff':  # Only request if not already in the 'Puff' state
                print("Requesting Puff state")
                self.fsm.request('Puff')
        else:
            if self.fsm.state != 'Neutral':  # Only request if not already in the 'Neutral' state
                print("Requesting Neutral state")
                self.fsm.request('Neutral')
        
        return Task.cont

if __name__ == "__main__":
    app = MousePortal("cfg.json")
    app.run()