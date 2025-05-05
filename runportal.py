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
from panda3d.core import CardMaker, NodePath, Texture, WindowProperties, Fog, GraphicsPipe
from direct.showbase import DirectObject
from direct.fsm.FSM import FSM

class Stopwatch:
    """
    A simple stopwatch class to measure elapsed time.
    """
    def __init__(self):
        self.start_time = None
        self.elapsed_time = 0
        self.running = False

    def start(self):
        """
        Start or resume the stopwatch.
        """
        if not self.running:
            self.start_time = time.time() - self.elapsed_time
            self.running = True

    def stop(self):
        """
        Stop the stopwatch and record the elapsed time.
        """
        if self.running:
            self.elapsed_time = time.time() - self.start_time
            self.running = False

    def reset(self):
        """
        Reset the stopwatch to zero.
        """
        self.start_time = None
        self.elapsed_time = 0
        self.running = False

    def get_elapsed_time(self):
        """
        Get the elapsed time in seconds.
        
        Returns:
            float: The elapsed time in seconds.
        """
        if self.running:
            return time.time() - self.start_time
        return self.elapsed_time

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

class DataGenerator:
    """
    A class to generate Gaussian data based on configuration parameters.
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the DataGenerator with configuration.

        Args:
            config (dict): Configuration dictionary containing Gaussian parameters.
        """
        self.config = config

    def generate_gaussian_data(self, key: str, size: int = 250, min_value: float = None) -> np.ndarray:
        """
        Generate Gaussian data based on the configuration.

        Args:
            key (str): The key in the configuration for the Gaussian parameters.
            size (int): The number of samples to generate.
            min_value (float): Minimum value to accept (optional).

        Returns:
            np.ndarray: Rounded Gaussian data.
        """
        loc = self.config[key]["loc"]
        scale = self.config[key]["scale"]

        if min_value is not None:
            data = []
            while len(data) < size:
                sample = np.random.normal(loc=loc, scale=scale)
                if sample >= min_value:
                    data.append(sample)
            return np.round(data)
        else:
            data = np.random.normal(loc=loc, scale=scale, size=size)
            return np.round(data)
        
@dataclass
class CapacitiveData:
    """
    Represents a single capacitive sensor reading.
    """
    capacitive_value: int

    def __repr__(self):
        return f"CapacitiveData(capacitive_value={self.capacitive_value})"

class CapacitiveSensorLogger(DirectObject.DirectObject):
    """
    Logs capacitive sensor data to a CSV file.
    """
    def __init__(self, filename: str) -> None:
        """
        Initialize the capacitive sensor logger.

        Args:
            filename (str): Path to the CSV file.
        """
        self.filename = filename
        self.fieldnames = ['timestamp', 'capacitive_value']
        file_exists = os.path.isfile(self.filename)
        self.file = open(self.filename, 'a', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        if not file_exists:
            self.writer.writeheader()

        # Listen for capacitive data events
        self.accept('readCapacitive', self.log)

    def log(self, data: CapacitiveData) -> None:
        """
        Log the capacitive sensor data to the CSV file.

        Args:
            data (CapacitiveData): The capacitive sensor data to log.
        """
        self.writer.writerow({
            'timestamp': round(global_stopwatch.get_elapsed_time(), 2),  # Elapsed time from the global stopwatch
            'capacitive_value': data.capacitive_value
        })
        self.file.flush()

    def close(self) -> None:
        """
        Close the CSV file.
        """
        self.file.close()

@dataclass
class TreadmillData:
    """ Represents a single encoder reading."""
    timestamp: int
    distance: float
    speed: float

    def __repr__(self):
        return (f"TreadmillData(timestamp={self.timestamp}, "
                f"distance={self.distance:.3f} mm, speed={self.speed:.3f} mm/s)")

class TreadmillLogger:
    """
    Logs movement data to a CSV file.
    """
    def __init__(self, filename):
        """
        Initialize the treadmill logger.
        
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

    def log(self, data: TreadmillData):
        self.writer.writerow({'timestamp': data.timestamp, 'distance': data.distance, 'speed': data.speed})
        self.file.flush()

    def close(self):
        self.file.close()

class Corridor:
    def __init__(self, base: ShowBase, config: Dict[str, Any],
                 rounded_gaussian_data: np.ndarray,
                 rounded_stay_data: np.ndarray,
                 rounded_go_data: np.ndarray) -> None:
        """
        Initialize the corridor by creating segments for each face.
        
        Parameters:
            base (ShowBase): The Panda3D base instance.
            config (dict): Configuration parameters.
            rounded_gaussian_data (np.ndarray): Gaussian data for texture changes.
            rounded_stay_data (np.ndarray): Gaussian data for stay textures.
            rounded_go_data (np.ndarray): Gaussian data for go textures.
        """
        self.base = base
        self.config = config
        self.rounded_gaussian_data = rounded_gaussian_data
        self.rounded_stay_data = rounded_stay_data
        self.rounded_go_data = rounded_go_data

        self.segment_length: float = config["segment_length"]
        self.corridor_width: float = config["corridor_width"]
        self.wall_height: float = config["wall_height"]
        self.num_segments: int = config["num_segments"]
        self.left_wall_texture: str = config["left_wall_texture"]
        self.right_wall_texture: str = config["right_wall_texture"]
        self.ceiling_texture: str = config["ceiling_texture"]
        self.floor_texture: str = config["floor_texture"]
        self.go_texture: str = config["go_texture"]
        self.neutral_stim_1 = config["neutral_stim_1"]
        self.stop_texture = config["stop_texture"]
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
            self.go_texture,  # Texture 1
            self.stop_texture   # Texture 2
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
        if selected_texture == self.go_texture:
            stay_or_go_data = self.rounded_go_data
        else:
            stay_or_go_data = self.rounded_stay_data
        
        # Set the counter for segments to revert textures using a random value from stay_or_go_data
        self.segments_until_revert = int(random.choice(stay_or_go_data))
        self.base.zone_length = self.segments_until_revert
        
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
            self.neutral_stim_1,   # Texture 2
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
        segments_to_wait = random.choice(self.rounded_gaussian_data)
        
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
            # Trigger the texture change
            self.change_wall_textures(None)
            
            # Check if the new texture is the go texture
            new_front_texture = self.left_segments[0].getTexture().getFilename()
            if new_front_texture == self.go_texture:
                # Update the enter_go_time in the MousePortal instance
                self.base.enter_go_time = global_stopwatch.get_elapsed_time()
                #print(f"enter_go_time updated to {self.base.enter_go_time:.2f} seconds")
            elif new_front_texture == self.stop_texture:
                # Update the enter_stay_time in the MousePortal instance
                self.base.enter_stay_time = global_stopwatch.get_elapsed_time()
                #print(f"enter_stay_time updated to {self.base.enter_stay_time:.2f} seconds")

            # Schedule the next texture change
            self.schedule_texture_change()

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
    for serial data from the Teensy Board and Arduino.
    """
    def __init__(self, teensy_port: str, teensy_baudrate: int = 57600, 
                 arduino_serial: serial.Serial = None, 
                 messenger: DirectObject = None, test_mode: bool = False) -> None:
        self.teensy_port = teensy_port
        self.teensy_baudrate = teensy_baudrate
        self.arduino_serial = arduino_serial  # Use the shared instance
        self.test_mode = test_mode
        self.test_file = None
        self.test_reader = None
        self.test_data = None

        # Initialize Teensy connection
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
                self.teensy_serial = serial.Serial(self.teensy_port, self.teensy_baudrate, timeout=1)
            except serial.SerialException as e:
                print(f"{self.__class__}: Failed to open Teensy serial port {self.teensy_port}: {e}")
                raise

        self.accept('readSerial', self._store_data)
        self.accept('readCapacitive', self._store_capacitive_data)
        self.data = TreadmillData(0, 0.0, 0.0)
        self.capacitive_data = CapacitiveData(0)
        self.messenger = messenger

    def _store_data(self, data: TreadmillData):
        self.data = data

    def _store_capacitive_data(self, data: CapacitiveData):
        self.capacitive_data = data

    def _read_teensy_serial(self, task: Task) -> Task:
        """Internal loop for continuously reading lines from the Teensy."""
        if self.teensy_serial:
            raw_line = self.teensy_serial.readline()
            line = raw_line.decode('utf-8', errors='replace').strip()
            if line:
                data = self._parse_line(line)
                if data:
                    self.messenger.send("readSerial", [data])
        return Task.cont

    def _read_arduino_serial(self, task: Task) -> Task:
        """Internal loop for continuously reading lines from the Arduino."""
        if self.arduino_serial:
            raw_line = self.arduino_serial.readline()
            line = raw_line.decode('utf-8', errors='replace').strip()
            try:
                # Attempt to parse the line as an integer
                capacitive_value = int(line)
                # Wrap the value in a CapacitiveData object
                capacitive_data = CapacitiveData(capacitive_value=capacitive_value)
                self.messenger.send("readCapacitive", [capacitive_data])
                #print("Parsed capacitive data:", capacitive_data)
            except ValueError:
                pass  # Ignore non-integer lines
        return Task.cont

    def _parse_line(self, line: str):
        """
        Parse a line of serial output from the Teensy.

        Expected line formats:
          - "timestamp,distance,speed"  or
          - "distance,speed"

        Args:
            line (str): A single line from the serial port.

        Returns:
            Treadmilldata: An instance with parsed values, or None if parsing fails.
        """
        parts = line.split(',')
        try:
            if len(parts) == 3:
                # Format: timestamp, distance, speed
                timestamp = int(parts[0].strip())
                distance = float(parts[1].strip())
                speed = float(parts[2].strip())
                return TreadmillData(distance=distance, speed=speed, timestamp=timestamp)
            elif len(parts) == 2:
                # Format: distance, speed
                distance = float(parts[0].strip())
                speed = float(parts[1].strip())
                return TreadmillData(distance=distance, speed=speed)
            else:
                # Likely a header or message line (non-data)
                return None
        except ValueError:
            # Non-numeric data (e.g., header info)
            return None

    def _parse_capacitive_line(self, line: str):
        """
        Parse a line of capacitive sensor data from the Arduino.

        Expected line format:
          - A single integer value.

        Args:
            line (str): A single line from the serial port.

        Returns:
            CapacitiveData: An instance with the parsed integer value, or None if parsing fails.
        """
        try:
            capacitive_value = int(line.strip())  # Attempt to parse the line as an integer
            return CapacitiveData(capacitive_value=capacitive_value)
        except ValueError:
            # If the line is not a valid integer, return None
            print("no data")

    def close(self):
        if self.test_mode and self.test_file:
            self.test_file.close()
        if not self.test_mode:
            if self.teensy_serial:
                self.teensy_serial.close()
            if self.arduino_serial:
                self.arduino_serial.close()

class SerialOutputManager(DirectObject.DirectObject):
    """
    Manages serial output to an Arduino.
    
    This class abstracts the serial connection and provides methods to send output signals
    based on input from the FSM class.
    """
    def __init__(self, arduino_serial: serial.Serial) -> None:
        self.serial = arduino_serial  # Use the shared instance

    def send_signal(self, signal: Any) -> None:
        """
        Send a signal to the Arduino.
        
        Parameters:
            signal (Any): The signal to send, e.g., 'reward', 'puff', or an integer.
        """
        if self.serial.is_open:
            try:
                if isinstance(signal, int):
                    # Send larger integers
                    self.serial.write(str(signal).encode() + b"\n")
                elif isinstance(signal, str):
                    # Send strings as UTF-8 encoded bytes
                    self.serial.write(f"{signal}".encode('utf-8'))
                else:
                    raise ValueError("Unsupported signal type. Must be int or str.")
               #print(f"Sent signal: {signal}")
            except Exception as e:
                print(f"Failed to send signal: {e}")
        else:
            print("Arduino serial port is not open.")

    def close(self) -> None:
        """Close the serial connection."""
        if self.serial:
            self.serial.close()
            print("Arduino serial port closed.")

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
        self.trial_data = config["trial_data"]
        self.accept('puff-event', self.request, ['Puff'])
        self.accept('reward-event', self.request, ['Reward'])
        self.accept('neutral-event', self.request, ['Neutral'])

    def enterPuff(self):
        """
        Enter the Puff state.
        """
        with open(self.trial_data, "a") as f:
            f.write(f"Mouse puffed at {global_stopwatch.get_elapsed_time():.2f} seconds\n")
        self.base.serial_output.send_signal(12500)
        self.base.taskMgr.doMethodLater(1.0, self._transitionToNeutral, 'return-to-neutral')

    def exitPuff(self):
        """
        Exit the Puff state.
        """
        #print("Exiting Puff state")
        
    def enterReward(self):
        with open(self.trial_data, "a") as f:
            f.write(f"Mouse rewarded at {global_stopwatch.get_elapsed_time():.2f} seconds\n")
        self.base.serial_output.send_signal(21000)
        self.base.taskMgr.doMethodLater(1.0, self._transitionToNeutral, 'return-to-neutral')

    def exitReward(self):
        """
        Exit the Reward state."""
        #print("Exiting Reward state.")

    def enterNeutral(self):
        """
        Enter the Neutral state."""
        #print("Entering Neutral state: waiting...")

    def exitNeutral(self):
        """
        Exit the Neutral state."""
        #print("Exiting Neutral state.")

    def _transitionToNeutral(self, task):
        """
        Transition to the Neutral state only if the wall texture is the original wall texture.
        """
        # Get the current texture of the left wall
        current_texture = self.base.corridor.left_segments[0].getTexture().getFilename()

        # Check if the current texture matches the original wall texture
        if current_texture == self.base.corridor.left_wall_texture:
            self.request('Neutral')
        else:
            pass

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

        # Get the display width and height for both monitors
        pipe = self.win.getPipe()
        display_width = pipe.getDisplayWidth()
        display_height = pipe.getDisplayHeight()

        # Set window properties to span across both monitors
        wp = WindowProperties()
        wp.setSize(display_width * 2, display_height)  # Double the width for two monitors
        wp.setOrigin(0, 0)  # Start at the leftmost edge
        wp.setFullscreen(False)  # Ensure it's not in fullscreen mode
        self.setFrameRateMeter(False)
        self.disableMouse()  # Disable default mouse-based camera control
        wp.setCursorHidden(True)
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

        # Set up shared Arduino serial connection
        self.arduino_serial = serial.Serial(
            self.cfg["arduino_port"],
            self.cfg["arduino_baudrate"],
            timeout=1
        )

        # Set up treadmill input
        self.treadmill = SerialInputManager(
            teensy_port=self.cfg["teensy_port"],
            teensy_baudrate=self.cfg["teensy_baudrate"],
            arduino_serial=self.arduino_serial,  # Pass the shared instance
            messenger=self.messenger,
            test_mode=self.cfg.get("test_mode", False)
        )

        # Set up serial output to Arduino
        self.serial_output = SerialOutputManager(
            arduino_serial=self.arduino_serial  # Pass the shared instance
        )

        # Initialize the DataGenerator
        data_generator = DataGenerator(self.cfg)

        # Generate Gaussian data
        self.rounded_gaussian_data = data_generator.generate_gaussian_data("gaussian_data")
        self.rounded_stay_data = data_generator.generate_gaussian_data("stay_gaussian_data", min_value=1)
        self.rounded_go_data = data_generator.generate_gaussian_data("go_gaussian_data", min_value=1)

        # Create corridor geometry and pass Gaussian data
        self.corridor: Corridor = Corridor(
            base=self,
            config=self.cfg,
            rounded_gaussian_data=self.rounded_gaussian_data,
            rounded_stay_data=self.rounded_stay_data,
            rounded_go_data=self.rounded_go_data
        )
        self.segment_length: float = self.cfg["segment_length"]
        
        # Initialize the RewardOrPuff FSM
        self.fsm = RewardOrPuff(self, self.cfg)
        self.reward_time = self.cfg["reward_time"]
        self.puff_time = self.cfg["puff_time"]
        self.zone_length = 0

        # Variable to track movement since last recycling
        self.distance_since_recycle: float = 0.0
        
        # Movement speed (units per second)
        self.movement_speed: float = 10.0
        
        # Initialize treadmill logger
        self.treadmill_logger = TreadmillLogger(self.cfg["treadmill_logging_file"])
        self.capacitive_logger = CapacitiveSensorLogger(self.cfg["capacitive_logging_file"])

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
            "teensySerialInput",
            numThreads=1,
            tickClock=None,
            threadPriority=None,
            frameBudget=None,
            frameSync=True,
            timeslicePriority=None
        )
        self.taskMgr.setupTaskChain(
            "arduinoSerialInput",
            numThreads=1,
            tickClock=None,
            threadPriority=None,
            frameBudget=None,
            frameSync=True,
            timeslicePriority=None
        )
        self.taskMgr.add(self.treadmill._read_teensy_serial, name="readTeensySerial", taskChain="teensySerialInput")
        self.taskMgr.add(self.treadmill._read_arduino_serial, name="readArduinoSerial", taskChain="arduinoSerialInput")

        # Enable verbose messaging
        #self.messenger.toggleVerbose()

        # Add an attribute to track the number of segments passed for the FSM logic
        self.segments_with_go_texture = 0
        self.segments_with_stay_texture = 0

        # Add attributes to store time points
        self.enter_go_time = 0.0
        self.enter_stay_time = 0.0

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
                # Recycle the segment in the forward direction
                self.corridor.recycle_segment(direction="forward")
                self.distance_since_recycle -= self.segment_length
                self.corridor.segments_until_texture_change -= 1
                self.corridor.update_texture_change()

                # Check if the new front segment has the stay or go textures
                new_front_texture = self.corridor.left_segments[0].getTexture().getFilename()
                if new_front_texture == self.corridor.go_texture:
                    self.segments_with_go_texture += 1
                    #print(f"New segment with go texture counted: {self.segments_with_go_texture}")
                elif new_front_texture == self.corridor.stop_texture:
                    self.segments_with_stay_texture += 1
                    #print(f"New segment with stay texture counted: {self.segments_with_stay_texture}")
        
        elif move_distance < 0:
            self.distance_since_recycle += move_distance
            while self.distance_since_recycle <= -self.segment_length:
                self.corridor.recycle_segment(direction="backward")
                self.distance_since_recycle += self.segment_length

        # Log movement data (timestamp, distance, speed)
        self.treadmill_logger.log(self.treadmill.data)

        # FSM state transition logic
        # Dynamically get the current texture of the left wall
        selected_texture = self.corridor.left_segments[0].getTexture().getFilename()

        # Get the elapsed time from the global stopwatch
        current_time = global_stopwatch.get_elapsed_time()

        if selected_texture == self.corridor.stop_texture:
            #print(self.zone_length)
            if self.segments_with_stay_texture <= self.zone_length and self.fsm.state != 'Reward' and current_time >= self.enter_stay_time + (self.reward_time * self.zone_length):
                print("Requesting Reward state")
                self.fsm.request('Reward')
        elif selected_texture == self.corridor.go_texture:
            #print(self.zone_length)
            if self.segments_with_go_texture <= self.zone_length and self.fsm.state != 'Puff' and current_time >= self.enter_go_time + (self.puff_time * self.zone_length):
                print("Requesting Puff state")
                self.fsm.request('Puff')
        else:
            self.segments_with_go_texture = 0 
            self.segments_with_stay_texture = 0
            if self.fsm.state != 'Neutral':
                #print("Requesting Neutral state")
                self.fsm.request('Neutral')
        
        return Task.cont

    def close(self):
        if self.arduino_serial and self.arduino_serial.is_open:
            self.arduino_serial.close()
        if self.treadmill:
            self.treadmill.close()
        if self.serial_output:
            self.serial_output.close()

if __name__ == "__main__":
    app = MousePortal("cfg.json")
    app.run()