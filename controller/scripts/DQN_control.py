#!/usr/bin/env python3
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import rclpy
from rclpy.node import Node
import threading
from collections import deque
from squaternion import Quaternion
import subprocess

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # or "0", "2", or "3"
import tensorflow as tf
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR) # or logging.INFO, logging.WARNING, etc.

import math
import random
import time
import cv2

from gazebo_msgs.msg import ModelState, ContactsState, EntityState, ModelStates
from gazebo_msgs.srv import DeleteEntity, SpawnEntity, SetEntityState
from geometry_msgs.msg import Twist, Pose
from std_srvs.srv import Empty
import std_msgs.msg as std
from visualization_msgs.msg import Marker

from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np


from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import Adam





depth = None
imu = None
velocity = None
pose = None
collision = None
reset = False




class ReplayBuffer:

    def __init__(self, max_size, image_shape, numerical_dim, action_dim):
        self.max_size = max_size
        self.state_image_buffer = np.zeros((int(max_size), *image_shape), dtype=np.float32)
        self.state_numerical_buffer = np.zeros((int(max_size), numerical_dim), dtype=np.float32)
        self.action_buffer = np.zeros((int(max_size), action_dim), dtype=np.float32)
        self.next_state_image_buffer = np.zeros((int(max_size), *image_shape), dtype=np.float32)
        self.next_state_numerical_buffer = np.zeros((int(max_size), numerical_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((int(max_size),), dtype=np.float32)
        self.done_buffer = np.zeros((int(max_size),), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def store(self, state_image, state_numerical, action, next_state_image, next_state_numerical, reward, done):
        self.state_image_buffer[self.ptr] = state_image
        self.state_numerical_buffer[self.ptr] = state_numerical
        self.action_buffer[self.ptr] = action
        self.next_state_image_buffer[self.ptr] = next_state_image
        self.next_state_numerical_buffer[self.ptr] = next_state_numerical
        self.reward_buffer[self.ptr] = reward
        self.done_buffer[self.ptr] = done

        self.ptr = int((self.ptr + 1) % self.max_size)
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return dict(
            state_image=self.state_image_buffer[indices],
            state_numerical=self.state_numerical_buffer[indices],
            action=self.action_buffer[indices],
            next_state_image=self.next_state_image_buffer[indices],
            next_state_numerical=self.next_state_numerical_buffer[indices],
            reward=self.reward_buffer[indices],
            done=self.done_buffer[indices]
        )

    def size(self):
        return self.size


    def clear(self):

        self.ptr = 0
        self.size = 0
        self.state_image_buffer.fill(0)
        self.state_numerical_buffer.fill(0)
        self.action_buffer.fill(0)
        self.next_state_image_buffer.fill(0)
        self.next_state_numerical_buffer.fill(0)
        self.reward_buffer.fill(0)
        self.done_buffer.fill(0)



class Actor(Model):

    def __init__(self, image_shape, numerical_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action

        self.conv1 = layers.Conv2D(32, kernel_size=3, strides=2, activation='relu')
        self.conv2 = layers.Conv2D(64, kernel_size=3, strides=2, activation='relu')
        self.conv3 = layers.Conv2D(128, kernel_size=3, strides=2, activation='relu')
        self.flatten = layers.Flatten()

        self.dense1_numerical = layers.Dense(64, activation='relu')
        self.dense2_numerical = layers.Dense(64, activation='relu')

        self.dense1_combined = layers.Dense(256, activation='relu')
        self.dense2_combined = layers.Dense(256, activation='relu')

        self.action_output = layers.Dense(action_dim, activation='tanh')

    def call(self, inputs):
        image_input, numerical_input = inputs

        if len(image_input.shape) == 3:
            image_input = tf.expand_dims(image_input, axis=-1)

        x = self.conv1(image_input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)

        y = self.dense1_numerical(numerical_input)
        y = self.dense2_numerical(y)

        combined = layers.Concatenate()([x, y])

        combined = self.dense1_combined(combined)
        combined = self.dense2_combined(combined)

        action = self.action_output(combined)
        scaled_action = tf.multiply(action, self.max_action)

        return scaled_action



class Critic(Model):

    def __init__(self, image_shape, numerical_dim, action_dim):
        super(Critic, self).__init__()

        self.conv1 = layers.Conv2D(32, kernel_size=3, strides=2, activation='relu')
        self.conv2 = layers.Conv2D(64, kernel_size=3, strides=2, activation='relu')
        self.conv3 = layers.Conv2D(128, kernel_size=3, strides=2, activation='relu')
        self.flatten = layers.Flatten()

        self.dense1_numerical = layers.Dense(64, activation='relu')
        self.dense2_numerical = layers.Dense(64, activation='relu')

        self.action_input = layers.Input(shape=(action_dim,))

        self.dense1_combined = layers.Dense(256, activation='relu')
        self.dense2_combined = layers.Dense(256, activation='relu')

        self.q1_output = layers.Dense(1)
        self.q2_output = layers.Dense(1)

    def call(self, inputs):
        image_input, numerical_input, action_input = inputs

        if len(image_input.shape) == 3:
            image_input = tf.expand_dims(image_input, axis=-1)

        x = self.conv1(image_input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)

        y = self.dense1_numerical(numerical_input)
        y = self.dense2_numerical(y)

        combined = layers.Concatenate()([x, y, action_input])

        combined = self.dense1_combined(combined)
        combined = self.dense2_combined(combined)

        q1 = self.q1_output(combined)
        q2 = self.q2_output(combined)

        return q1, q2



class td3(object):

    def __init__(self, state_image_shape, numerical_dim, action_dim, max_action, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.005, 
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2, buffer_size=int(1e6), batch_size=64):
        self.actor = Actor(state_image_shape, numerical_dim, action_dim, max_action)
        self.critic = Critic(state_image_shape, numerical_dim, action_dim)
        
        self.target_actor = Actor(state_image_shape, numerical_dim, action_dim, max_action)
        self.target_critic = Critic(state_image_shape, numerical_dim, action_dim)

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.mse = mse = tf.keras.losses.MeanSquaredError()
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        
        self.replay_buffer = ReplayBuffer(buffer_size, state_image_shape, numerical_dim, action_dim)


    
    def update_target_weights(self, target_model, model, tau):
        target_weights = target_model.get_weights()
        model_weights = model.get_weights()
        new_weights = []
        for target_weight, model_weight in zip(target_weights, model_weights):
            new_weights.append(tau * model_weight + (1 - tau) * target_weight)
        target_model.set_weights(new_weights)



    def add_noise(self, action, noise_std=0.1):
        noise = np.random.normal(0, noise_std, size=action.shape)  # gaussian noise
        noise = np.clip(noise, -self.noise_clip, self.noise_clip) 
        noisy_action = action + noise
        return np.clip(noisy_action, -self.max_action, self.max_action) 




    def train(self, replay_buffer, iterations):
        av_Q = 0
        max_Q = -float('inf')
        av_loss = 0
        av_actor_loss = 0

        for it in range(iterations):
            # Sample a batch from the replay buffer
            batch = replay_buffer.sample(self.batch_size)
            state_image = tf.convert_to_tensor(batch['state_image'], dtype=tf.float32)
            state_numerical = tf.convert_to_tensor(batch['state_numerical'], dtype=tf.float32)
            action = tf.convert_to_tensor(batch['action'], dtype=tf.float32)
            reward = tf.convert_to_tensor(batch['reward'], dtype=tf.float32)
            done = tf.convert_to_tensor(batch['done'], dtype=tf.float32)
            next_state_image = tf.convert_to_tensor(batch['next_state_image'], dtype=tf.float32)
            next_state_numerical = tf.convert_to_tensor(batch['next_state_numerical'], dtype=tf.float32)

            # Get the target action from the actor-target network and add noise
            next_action = self.target_actor([next_state_image, next_state_numerical])

            noise = tf.random.normal(shape=tf.shape(next_action), mean=0.0, stddev=self.policy_noise)
            noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
            next_action = tf.clip_by_value(next_action + noise, -self.max_action, self.max_action)

            # Get target Q values from critic target network
            target_Q1, target_Q2 = self.target_critic([next_state_image, next_state_numerical, next_action])
            target_Q = tf.minimum(target_Q1, target_Q2)

            # Bellman equation for Q value
            target_Q = reward + (1 - done) * self.gamma * target_Q

            # Train Critic
            with tf.GradientTape() as tape:
                current_Q1, current_Q2 = self.critic([state_image, state_numerical, action])
                critic_loss = tf.reduce_mean(self.mse(current_Q1, target_Q)) + \
                              tf.reduce_mean(self.mse(current_Q2, target_Q))

            # Apply gradients to critic
            critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

            av_loss += critic_loss
            av_Q += tf.reduce_mean(target_Q)
            max_Q = max(max_Q, tf.reduce_max(target_Q))

            # Delayed policy update
            if it % self.policy_delay == 0:
                # Train Actor (policy gradient)
                with tf.GradientTape() as tape:
                    actions_pred = self.actor([state_image, state_numerical])
                    actor_loss = -tf.reduce_mean(self.critic([state_image, state_numerical, actions_pred])[0])

                # Apply gradients to actor
                actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

                # Soft update for the actor target
                self.update_target_weights(self.target_actor, self.actor, self.tau)

                # Soft update for the critic target
                self.update_target_weights(self.target_critic, self.critic, self.tau)

                av_actor_loss += actor_loss

        return av_loss / iterations, av_Q / iterations, max_Q, av_actor_loss / (iterations // self.policy_delay)





class Balloon(Node):
    def __init__(self, pos):
        super().__init__('stretch_marker')
        self.publisher_ = self.create_publisher(Marker, 'balloon', 10)  

        self.marker = Marker()
        self.marker.header.frame_id = 'simple_drone/odom'
        self.marker.header.stamp = self.get_clock().now().to_msg()
        self.marker.type = self.marker.SPHERE
        self.marker.id = 0
        self.marker.action = self.marker.ADD
        self.marker.scale.x = 0.5
        self.marker.scale.y = 0.5
        self.marker.scale.z = 0.5
        self.marker.color.r = 1.0
        self.marker.color.g = 0.0
        self.marker.color.b = 0.0
        self.marker.color.a = 1.0
        self.marker.pose.position.x = pos[0]
        self.marker.pose.position.y = pos[1]
        self.marker.pose.position.z = pos[2]
        self.get_logger().info("Publishing the balloon topic. Use RViz to visualize.")

    def publish_marker(self):
        self.publisher_.publish(self.marker)
  



class GazeboEnv(Node):
    """Superclass for all Gazebo environments."""

    def __init__(self):
        super().__init__('env')
        global depth, imu, velocity, pose, collision, reset
        self.get_logger().info('env started!')


        self.done = False
        self.target = False
        self.state = {'image': None, 'numerical': None}


        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        self.depth_image = depth
        self.imu_data = imu
        self.velocity_data = velocity
        self.position_data = pose
        self.collision = collision
        self.collision = False

        self.vel_pub = self.create_publisher(Twist, '/simple_drone/cmd_vel', 1)
        self.set_state = self.create_publisher(ModelState, 'gazebo/set_model_state', 10)
        self.takeoff = self.create_publisher(std.Empty, '/simple_drone/takeoff', 10)
        self.land = self.create_publisher(std.Empty, '/simple_drone/land', 10)

        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        self.reset_proxy = self.create_client(Empty, "/reset_world")

        self.reset_client = self.create_client(Empty, '/reset_world')
        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')

        #self.reset_client.wait_for_service()
        #self.delete_client.wait_for_service()
        #self.spawn_client.wait_for_service()

        self.req = Empty.Request
        self.TIME_DELTA = 0.3
        self.rate = self.create_rate(1)

        self.goal = [0.0, 0.0, 0.0]
        self.goal_viz = Balloon(self.goal)
        self.obstacles = []
        self.bounds = [[-5.0, 5.0], [-5.0, 5.0]]
        self.max_height = 6.0


        
    def get_obs(self):
        global depth, imu, velocity, pose, collision, reset

        self.depth_image = depth
        self.imu_data = imu
        self.velocity_data = velocity
        self.position_data = pose
        self.collision = collision

        return None


    def normalize_numerical_state(self, numerical_state):
        numerical_state = np.array(numerical_state)
        
        mean = np.mean(numerical_state, axis=0)
        std = np.std(numerical_state, axis=0)
        
        # Prevent division by zero
        if std == 0:
            std = 1
        
        normalized_state = (numerical_state - mean) / std
        
        return normalized_state

    def get_num_state(self):

        numerical_state = []
        temp = [self.imu_data, self.velocity_data, self.position_data]

        #self.get_logger().info(f'Temp state: {temp}')

        for count, obs in enumerate(temp):
            for data in obs.values():
                for i in data:
                    numerical_state.append(i)

        numerical_state = np.array(numerical_state)
        numerical_state = self.normalize_numerical_state(numerical_state)

        self.get_logger().info(f'Numerical state: {numerical_state}')

        return numerical_state




    def step(self, action, timestep):
        """perform action and read new state"""

        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.linear.y = action[1]
        vel_cmd.linear.z = action[2]

        vel_cmd.angular.x = action[3]
        vel_cmd.angular.y = action[4]
        vel_cmd.angular.z = action[5]

        self.vel_pub.publish(vel_cmd)

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            self.unpause.call_async(Empty.Request())
        except:
            print("/unpause_physics service call failed")

        time.sleep(self.TIME_DELTA)

        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            self.pause.call_async(Empty.Request())
        except (rclpy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")


        self.state['image'] = self.depth_image
        self.state['numerical'] = self.get_num_state()

        dist_to_goal = np.linalg.norm(np.array(self.goal) - np.array(self.position_data['position']))
        reward = self.get_reward(dist_to_goal, action, timestep)
        target = dist_to_goal < 0.5
        done = any([self.collision, target])


        return self.state, reward, done, target


    """
    def get_obstacles(self):
        cylinder_models = ["unit_cylinder", "unit_cylinder_clone"]
        num_clones = 15 

        for i in range(num_clones + 1):
            cylinder_models.append(f"unit_cylinder_clone_{i}")

        for model in cylinder_models:
            result = subprocess.run(['gz', 'model', '-m', model, '-p'], capture_output=True, text=True)
            p = result.stdout
            pose = p.split()
            pose = [float(i) for i in pose][:2]

            self.obstacles.append(pose)

        return self.obstacles
    """

    def wait(self, t):

        count = 0

        while count < t:
            count += 1

            self.rate.sleep()
        


    def reset(self):
        #next: randomize obstacle placement

        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            self.pause.call_async(Empty.Request())
        except:
            print("/gazebo/pause_physics service call failed")

        time.sleep(3)

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            self.unpause.call_async(Empty.Request())
        except:
            print("/gazebo/unpause_physics service call failed")

        subprocess.run(['gz', 'world', '-o'], capture_output=True, text=True)

        land_msg = std.Empty()

        self.land.publish(land_msg)
        self.get_logger().info(f'Landing')
        time.sleep(2)

        takeoff_msg = std.Empty()

        self.takeoff.publish(takeoff_msg)
        self.get_logger().info(f'Taking off')
        time.sleep(4)

        # vel_cmd = Twist()
        # vel_cmd.linear.x = 0.0
        # vel_cmd.linear.y = 0.0
        # vel_cmd.linear.z = 0.0

        # vel_cmd.angular.x = 0.0
        # vel_cmd.angular.y = 0.0
        # vel_cmd.angular.z = 0.0

        # self.vel_pub.publish(vel_cmd)

        self.set_goal()
        self.get_obs()
        self.state['image'] = self.depth_image    
        self.state['numerical'] = self.get_num_state()

        return self.state


    def reset2(self):

        # Resets the state of the environment and returns an initial observation.
        while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')

        try:
            self.reset_proxy.call_async(Empty.Request())
        except rclpy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state

        x = 0
        y = 0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = True#check_pos(x, y)
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        object_state.pose.position.z = 3.0
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        # set a random goal in empty space in environment
        self.set_goal()
        # randomly scatter boxes in the environment
        #self.random_box()
        #self.publish_markers([0.0, 0.0])

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            self.unpause.call_async(Empty.Request())
        except:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(self.TIME_DELTA)

        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            self.pause.call_async(Empty.Request())
        except:
            print("/gazebo/pause_physics service call failed")


        self.get_obs()
        self.state['image'] = self.depth_image    
        self.state['numerical'] = self.get_num_state()


        return self.state



    """
    def reset_2(self):
        self.obstacles = []
        self.reset_world()
        self.set_goal()

        takeoff_msg = std.Empty()

        time.sleep(1)
        self.takeoff.publish(takeoff_msg)
        self.get_logger().info(f'Taking off')
        time.sleep(4)


        self.state['image'] = self.depth_image
        self.state['numerical'] = self.get_num_state()

        return self.state


    def reset_world(self):
        reset_request = Empty.Request()

        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')

        future = self.reset_client.call_async(reset_request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info('Gazebo world reset successfully.')
        else:
            self.get_logger().error('Failed to reset Gazebo world.')

        self.set_state.publish(self.set_self_state)

    """




    def delete_model(self, model_name):
        while not self.delete_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('delete : service not available, waiting again...')

        delete_request = DeleteEntity.Request()
        delete_request.name = model_name

        future = self.delete_client.call_async(delete_request)
        rclpy.spin_until_future_complete(self, future)




    def spawn_cylinder(self, model_name, x, y, height):
        #Cylinder SDF
        with open("/usr/share/sdformat9/1.7/cylinder_shape.sdf", "r") as f:
            model_xml = f.read()

        model_xml = model_xml.replace("<length>2.0</length>", f"<length>{height}</length>")

        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('spawn : service not available, waiting again...')

        spawn_request = SpawnEntity.Request()
        spawn_request.name = model_name
        spawn_request.xml = model_xml
        spawn_request.reference_frame = 'world'

        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.0  # ground level
        pose.orientation.w = 1.0  # neutral orientation
        spawn_request.initial_pose = pose

        future = self.spawn_client.call_async(spawn_request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f'Spawned cylinder "{model_name}" at ({x}, {y}).')
        else:
            self.get_logger().error(f'Failed to spawn cylinder "{model_name}".')




    def build_world(self):
        CYLINDER_COUNT = 20
        for i in range(CYLINDER_COUNT):
            x = random.uniform(self.bounds[0][0], self.bounds[0][1])
            y = random.uniform(self.bounds[1][0], self.bounds[1][1])
            model_name = f"cylinder_{i}"

            self.delete_model(model_name) # Delete the model if it exists
            self.spawn_cylinder(model_name, x, y, self.max_height)
            self.obstacles.append([x, y])



    def set_goal(self):
        goal_ok = False

        while not goal_ok:
            goal = [
            float(random.randint(self.bounds[0][0], self.bounds[0][1])),
            float(random.randint(self.bounds[1][0], self.bounds[1][1])), 
            float(random.randint(0, self.max_height-1.0))
            ]

            if goal not in self.obstacles:
                self.goal = goal
                goal_ok = True

        return goal_ok






    def get_reward(self, dist_to_goal, action, timestep):

        reward = 0.0
        progress_reward_scale = 6.0
        speed_reward_scale = 0.05
        timestep_scale = 0.5
        proximity_penalty_scale = -1.5
        smooth_penalty_scale = -0.01
        safe_distance_threshold = 1.5 
        min_distance = 2.0



        if dist_to_goal < 1.0:
            reward = 100.0
        elif self.collision or (self.position_data['position'][2] > self.max_height):
            reward = -100.0
        else:
            progress_reward = progress_reward_scale * (1.0 / (dist_to_goal + 1e-6)) 
            speed_reward = speed_reward_scale * np.linalg.norm([action[0], action[1], action[2]])
            smooth_penalty = smooth_penalty_scale * np.linalg.norm(self.imu_data['linear_acceleration'])**2

            if min_distance < safe_distance_threshold:
                proximity_penalty = proximity_penalty_scale * (safe_distance_threshold - min_distance)
            else:
                proximity_penalty = 0.0

            if self.position_data['position'][2] > self.max_height - 1.0 or self.position_data['position'][2] < 1.0:
                height_penalty = -10
            else:
                height_penalty = 0

            reward = progress_reward# + speed_reward# + height_penalty + proximity_penalty + smooth_penalty

        return reward





class DepthSubscriber(Node):

    def __init__(self):
        super().__init__('depth_subscriber')
        self.subscription = self.create_subscription(Image,'/simple_drone/depth/depth/image_raw', self.depth_image_callback, 10)
        self.bridge = CvBridge()  
        self.data = None

    def preprocess(self, img, max_dist=70.0):
        self.data = cv2.resize(img, (160, 90)) #original shape: (360, 640)

        #self.data = self.data / max_dist
        #self.data = np.where(np.isinf(self.data), 1.0, self.data)

        self.data = np.copy(self.data)
        finite_mask = np.isfinite(self.data)

        self.data[np.isinf(self.data)] = 1.0
        self.data[finite_mask] = self.data[finite_mask] / max_dist


        return self.data

    def depth_image_callback(self, msg):
        global depth
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth = self.preprocess(cv_image)

            #self.get_logger().info(f"Shape: {self.data.shape} and dtype: {self.data.dtype}")
            #self.get_logger().info(f'{depth}')

            cv2.imshow("depth", depth)
            cv2.waitKey(3)

        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")







class ImuSubscriber(Node):
    
    def __init__(self):
        super().__init__('imu_subscriber')
        self.subscription = self.create_subscription(Imu, '/simple_drone/imu/out', self.imu_callback, 10)
        self.data = {
            'orientation': [0.0, 0.0, 0.0, 0.0],
            'angular_velocity': [0.0, 0.0, 0.0],
            'linear_acceleration': [0.0, 0.0, 0.0]
        }


    def imu_callback(self, msg):
        global imu
        self.data = {
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        }
        imu = self.data

        #self.get_logger().info(f"IMU data: Orientation: {self.data['orientation']}, "
        #               f"Angular Velocity: {self.data['angular_velocity']}, "
        #               f"Linear Acceleration: {self.data['linear_acceleration']}")



class VelocitySubscriber(Node):
    
    def __init__(self):
        super().__init__('velocity_subscriber')
        self.subscription = self.create_subscription(Twist, '/simple_drone/gt_vel', self.velocity_callback, 1)
        self.data = {
            'linear_velocity': [0.0, 0.0, 0.0],
            'angular_velocity': [0.0, 0.0, 0.0]
        }


    def velocity_callback(self, msg):
        global velocity
        self.data = {
            'linear_velocity': [msg.linear.x, msg.linear.y, msg.linear.z],
            'angular_velocity': [msg.angular.x, msg.angular.y, msg.angular.z]
        }
        velocity = self.data

        #self.get_logger().info(f"Velocity data: Linear: {self.data['linear_velocity']}, "
        #                       f"Angular: {self.data['angular_velocity']}")


class CmdVelListener(Node):

    def __init__(self):
        super().__init__('cmd_vel_listener')

        self.subscription = self.create_subscription(
            Twist,
            '/simple_drone/cmd_vel',
            self.cmd_vel_callback,
            1)

    def cmd_vel_callback(self, msg):
        self.get_logger().info(f'Received cmd_vel: linear={msg.linear.x}, angular={msg.angular.z}')


class PoseSubscriber(Node):
    
    def __init__(self):
        super().__init__('pose_subscriber')
        self.subscription = self.create_subscription(Pose, '/simple_drone/gt_pose', self.pose_callback, 10)
        self.data = {
            'position': [0.0, 0.0, 0.0]
        }


    def pose_callback(self, msg):
        global pose
        position = msg.position
        orientation = msg.orientation
        self.data = {
            'position': [position.x, position.y, position.z]
        }
        pose = self.data
        
        #self.get_logger().info(f"Position data: Position: {self.data['position']}, "
        #                       f"Orientation: {self.data['orientation']}")



class CollisionSubscriber(Node):
    
    def __init__(self):
        super().__init__('collision_subscriber')
        self.subscription = self.create_subscription(ContactsState, '/simple_drone/bumper_states', self.collision_callback, 10)
        self.data = None


    def collision_callback(self, msg):
        global collision
        if msg.states:
            self.data = True
            for state in msg.states:
                pass#self.get_logger().info(f'Collision between {state.collision1_name} and {state.collision2_name}')
        else:
            self.data = False

        collision = self.data







def evaluate(network, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        env.get_logger().info(f"evaluating episode {_}")
        count = 0
        state = env.reset()
        done = False
        while not done and count < 501:
            action = network.get_action(np.array(state))
            env.get_logger().info(f"action : {action}")
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done, _ = env.step(a_in)
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    env.get_logger().info("..............................................")
    env.get_logger().info(
        "Average Reward over %i Evaluation Episodes, Epoch %i: avg_reward %f, avg_col %f"
        % (eval_episodes, epoch, avg_reward, avg_col)
    )
    env.get_logger().info("..............................................")
    return avg_reward




def main():

    rclpy.init(args=None)

    timestep = 0
    epoch = 0
    timesteps_since_eval = 0
    episode_num = 0
    max_ep = 200
    eval_freq = 5000
    max_timestep = 5000000
    done = True
    numerical_state_dim = 19
    action_dim = 6
    max_action = 1
    depth_img_shape = (90, 160)
    evaluations = []
    
    expl_noise = 0.1  # initial exploration noise
    expl_min = 0.01  # minimum exploration noise
    expl_decay_steps = 100000 

    depth_sub = DepthSubscriber()
    imu_sub = ImuSubscriber()
    velocity_sub = VelocitySubscriber()
    pose_sub = PoseSubscriber()
    collision_sub = CollisionSubscriber()
    cmd_vel_sub = CmdVelListener()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(depth_sub)
    executor.add_node(imu_sub)
    executor.add_node(velocity_sub)
    executor.add_node(pose_sub)
    executor.add_node(collision_sub)
    #executor.add_node(cmd_vel_sub)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    network = td3(depth_img_shape, numerical_state_dim, action_dim, max_action)
    env = GazeboEnv()
    replay_buffer = ReplayBuffer(max_size=1e6, image_shape=depth_img_shape, numerical_dim=numerical_state_dim, action_dim=action_dim)

    try:
        while rclpy.ok():
            if timestep < max_timestep:
                if done:
                    #env.wait(3)
                    env.get_logger().info(f'Episode {episode_num} done. Timestep: {timestep}')

                    if timestep != 0:
                        env.get_logger().info(f'\n\nReplay buffer: {replay_buffer.reward_buffer}\n\nTraining at timestep {timestep}')
                        network.train(replay_buffer, episode_timesteps)

                    if timesteps_since_eval >= eval_freq:
                        env.get_logger().info(f'Validating at epoch {epoch}')
                        timesteps_since_eval %= eval_freq
                        evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=10)) 

                        #network.save(filename=f'td3_model_{epoch}', directory='./dev_ws/src/drones_ROS2/controller/models')
                        #np.save('./dev_ws/src/drones_ROS2/controller/results/evaluation.npy', evaluations)
                        epoch += 1


                    state = env.reset()
                    env.goal_viz.destroy_node()
                    env.goal_viz = Balloon(env.goal)
                    done = False
                    episode_reward = 0
                    episode_timesteps = 0
                    episode_num += 1
                env.get_obs()
                env.goal_viz.publish_marker()


                if expl_noise > expl_min:
                    expl_noise -= ((expl_noise - expl_min) / expl_decay_steps)


                #action = network.select_action(tf.expand_dims(state['image'], -1), state['numerical'])


                action = network.actor( (np.expand_dims(np.asarray(state['image']).astype(np.float32), axis=0),
                                        np.expand_dims(np.asarray(state['numerical']).astype(np.float32), axis=0)) )[0].numpy()

                env.get_logger().info(f'\nAction: {action}, Collision: {env.collision}')
                action = network.add_noise(action, noise_std=expl_noise)

                next_state, reward, done, target = env.step(action, episode_timesteps)

                if episode_timesteps + 1 == max_ep:
                    done = True


                done_bool = float(done)


                env.get_logger().info(f'\nEpsiode reward: {episode_reward}')


                replay_buffer.store(
                    state['image'], state['numerical'], action,
                    next_state['image'], next_state['numerical'], reward, done_bool
                )

                state = next_state
                episode_reward += reward
                episode_timesteps += 1
                timestep += 1
                timesteps_since_eval += 1

    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


def test():
    global depth, imu, velocity, pose, collision
    rclpy.init(args=None)

    depth_sub = DepthSubscriber()
    imu_sub = ImuSubscriber()
    velocity_sub = VelocitySubscriber()
    pose_sub = PoseSubscriber()
    collision_sub = CollisionSubscriber()
    takeoff_msg = std.Empty()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(depth_sub)
    executor.add_node(imu_sub)
    executor.add_node(velocity_sub)
    executor.add_node(pose_sub)
    executor.add_node(collision_sub)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    env = GazeboEnv()

    try:
        while rclpy.ok():
            env.get_obs()

            #env.get_logger().info(f'\n\nDepth: {depth}\n Collision: {collision}')


    except KeyboardInterrupt:
        pass

    rclpy.shutdown()





if __name__ == "__main__":
    main()
    #test()


#manual control: ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args --remap cmd_vel:=/simple_drone/cmd_vel