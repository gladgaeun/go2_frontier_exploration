#!/usr/bin/env python3
# Original Code Copyright (c) 2023, Adrian Sochaniwsky (BSD 3-Clause License)
# Source: https://github.com/adrian-soch/frontier_exploration
# Modified & Ported for Unitree Go2 ROS 2 navigation by Gaeun Bang (2026)

'''
This node is for autonomous exploration. It requests frontier regions from a service and
sends the goal points to the nav2 stack until the the entire environment has been explored.

Reference code: https://automaticaddison.com/how-to-send-goals-to-the-ros-2-navigation-stack-nav2/
'''
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from frontier_interfaces.srv import FrontierGoal
from frontier_exploration.robot_navigator import BasicNavigator, NavigationResult
 
class FrontierExplorer(Node):

    def __init__(self):
        super().__init__('frontier_explorer')
        self.cli = self.create_client(FrontierGoal, 'frontier_pose')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        self.req = FrontierGoal.Request()
        self.navigator = BasicNavigator()

        self.EXPLORATION_TIME_OUT_SEC = Duration(seconds=1200)
        self.NAV_TO_GOAL_TIMEOUT_SEC = 75
        self.DIST_THRESH_FOR_HEADING_CALC = 0.25

        self.goal_pose = PoseStamped()
        self.failed_goals = []

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        import time
        self.get_logger().info('Waiting for map and frontiers to be ready...')
        time.sleep(2.0)  # Allow time for the C++ detector to process the initial map

        self.start_time = self.get_clock().now()
        self.get_logger().info('Starting frontier exploration...')
        self.explore()

    def explore(self):
        import time

        while (self.get_clock().now() - self.start_time) < self.EXPLORATION_TIME_OUT_SEC:
            
            time.sleep(0.5)

            # Get a frontier we can drive to
            self.goal_pose = self.get_reachable_goal()

            # If we cant find a path to any frontiers
            if self.goal_pose is None:
                self.get_logger().error('No reachable frontiers!')
                exit(-1)
            elif self.goal_pose == "Done":
                self.get_logger().info('Exploration complete! No frontiers detected.')
                exit(0)
            
            # Go to the goal pose
            self.navigator.goToPose(self.goal_pose)

            # Allow time for the Nav2 action server to accept and transition to execution state
            time.sleep(0.5)
            
            # Keep doing stuff as long as the robot is moving towards the goal
            i = 0
            while not self.navigator.isNavComplete():

                i = i + 1
                feedback = self.navigator.getFeedback()
                if feedback and i % 10 == 0:
                    self.get_logger().info('Distance remaining: ' + '{:.2f}'.format(
                        feedback.distance_remaining) + ' meters.')
                
                    # Set goal pose heading to robots current heading once it is close]
                    # This avoids unecessary rotation once reaching the goal position
                    # if feedback.distance_remaining <= self.DIST_THRESH_FOR_HEADING_CALC:
                    #     self.get_logger().info('Setting new heading')
                    #     self.set_goal_heading()
                    #     self.navigator.goToPose(self.goal_pose)
            
                # Cancel the goal if robot takes too long
                if feedback and hasattr(feedback, 'navigation_time'):
                    if Duration.from_msg(feedback.navigation_time) > Duration(seconds=self.NAV_TO_GOAL_TIMEOUT_SEC):
                        self.get_logger().warn('Navigation timed out. Canceling goal...')
                        self.navigator.cancelNav()
                                
                time.sleep(0.1)
            
            # Print result when nav completes
            result = self.navigator.getResult()
            self.log_nav_status(result)

            # Allow time for SLAM to expand the map with new LiDAR scans upon goal arrival
            if result == NavigationResult.SUCCEEDED:
                time.sleep(1.0)

            # Add to blacklist on failure/cancellation to prevent goal ping-pong loops
            if result == NavigationResult.FAILED or result == NavigationResult.CANCELED:
                self.failed_goals.append((self.goal_pose.pose.position.x, self.goal_pose.pose.position.y))
                if len(self.failed_goals) > 30:
                    self.failed_goals.pop(0) # Keep maximum 30 entries

    def get_reachable_goal(self):
        rank = 0
        reachable = False
        import math
        while not reachable:
            goal = self.send_request(rank)
            if goal is None:
                return "Done"

            # Handle dummy/invalid goals by retrying or completing exploration
            if self.is_empty_goal(goal):
                if rank == 0:
                    for _ in range(5):
                        import time
                        time.sleep(1.0)
                        goal = self.send_request(0)
                        if not self.is_empty_goal(goal):
                            break
                    if self.is_empty_goal(goal):
                        return "Done"
                else:
                    return "Done"

            self.goal_pose = goal
            self.goal_pose.header.frame_id = 'map'
            self.goal_pose.header.stamp = self.get_clock().now().to_msg()

            # Check if the candidate goal is near a previously blacklisted position (within 0.5m)
            is_blacklisted = False
            for fg in self.failed_goals:
                if math.hypot(goal.pose.position.x - fg[0], goal.pose.position.y - fg[1]) < 0.5:
                    is_blacklisted = True
                    break
            if is_blacklisted:
                rank += 1
                if rank > 20:
                    return "Done"
                continue

            # Retrieve the robot's current pose
            initial_pose = self.get_current_pose()
            if initial_pose is not None:
                # Calculate Euclidean distance between the robot and candidate goal
                dx = goal.pose.position.x - initial_pose.pose.position.x
                dy = goal.pose.position.y - initial_pose.pose.position.y
                dist = math.hypot(dx, dy)

                if dist < 0.2:
                    rank += 1
                    if rank > 20:
                        return "Done"
                    continue

                # Validate path reachability via Nav2 planner
                path = self.navigator.getPath(initial_pose, self.goal_pose)
                if path is not None:
                    return goal
            else:
                return goal

            if rank > 20:
                return None
            rank += 1

    @staticmethod
    def is_empty_goal(goal):
        # The detector answers with an empty frame_id when there is no frontier for this rank
        return goal is None or goal.header.frame_id == ''

    def send_request(self, rank):
        self.req.goal_rank = rank
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        res = self.future.result()
        return res.goal_pose

    def get_current_pose(self) -> PoseStamped:
        try:
            t = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rclpy.time.Time(), 
                Duration(seconds=1.0)
            )
        except TransformException as ex:
            self.get_logger().warn(f'Current pose unavailable: {ex}')
            return None
            
        p = PoseStamped()
        p.header.stamp = t.header.stamp
        p.header.frame_id = 'map'
        p.pose.position.x = t.transform.translation.x
        p.pose.position.y = t.transform.translation.y
        p.pose.position.z = 0.0
        p.pose.orientation = t.transform.rotation
        return p
    
    def set_goal_heading(self):
        curr_pose = self.get_current_pose()

        if curr_pose is None:
            return
        
        # Set goal orientation to match current heading
        self.goal_pose.pose.orientation.x = curr_pose.pose.orientation.x
        self.goal_pose.pose.orientation.y = curr_pose.pose.orientation.y
        self.goal_pose.pose.orientation.z = curr_pose.pose.orientation.z
        self.goal_pose.pose.orientation.w = curr_pose.pose.orientation.w
    
    def log_nav_status(self, result):
        if result == NavigationResult.SUCCEEDED:
            self.get_logger().info('Goal succeeded!')
        elif result == NavigationResult.CANCELED:
            self.get_logger().info('Goal was canceled!')
        elif result == NavigationResult.FAILED:
            self.get_logger().info('Goal failed!')
        else:
            self.get_logger().error('Goal has an invalid return status!')

def main(args=None):
    rclpy.init(args=args)
    frontier_explorer = FrontierExplorer()   
    frontier_explorer.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()