#!/usr/bin/env python3
"""Exploration executor of trav_exploration: frontier_goal_node -> Nav2.

Asks frontier_goal_node (frontier_pose service, FrontierGoal) for the rank-th goal, sends it to
Nav2 (navigate_to_pose, whose global path is voxel_mapper_node's 3-D support graph planner) and
waits for the result, until frontier_goal_node has no goal left.

It replaces the 2-D Go2 exploration loop (frontier_exploration_node.py, left unchanged for the 2-D
stack), whose blacklist of failed or canceled goals by (x, y) within 0.5 m does not fit the 3-D
map of the paper:
  - it has no z: a goal canceled on one floor turned down a different frontier cluster on the floor
    above (HM3D 00891: (-1.68, 2.03) at z 2.95 refused within 3 ms for (-1.28, 2.13) at z -0.10, the
    new floor then had "no frontier left");
  - frontier_goal_node's own replan (the frontier gone on the way) cancels goals, which is not a
    failure;
  - a failure is fed back into the map instead: execute-review (eq. 17, collision_guard_node /
    voxel_mapper_node) marks the blocked voxels, and frontier_goal_node keeps a frontier cluster it
    did not reach or that stayed after its visit dormant until its voxels change.
So nothing is remembered here: every goal is frontier_goal_node's choice on the current map.

Kept from the 2-D loop: a goal within goal_min_distance of the robot or without a path is turned
down (the next rank is requested, which frontier_goal_node reviews), the Nav2 timeout per goal,
the overall time-out, and the end of the loop ("Exploration complete! No frontiers detected.",
exit 0) that the launch file and the episode runner react to.
"""
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from frontier_exploration.robot_navigator import BasicNavigator, NavigationResult
from frontier_interfaces.srv import FrontierGoal
from trav_exploration.recorder import Recorder


class ExploreExecutor(Node):

    def __init__(self):
        super().__init__('explore_executor')
        p = self.declare_parameter
        self.map_frame = p('map_frame', 'map').value
        self.base_frame = p('base_frame', 'base_link').value
        self.goal_min_distance = float(p('goal_min_distance', 0.2).value)     # m: "at it"
        self.nav_timeout = float(p('nav_timeout', 75.0).value)                # s per goal
        self.exploration_timeout = float(p('exploration_timeout', 1200.0).value)
        self.max_rank = int(p('max_rank', 20).value)
        self.empty_retries = int(p('empty_retries', 5).value)                 # rank 0 empty: wait 1 s, ask again

        # record_dir: goal and result events for offline datasets (recorder.py; '' = off)
        self.recorder = Recorder(self, 'explore_executor', str(p('record_dir', '').value))
        self.cli = self.create_client(FrontierGoal, 'frontier_pose')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('frontier_pose not available, waiting again...')
        self.navigator = BasicNavigator()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    # -- frontier_goal_node -----------------------------------------------------------------------

    def _request(self, rank):
        req = FrontierGoal.Request()
        req.goal_rank = rank
        fut = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()
        return None if res is None else res.goal_pose

    @staticmethod
    def _empty(goal):
        # frontier_goal_node answers with an empty frame_id when there is no goal for this rank
        return goal is None or goal.header.frame_id == ''

    def _robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, rclpy.time.Time(),
                                                Duration(seconds=1.0))
        except TransformException as ex:
            self.get_logger().warn(f'robot pose unavailable: {ex}')
            return None
        ps = PoseStamped()
        ps.header.stamp = t.header.stamp
        ps.header.frame_id = self.map_frame
        ps.pose.position.x = t.transform.translation.x
        ps.pose.position.y = t.transform.translation.y
        ps.pose.position.z = t.transform.translation.z
        ps.pose.orientation = t.transform.rotation
        return ps

    def _next_goal(self):
        """The best goal frontier_goal_node offers that is not at the robot and has a path; None
        when there is none (the end of the exploration)."""
        for rank in range(self.max_rank + 1):
            goal = self._request(rank)
            if self._empty(goal) and rank == 0:
                for _ in range(self.empty_retries):      # the map may not be ready yet
                    time.sleep(1.0)
                    goal = self._request(0)
                    if not self._empty(goal):
                        break
            if self._empty(goal):
                return None
            goal.header.frame_id = self.map_frame
            goal.header.stamp = self.get_clock().now().to_msg()
            pose = self._robot_pose()
            if pose is None:
                return goal
            d = ((goal.pose.position.x - pose.pose.position.x) ** 2
                 + (goal.pose.position.y - pose.pose.position.y) ** 2) ** 0.5
            if d < self.goal_min_distance:
                self.get_logger().info(f'goal ({goal.pose.position.x:.2f}, {goal.pose.position.y:.2f}) '
                                       f'at the robot ({d:.2f} m): next rank')
                continue
            if self.navigator.getPath(pose, goal) is not None:
                return goal
            self.get_logger().info(f'no path to ({goal.pose.position.x:.2f}, {goal.pose.position.y:.2f}): next rank')
        return None

    # -- loop ---------------------------------------------------------------------------------------

    def run(self):
        time.sleep(2.0)                                   # the first map and frontiers
        self.get_logger().info('Starting frontier exploration...')
        t0 = time.time()
        while time.time() - t0 < self.exploration_timeout:
            time.sleep(0.5)
            goal = self._next_goal()
            if goal is None:
                self.recorder.record('end', reason='no goal')
                self.get_logger().info('Exploration complete! No frontiers detected.')
                return 0
            self.navigator.goToPose(goal)
            t_goal = time.time()
            self.recorder.record('goal', x=goal.pose.position.x, y=goal.pose.position.y)
            time.sleep(0.5)                               # Nav2 accepts the goal
            i = 0
            timed_out = False
            while not self.navigator.isNavComplete():
                i += 1
                fb = self.navigator.getFeedback()
                if fb and i % 10 == 0:
                    self.get_logger().info(f'Distance remaining: {fb.distance_remaining:.2f} meters.')
                if fb and hasattr(fb, 'navigation_time') and \
                        Duration.from_msg(fb.navigation_time) > Duration(seconds=self.nav_timeout):
                    self.get_logger().warn('Navigation timed out. Canceling goal...')
                    timed_out = True
                    self.navigator.cancelNav()
                time.sleep(0.1)
            result = self.navigator.getResult()
            self.recorder.record('result', status={NavigationResult.SUCCEEDED: 'succeeded',
                                                   NavigationResult.CANCELED: 'canceled',
                                                   NavigationResult.FAILED: 'failed'}.get(result, 'unknown'),
                                 timed_out=bool(timed_out), x=goal.pose.position.x, y=goal.pose.position.y,
                                 wall_s=round(time.time() - t_goal, 2))
            if result == NavigationResult.SUCCEEDED:
                self.get_logger().info('Goal succeeded!')
                time.sleep(1.0)                           # the map grows with the view from the goal
            elif result == NavigationResult.CANCELED:
                self.get_logger().info('Goal was canceled!')
            elif result == NavigationResult.FAILED:
                self.get_logger().info('Goal failed!')
            else:
                self.get_logger().error('Goal has an invalid return status!')
        self.get_logger().warn('exploration time-out')
        return 0


def main(args=None):
    rclpy.init(args=args)
    node = ExploreExecutor()
    code = 0
    try:
        code = node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.navigator.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(code)


if __name__ == '__main__':
    main()
