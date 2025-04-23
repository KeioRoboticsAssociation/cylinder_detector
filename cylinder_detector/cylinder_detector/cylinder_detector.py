#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point  # 標準メッセージ
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import math

class CylinderDetector(Node):
    def __init__(self):
        super().__init__('cylinder_detector')
        # PointCloud2のサブスクライバ
        self.subscription = self.create_subscription(
            PointCloud2,
            'pointcloud',  # 入力トピック名
            self.listener_callback,
            10
        )
        # centerをpublishするためのパブリッシャ (geometry_msgs/Point)
        self.publisher_ = self.create_publisher(Point, '/centers_listener', 10)
        self.get_logger().info('CylinderDetector node started.')
        
        # パラメータ
        self.declare_parameter('ransac_iterations', 20) # RANSACの試行回数
        self.declare_parameter('ransac_threshold', 2.0) # RANSACの誤差許容範囲
        self.declare_parameter('window_width', 50.0) # スライディングウィンドウの幅
        self.declare_parameter('window_step', 5.0) # スライディングウィンドウをどれだけずらすか
        self.declare_parameter('x_min', -100.0) # ウィンドウをスライドさせる範囲のx座標の最小値
        self.declare_parameter('x_max', 100.0) # ウィンドウをスライドさせる範囲のx座標の最大値
        self.declare_parameter('y_min', -110.0) # ウィンドウをスライドさせる範囲のy座標の最小値
        self.declare_parameter('y_max', 110.0) # ウィンドウをスライドさせる範囲のy座標の最大値
        
        self.keepalive_timer = self.create_timer(1.0, lambda: None)

        self.get_logger().info('PointCloudListener node started.')

    def listener_callback(self, msg):
        ransac_iterations = self.get_parameter('ransac_iterations').value
        ransac_threshold = self.get_parameter('ransac_threshold').value
        windowWidth = self.get_parameter('window_width').value
        step = self.get_parameter('window_step').value
        xMin = self.get_parameter('x_min').value
        xMax = self.get_parameter('x_max').value
        yMin = self.get_parameter('y_min').value
        yMax = self.get_parameter('y_max').value
        
        # sensor_msgs/PointCloud2 から (x, y, z) の点群を取得（NaNは除外）
        points = pc2.read_points_list(msg, field_names=("x", "y", "z"), skip_nans=True)
        num_points = len(points)
        self.get_logger().info(f"Received point cloud with {num_points} points.")

        if num_points == 0:
            return

        # NumPy配列に変換 (shape: (N, 3))
        points_np = np.vstack(points)

        # --- 1. 条件に合致する窓ごとに平均点（x, z）を算出
        centers = []  # 各窓で算出された平均 (x, z) を格納
        currentX = xMin
        while currentX <= xMax - windowWidth:
            xWindowMin = currentX
            xWindowMax = currentX + windowWidth
            # 条件: xがウィンドウ内、かつyが[-110, 110]の点を抽出
            mask = ((points_np[:, 0] >= xWindowMin) & (points_np[:, 0] <= xWindowMax) &
                    (points_np[:, 1] > yMin) & (points_np[:, 1] < yMax))
            selected = points_np[mask]
            if selected.shape[0] > 0:
                avg_x = np.mean(selected[:, 0])
                avg_z = np.mean(selected[:, 2])
                centers.append((avg_x, avg_z))
            currentX += step
            
        if len(centers) < 3:
            self.get_logger().info("Not enough centers to compute circle. Exiting.")
            return

        # --- 2. RANSACアルゴリズムを用いて円を複数回検出し、最良の結果を採用
        best_inliers = 0
        best_diameter = 0
        best_center = None
        
        for _ in range(ransac_iterations):
            # 3点をランダムに選択
            indices = np.random.choice(len(centers), 3, replace=False)
            p1 = centers[indices[0]]
            p2 = centers[indices[1]]
            p3 = centers[indices[2]]
            
            # 3点から円を計算
            success, computedDiameter, circle_center = self.compute_circle_from_three_points(p1, p2, p3)
            if not success:
                continue
            # 他の点が円内に入るか判定
            inliers = 0
            for center in centers:
                distance = math.sqrt((center[0] - circle_center[0]) ** 2 + (center[1] - circle_center[1]) ** 2)
                if abs(distance - computedDiameter / 2) < ransac_threshold:
                    inliers += 1
            if inliers > best_inliers:
                best_inliers = inliers
                best_diameter = computedDiameter
                best_center = circle_center
                
        if best_center is None:
            self.get_logger().info("Failed to compute circle from any set of three points.")
            return
        self.get_logger().info(f"Generated {len(centers)} center candidates: {centers}")
        self.get_logger().info(f"Computed diameter with {best_inliers} inliers: {best_diameter:.2f} mm")
        # テストの時best_inlinersの数を見て、それくらいの数をliners > ?? (line98)に代入

        success, computedDiameter, circle_center = self.compute_circle_from_three_points(p1, p2, p3)
        if not success:
            self.get_logger().info("Failed to compute circle from three points (points may be collinear).")
            return
        self.get_logger().info(f"Computed diameter from three points: {computedDiameter:.2f} mm")

        # --- 3. geometry_msgs/Pointメッセージとしてcenterをパブリッシュ
        center_msg = Point()
        center_msg.x = best_center[0]
        center_msg.y = 0.0  # Y座標は計算していないため0に設定
        center_msg.z = best_center[1]
        self.publisher_.publish(center_msg)
        self.get_logger().info(f"Published center: ({center_msg.x:.2f}, {center_msg.y:.2f}, {center_msg.z:.2f})")

    def compute_circle_from_three_points(self, p1, p2, p3):
        """
        3点 (p1, p2, p3) から円の中心と直径を計算する関数
        p1, p2, p3: タプル (x, z)
        戻り値: (成功フラグ, 直径, (center_x, center_z))
        """
        x1, z1 = p1
        x2, z2 = p2
        x3, z3 = p3
        # zをyとみなして計算
        y1, y2, y3 = z1, z2, z3

        # 3点が共線か判定するための行列式 (三角形面積の2倍)
        det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
        if abs(det) < 1e-6:
            return False, None, None

        # 中心公式
        A = x1 * x1 + y1 * y1
        B = x2 * x2 + y2 * y2
        C = x3 * x3 + y3 * y3
        D = 2.0 * det
        center_x = (A * (y2 - y3) + B * (y3 - y1) + C * (y1 - y2)) / D
        center_z = (A * (x3 - x2) + B * (x1 - x3) + C * (x2 - x1)) / D
        # 円の半径→直径の計算
        radius = math.sqrt((x1 - center_x) ** 2 + (y1 - center_z) ** 2)
        diameter = 2.0 * radius
        return True, diameter, (center_x, center_z)


def main(args=None):
    print('Hi from cylinder_detector.')
    rclpy.init(args=args)
    node = CylinderDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main(args=None)
