"""
ManipulaTHOR Agent Navigation Script
목표 좌표를 입력받아 ManipulaTHOR agent를 해당 위치까지 이동시킵니다.
"""

import argparse
import math
import time
from typing import Dict, Tuple, Optional
import numpy as np

try:
    from ai2thor.controller import Controller
except ImportError:
    print("⚠️  ai2thor not installed. Please install: pip install ai2thor")
    exit(1)


def distance_3d(pos1: Dict[str, float], pos2: Dict[str, float]) -> float:
    """3D 공간에서 두 점 사이의 유클리드 거리 계산"""
    dx = pos2['x'] - pos1['x']
    dy = pos2['y'] - pos1['y']
    dz = pos2['z'] - pos1['z']
    return math.sqrt(dx**2 + dy**2 + dz**2)


def calculate_angle_to_target(current_pos: Dict[str, float], 
                               current_rot: float,
                               target_pos: Dict[str, float]) -> float:
    """
    현재 위치와 회전 각도에서 목표 위치까지의 회전 각도 계산
    
    Args:
        current_pos: 현재 위치 {"x": float, "y": float, "z": float}
        current_rot: 현재 회전 각도 (도)
        target_pos: 목표 위치 {"x": float, "y": float, "z": float}
    
    Returns:
        목표를 향하기 위해 회전해야 할 각도 (도, -180 ~ 180)
    """
    # x, z 평면에서의 방향 벡터 계산
    dx = target_pos['x'] - current_pos['x']
    dz = target_pos['z'] - current_pos['z']
    
    # 목표 방향의 각도 계산 (도)
    target_angle = math.degrees(math.atan2(dx, dz))
    target_angle = (target_angle + 360) % 360
    
    # 현재 회전 각도와의 차이 계산
    angle_diff = target_angle - current_rot
    
    # -180 ~ 180 범위로 정규화
    if angle_diff > 180:
        angle_diff -= 360
    elif angle_diff < -180:
        angle_diff += 360
    
    return angle_diff


def find_closest_reachable_position(controller: Controller, 
                                    target_pos: Dict[str, float]) -> Optional[Dict[str, float]]:
    """
    목표 위치에 가장 가까운 도달 가능한 위치 찾기
    
    Args:
        controller: AI2-THOR Controller
        target_pos: 목표 위치 {"x": float, "y": float, "z": float}
    
    Returns:
        가장 가까운 도달 가능한 위치 또는 None
    """
    event = controller.step(action="GetReachablePositions")
    reachable_positions = event.metadata.get("actionReturn", [])
    
    if not reachable_positions:
        return None
    
    # 가장 가까운 위치 찾기
    closest_pos = None
    min_distance = float('inf')
    
    for pos in reachable_positions:
        dist = distance_3d(pos, target_pos)
        if dist < min_distance:
            min_distance = dist
            closest_pos = pos
    
    return closest_pos


def find_next_waypoint(controller: Controller,
                       current_pos: Dict[str, float],
                       target_pos: Dict[str, float],
                       max_search_radius: float = 5.0,
                       min_distance: float = 0.3) -> Optional[Dict[str, float]]:
    """
    GetReachablePositions를 사용하여 현재 위치에서 목표에 더 가까워지는 다음 waypoint 찾기
    
    Args:
        controller: AI2-THOR Controller
        current_pos: 현재 위치
        target_pos: 목표 위치
        max_search_radius: 검색할 최대 반경 (미터)
        min_distance: 현재 위치에서 최소 거리 (미터, 너무 가까운 위치 제외)
    
    Returns:
        다음 waypoint 위치 또는 None
    """
    # 모든 도달 가능한 위치 가져오기
    event = controller.step(action="GetReachablePositions")
    reachable_positions = event.metadata.get("actionReturn", [])
    
    if not reachable_positions:
        return None
    
    # 현재 위치에서 목표까지의 거리
    current_to_target = distance_3d(current_pos, target_pos)
    
    # 현재 위치에서 가까운 도달 가능한 위치들 중에서
    # 목표에 더 가까운 위치 찾기
    best_waypoint = None
    best_improvement = 0.0  # 거리 개선량
    
    for pos in reachable_positions:
        # 현재 위치에서의 거리
        dist_from_current = distance_3d(current_pos, pos)
        
        # 너무 가깝거나 너무 먼 위치는 제외
        if dist_from_current < min_distance or dist_from_current > max_search_radius:
            continue
        
        # 목표까지의 거리
        dist_to_target = distance_3d(pos, target_pos)
        
        # 거리 개선량 계산
        improvement = current_to_target - dist_to_target
        
        # 더 가까워지는 위치 중에서 가장 큰 개선량을 가진 위치 선택
        if improvement > best_improvement:
            best_improvement = improvement
            best_waypoint = pos
    
    return best_waypoint


def navigate_to_position(controller: Controller,
                        target_pos: Dict[str, float],
                        goal_threshold: float = 0.3,
                        max_iterations: int = 200,
                        move_distance: float = 0.25) -> bool:
    """
    ManipulaTHOR agent를 목표 위치까지 이동 (GetReachablePositions 기반 경로 탐색)
    
    Args:
        controller: AI2-THOR Controller
        target_pos: 목표 위치 {"x": float, "y": float, "z": float}
        goal_threshold: 목표 도달로 간주할 거리 (미터, 기본값: 0.3m)
        max_iterations: 최대 반복 횟수 (기본값: 200)
        move_distance: 한 번에 이동할 거리 (미터, 기본값: 0.25m)
    
    Returns:
        성공 여부
    """
    iteration = 0
    stuck_count = 0
    prev_distance = float('inf')
    consecutive_failures = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # 현재 agent 상태 가져오기
        metadata = controller.last_event.metadata
        agent_pos = metadata["agent"]["position"]
        agent_rot = metadata["agent"]["rotation"]["y"]
        
        # 현재 위치에서 목표까지의 거리 계산
        current_distance = distance_3d(agent_pos, target_pos)
        
        print(f"[Iteration {iteration}] Current: ({agent_pos['x']:.3f}, {agent_pos['y']:.3f}, {agent_pos['z']:.3f}), "
              f"Target: ({target_pos['x']:.3f}, {target_pos['y']:.3f}, {target_pos['z']:.3f}), "
              f"Distance: {current_distance:.3f}m")
        
        # 목표 도달 확인
        if current_distance <= goal_threshold:
            print(f"✓ Reached target position! (distance: {current_distance:.3f}m)")
            return True
        
        # 멈춤 감지 (거리가 거의 변하지 않음)
        if abs(current_distance - prev_distance) < 0.05:
            stuck_count += 1
            if stuck_count >= 3:
                print(f"⚠ Stuck detected. Finding alternative path using GetReachablePositions...")
                # GetReachablePositions를 사용하여 다음 waypoint 찾기
                next_waypoint = find_next_waypoint(controller, agent_pos, target_pos, min_distance=0.5)
                
                if next_waypoint:
                    waypoint_dist = distance_3d(agent_pos, next_waypoint)
                    print(f"  → Found waypoint: ({next_waypoint['x']:.3f}, {next_waypoint['y']:.3f}, {next_waypoint['z']:.3f}), distance: {waypoint_dist:.3f}m")
                    # Waypoint 방향으로 회전 및 이동
                    angle_to_waypoint = calculate_angle_to_target(agent_pos, agent_rot, next_waypoint)
                    
                    if abs(angle_to_waypoint) > 5.0:
                        rotation_degrees = max(-90.0, min(90.0, angle_to_waypoint))
                        print(f"  → Rotating {rotation_degrees:.1f}° towards waypoint")
                        event = controller.step({
                            "action": "RotateAgent",
                            "degrees": rotation_degrees,
                            "returnToStart": True,
                            "speed": 1.0,
                            "fixedDeltaTime": 0.02
                        })
                        if not event.metadata.get('lastActionSuccess', False):
                            print(f"  ⚠ Rotation failed: {event.metadata.get('errorMessage', 'Unknown error')}")
                        time.sleep(0.1)
                    
                    # Waypoint 방향으로 이동
                    print(f"  → Moving towards waypoint")
                    event = controller.step({
                        "action": "MoveAgent",
                        "ahead": move_distance,
                        "right": 0.0,
                        "returnToStart": True,
                        "speed": 1.0,
                        "fixedDeltaTime": 0.02
                    })
                    if not event.metadata.get('lastActionSuccess', False):
                        print(f"  ⚠ Movement failed: {event.metadata.get('errorMessage', 'Unknown error')}")
                    stuck_count = 0
                    consecutive_failures = 0
                    time.sleep(0.2)
                    continue
                else:
                    print(f"  ⚠ No valid waypoint found")
                    consecutive_failures += 1
                    if consecutive_failures >= 5:
                        print(f"✗ Cannot find path to target")
                        return False
        else:
            stuck_count = 0
            consecutive_failures = 0
        
        prev_distance = current_distance
        
        # 목표 방향 계산
        angle_to_target = calculate_angle_to_target(agent_pos, agent_rot, target_pos)
        
        # 목표 방향으로 회전 (5도 이상 차이면)
        if abs(angle_to_target) > 5.0:
            # 회전 각도 제한 (최대 90도씩)
            rotation_degrees = max(-90.0, min(90.0, angle_to_target))
            
            print(f"  → Rotating {rotation_degrees:.1f}° towards target")
            event = controller.step({
                "action": "RotateAgent",
                "degrees": rotation_degrees,
                "returnToStart": True,
                "speed": 1.0,
                "fixedDeltaTime": 0.02
            })
            
            if not event.metadata.get('lastActionSuccess', False):
                print(f"  ⚠ Rotation failed: {event.metadata.get('errorMessage', 'Unknown error')}")
            
            time.sleep(0.1)
            continue
        
        # 목표 방향으로 이동
        print(f"  → Moving forward {move_distance}m")
        event = controller.step({
            "action": "MoveAgent",
            "ahead": move_distance,
            "right": 0.0,
            "returnToStart": True,
            "speed": 1.0,
            "fixedDeltaTime": 0.02
        })
        
        if not event.metadata.get('lastActionSuccess', False):
            error_msg = event.metadata.get('errorMessage', 'Unknown error')
            print(f"  ⚠ Movement failed: {error_msg}")
            # 이동 실패 시 GetReachablePositions로 대체 경로 찾기
            next_waypoint = find_next_waypoint(controller, agent_pos, target_pos, min_distance=0.5)
            if next_waypoint:
                waypoint_dist = distance_3d(agent_pos, next_waypoint)
                print(f"  → Trying alternative waypoint: ({next_waypoint['x']:.3f}, {next_waypoint['y']:.3f}, {next_waypoint['z']:.3f}), distance: {waypoint_dist:.3f}m")
                angle_to_waypoint = calculate_angle_to_target(agent_pos, agent_rot, next_waypoint)
                if abs(angle_to_waypoint) > 5.0:
                    rotation_degrees = max(-90.0, min(90.0, angle_to_waypoint))
                    print(f"  → Rotating {rotation_degrees:.1f}° towards waypoint")
                    event = controller.step({
                        "action": "RotateAgent",
                        "degrees": rotation_degrees,
                        "returnToStart": True,
                        "speed": 1.0,
                        "fixedDeltaTime": 0.02
                    })
                    if not event.metadata.get('lastActionSuccess', False):
                        print(f"  ⚠ Rotation failed: {event.metadata.get('errorMessage', 'Unknown error')}")
                    time.sleep(0.1)
        
        time.sleep(0.1)
    
    print(f"✗ Failed to reach target after {max_iterations} iterations")
    return False


def main():
    parser = argparse.ArgumentParser(description="ManipulaTHOR Agent Navigation")
    parser.add_argument("--scene", type=str, default="FloorPlan1_physics", 
                       help="Scene name (default: FloorPlan1_physics)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--goal-threshold", type=float, default=0.3,
                       help="Goal distance threshold in meters (default: 0.3)")
    parser.add_argument("--max-iterations", type=int, default=200,
                       help="Maximum number of iterations (default: 200)")
    
    args = parser.parse_args()
    
    # 좌표 입력 받기
    while True:
        try:
            coord_input = input("좌표를 입력하세요 (x, y, z 형식, 예: -1.0, 0.9, 1.0): ").strip()
            coords = [c.strip() for c in coord_input.split(",")]
            if len(coords) != 3:
                print("⚠ 좌표는 3개의 값이 필요합니다 (x, y, z)")
                continue
            target_position = {
                "x": float(coords[0]),
                "y": float(coords[1]),
                "z": float(coords[2])
            }
            break
        except ValueError:
            print("⚠ 잘못된 입력입니다. 숫자로 입력해주세요 (예: -1.0, 0.9, 1.0)")
        except Exception as e:
            print(f"⚠ 입력 오류: {e}")
    
    print("=" * 80)
    print("ManipulaTHOR Navigation")
    print("=" * 80)
    print(f"Scene: {args.scene}")
    print(f"Target Position: ({target_position['x']:.3f}, {target_position['y']:.3f}, {target_position['z']:.3f})")
    print("=" * 80)
    
    # Controller 초기화
    print("\n[1/3] Initializing Controller...")
    controller = Controller(
        agentMode="default",
        scene=args.scene,
        visibilityDistance=1.5,
        gridSize=0.25,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        width=300,
        height=300,
        fieldOfView=120,
        headless=args.headless
    )
    
    # Agent 초기화
    controller.step(dict(
        action='Initialize',
        agentMode="default",
        snapGrid=False,
        gridSize=0.25,
        rotateStepDegrees=20,
        visibilityDistance=1.5,
        fieldOfView=120,
        agentCount=1,
        handSphereRadius=0.2
    ))
    
    # 초기 agent 위치 확인
    metadata = controller.last_event.metadata
    initial_pos = metadata["agent"]["position"]
    print(f"✓ Controller initialized")
    print(f"  Initial Agent Position: ({initial_pos['x']:.3f}, {initial_pos['y']:.3f}, {initial_pos['z']:.3f})")
    
    # 목표 위치에 가장 가까운 도달 가능한 위치 찾기
    print("\n[2/3] Finding closest reachable position...")
    closest_reachable = find_closest_reachable_position(controller, target_position)
    
    if closest_reachable:
        closest_dist = distance_3d(closest_reachable, target_position)
        print(f"✓ Closest reachable position: ({closest_reachable['x']:.3f}, {closest_reachable['y']:.3f}, {closest_reachable['z']:.3f})")
        print(f"  Distance to target: {closest_dist:.3f}m")
        
        # 목표 위치를 가장 가까운 도달 가능한 위치로 업데이트
        target_position = closest_reachable
    else:
        print("⚠ No reachable positions found. Attempting to navigate to exact target position...")
    
    # 목표 위치까지 이동
    print("\n[3/3] Navigating to target position...")
    success = navigate_to_position(
        controller,
        target_position,
        goal_threshold=args.goal_threshold,
        max_iterations=args.max_iterations
    )
    
    # 최종 위치 확인
    final_metadata = controller.last_event.metadata
    final_pos = final_metadata["agent"]["position"]
    final_distance = distance_3d(final_pos, target_position)
    
    print("\n" + "=" * 80)
    print("Navigation Result")
    print("=" * 80)
    print(f"Success: {'✓' if success else '✗'}")
    print(f"Final Position: ({final_pos['x']:.3f}, {final_pos['y']:.3f}, {final_pos['z']:.3f})")
    print(f"Target Position: ({target_position['x']:.3f}, {target_position['y']:.3f}, {target_position['z']:.3f})")
    print(f"Final Distance: {final_distance:.3f}m")
    print("=" * 80)
    
    # Controller 종료
    controller.stop()
    print("\n✓ Controller stopped")


if __name__ == "__main__":
    main()

