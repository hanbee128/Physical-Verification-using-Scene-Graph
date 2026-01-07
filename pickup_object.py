#!/usr/bin/env python3
"""
Agent 위치 입력하면 teleport로 이동하고, 목표 객체를 입력하면 객체를 집는 스크립트
- metadata를 통해 손의 위치와 목표 객체의 위치를 받아옴
- moveArmBase의 y 파라미터를 계산하여 목표 객체 높이에 맞춤
- moveArm을 통해 목표 객체에 손을 가까이 이동
- 3분할 시각화: Top View, Right Side View, Agent View
"""
import sys
import time
import math

try:
    from ai2thor.controller import Controller
except ImportError:
    print("ai2thor not installed. Please install: pip install ai2thor")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not installed. Visualization will be disabled.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  cv2 not installed. Video recording will be disabled.")


def calculate_armbase_y_normalized(target_y: float, agent_y: float) -> float:
    """
    목표 객체의 y 좌표(높이)를 사용하여 MoveArmBase의 normalizedY 값 계산
    
    Args:
        target_y: 목표 객체의 y 좌표 (세계 좌표, 절대값)
        agent_y: Agent 베이스의 y 좌표 (세계 좌표, 절대값)
        
    Returns:
        normalizedY 값 (0.0~1.0)
    """
    # armBase의 y 범위는 agent_y 기준으로 대략 -0.5m ~ +1.5m 정도
    armbase_y_min = agent_y - 0.5  # 최소 높이
    armbase_y_max = agent_y + 1.5  # 최대 높이
    armbase_y_range = armbase_y_max - armbase_y_min  # 전체 범위: 2.0m
    
    if armbase_y_range <= 0:
        return 0.5  # 기본값
    
    # 목표 객체의 y 좌표와 로봇 베이스의 상대적 높이 차이 계산
    height_difference = target_y - armbase_y_min
    
    # 상대적 높이 차이를 0.0~1.0 범위로 매핑
    normalized_y = height_difference / armbase_y_range
    
    # 0.0~1.0 범위로 제한
    normalized_y = max(0.0, min(1.0, normalized_y))
    
    return normalized_y


def _update_right_side_camera(controller):
    """로봇의 오른쪽에서 Third Party Camera 추가 (고정 위치)"""
    try:
        controller.step(
            action="AddThirdPartyCamera",
            position=dict(x=2.2, y=2, z=2.2),
            rotation=dict(x=0, y=225, z=0),  # 로봇을 향하도록
            fieldOfView=90,
        )
    except Exception as e:
        # 카메라 업데이트 실패는 조용히 무시 (너무 많은 에러 메시지 방지)
        pass


def _update_realtime_visualization(controller, fig, ax_camera_top, ax_camera_right, ax_agent, 
                                   img_display_top, img_display_right, agent_img_display):
    """실시간 시각화 업데이트 (3분할: Top View, Right Side View, Agent View)"""
    if not MATPLOTLIB_AVAILABLE or not controller or not fig:
        return img_display_top, img_display_right, agent_img_display
    
    try:
        event = controller.last_event
        
        # 서드파티 카메라 이미지 업데이트
        if hasattr(event, 'third_party_camera_frames') and event.third_party_camera_frames:
            # Top view 카메라 (첫 번째)
            if len(event.third_party_camera_frames) > 0:
                img_data_top = event.third_party_camera_frames[0]
                if img_data_top is not None:
                    if isinstance(img_data_top, np.ndarray):
                        img_top = img_data_top
                    else:
                        img_top = np.array(img_data_top)
                    
                    if len(img_top.shape) == 3 and img_top.shape[2] == 4:
                        img_top = img_top[:, :, :3]
                    
                    if img_display_top is None:
                        img_display_top = ax_camera_top.imshow(img_top)
                        ax_camera_top.set_title("Top View Camera", fontsize=14, fontweight='bold')
                    else:
                        img_display_top.set_data(img_top)
                        img_display_top.set_clim(vmin=img_top.min(), vmax=img_top.max())
            
            # 로봇의 90도 오른쪽 카메라 (두 번째)
            if len(event.third_party_camera_frames) > 1:
                img_data_right = event.third_party_camera_frames[1]
                if img_data_right is not None:
                    if isinstance(img_data_right, np.ndarray):
                        img_right = img_data_right
                    else:
                        img_right = np.array(img_data_right)
                    
                    if len(img_right.shape) == 3 and img_right.shape[2] == 4:
                        img_right = img_right[:, :, :3]
                    
                    if img_display_right is None:
                        img_display_right = ax_camera_right.imshow(img_right)
                        ax_camera_right.set_title("Side View", fontsize=14, fontweight='bold')
                    else:
                        img_display_right.set_data(img_right)
                        img_display_right.set_clim(vmin=img_right.min(), vmax=img_right.max())
        
        # 에이전트 시야 이미지 업데이트
        agent_img = event.frame
        if agent_img is not None:
            if isinstance(agent_img, np.ndarray):
                agent_img_array = agent_img
            else:
                agent_img_array = np.array(agent_img)
            
            if len(agent_img_array.shape) == 3 and agent_img_array.shape[2] == 4:
                agent_img_array = agent_img_array[:, :, :3]
            
            if agent_img_display is None:
                agent_img_display = ax_agent.imshow(agent_img_array)
                ax_agent.set_title("Agent View", fontsize=14, fontweight='bold')
            else:
                agent_img_display.set_data(agent_img_array)
                agent_img_display.set_clim(vmin=agent_img_array.min(), vmax=agent_img_array.max())
        
        # 화면 업데이트
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)
    except Exception as e:
        # 시각화 업데이트 실패는 조용히 무시
        pass
    
    return img_display_top, img_display_right, agent_img_display


from scipy.spatial.transform import Rotation as R

def world_to_armbase_coords(world_pos: dict, agent_pos: dict, agent_rot: float) -> dict:
    # 1. 세계 좌표 차이 벡터 생성
    target_vec = np.array([
        world_pos["x"] - agent_pos["x"],
        world_pos["y"] - agent_pos["y"],
        world_pos["z"] - agent_pos["z"]
    ])
    
    # 2. 에이전트의 현재 회전(y축)에 대한 역행렬 생성
    # AI2-THOR는 y축 회전이 에이전트의 정면 방향을 결정함
    rot = R.from_euler('y', agent_rot, degrees=True)
    
    # 3. 역회전을 적용하여 에이전트 기준 로컬 좌표로 변환
    local_vec = rot.inv().apply(target_vec)
    
    # 4. y축 높이 보정: 에이전트 바닥에서 어깨까지의 높이(약 0.9m) 차감
    return {
        "x": local_vec[0],
        "y": local_vec[1] - 0.9, 
        "z": local_vec[2]
    }


def pickup_object_test(
    scene_name: str = "FloorPlan1_physics",
    grid_size: float = 0.25,
    position: str = None,
    object_name: str = None
):
    """
    Agent 위치를 입력받아 teleport로 이동하고, 목표 객체를 입력받아 집기
    
    Args:
        scene_name: AI2-THOR 씬 이름 (ManipulaTHOR는 _physics 접미사 필요)
        grid_size: 그리드 크기
        position: Agent 위치 (x, y, z 형식, 예: "-1.0, 0.9, 1.0"). None이면 입력받음
        object_name: 목표 객체 이름. None이면 입력받음
    """
    # Agent 위치 입력 받기
    target_position = None
    if position is None:
        print("=" * 80)
        print("Agent 위치를 입력하세요 (x, y, z 형식, 예: -1.0, 0.9, 1.0):")
        print("(엔터만 누르면 현재 위치 유지)")
        position_input = input().strip()
    else:
        position_input = position.strip()
    
    if position_input:
        try:
            position_parts = [p.strip() for p in position_input.split(",")]
            if len(position_parts) != 3:
                raise ValueError("좌표는 x, y, z 3개의 값이 필요합니다.")
            
            target_position = {
                "x": float(position_parts[0]),
                "y": float(position_parts[1]),
                "z": float(position_parts[2])
            }
            print(f"✓ 목표 좌표: ({target_position['x']:.3f}, {target_position['y']:.3f}, {target_position['z']:.3f})")
        except ValueError as e:
            print(f"❌ 좌표 입력 오류: {e}")
            print("올바른 형식: x, y, z (예: -1.0, 0.9, 1.0)")
            return
    
    # 목표 객체 입력 받기
    if object_name is None:
        print("\n" + "=" * 80)
        print("목표 객체 이름을 입력하세요 (예: Apple, Plate, Mug 등):")
        object_name = input().strip()
    
    if not object_name:
        print("❌ 객체 이름이 입력되지 않았습니다.")
        return
    
    print(f"✓ 목표 객체: {object_name}")
    
    # Controller 초기화 (ManipulaTHOR 모드)
    print(f"\n{'=' * 80}")
    print(f"Controller 초기화 중... (씬: {scene_name})")
    controller = Controller(
        height=1000,
        width=1000,
        headless=False
    )
    controller.reset(scene_name)
    
    # Agent 초기화 (이동은 default 모드로)
    controller.step(dict(
        action='Initialize',
        agentMode="default",  # 이동은 default 모드 사용
        snapGrid=False,
        gridSize=grid_size,
        rotateStepDegrees=20,
        visibilityDistance=1.5,
        fieldOfView=120,
        agentCount=1
    ))
    
    # Top view camera 추가
    event = controller.step(action="GetMapViewCameraProperties")
    controller.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])
    print(f"  ✓ Top View Camera 추가 완료")
    
    # 로봇의 90도 오른쪽에서 Third Party Camera 추가 (초기 위치)
    _update_right_side_camera(controller)
    
    print("✓ Controller 초기화 완료")
    
    # 3분할 시각화 초기화
    fig = None
    ax_camera_top = None
    ax_camera_right = None
    ax_agent = None
    img_display_top = None
    img_display_right = None
    agent_img_display = None
    
    if MATPLOTLIB_AVAILABLE:
        try:
            plt.ion()
            fig = plt.figure(figsize=(30, 8))
            fig.canvas.manager.set_window_title("Agent View & Third Party Camera Views")
            
            # Top view 카메라 뷰 (왼쪽)
            ax_camera_top = fig.add_subplot(131)
            ax_camera_top.axis('off')
            ax_camera_top.set_title("Top View Camera", fontsize=14, fontweight='bold')
            
            # 로봇의 90도 오른쪽 카메라 뷰 (가운데)
            ax_camera_right = fig.add_subplot(132)
            ax_camera_right.axis('off')
            ax_camera_right.set_title("Side View", fontsize=14, fontweight='bold')
            
            # 에이전트 시야 뷰 (오른쪽)
            ax_agent = fig.add_subplot(133)
            ax_agent.axis('off')
            ax_agent.set_title("Agent View", fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.show(block=False)
            
            print("  ✓ 3분할 시각화 초기화 완료")
        except Exception as e:
            print(f"  ⚠️ 실시간 시각화 초기화 실패: {e}")
            fig = None
    
    # 현재 agent 위치 확인
    current_event = controller.last_event
    current_pos = current_event.metadata.get("agent", {}).get("position", {})
    print(f"현재 위치: ({current_pos.get('x', 0):.3f}, {current_pos.get('y', 0):.3f}, {current_pos.get('z', 0):.3f})")
    
    # 1. Agent 위치로 Teleport (입력된 경우)
    if target_position:
        print(f"\n{'=' * 80}")
        print(f"목표 좌표로 Teleport 중...")
        teleport_result = controller.step(
            action="Teleport",
            position=target_position,
            agentId=0
        )
        
        # 카메라 업데이트 및 시각화
        _update_right_side_camera(controller)
        if fig:
            img_display_top, img_display_right, agent_img_display = _update_realtime_visualization(
                controller, fig, ax_camera_top, ax_camera_right, ax_agent,
                img_display_top, img_display_right, agent_img_display
            )
        
        if teleport_result.metadata.get("lastActionSuccess", False):
            new_pos = teleport_result.metadata.get("agent", {}).get("position", {})
            print(f"✓ Teleport 성공!")
            print(f"  새 위치: ({new_pos.get('x', 0):.3f}, {new_pos.get('y', 0):.3f}, {new_pos.get('z', 0):.3f})")
        else:
            error_msg = teleport_result.metadata.get('errorMessage', 'Unknown error')
            print(f"⚠️ Teleport 실패: {error_msg}")
            print("  목표 위치와 가장 가까운 가능한 위치를 찾는 중...")
            
            # GetReachablePositions로 이동 가능한 위치들 가져오기
            reachable_event = controller.step(action="GetReachablePositions")
            if reachable_event.metadata.get("lastActionSuccess", False):
                reachable_positions = reachable_event.metadata.get("actionReturn", [])
                
                if reachable_positions:
                    # 목표 위치와 가장 가까운 위치 찾기
                    target_x = target_position.get("x", 0)
                    target_y = target_position.get("y", 0)
                    target_z = target_position.get("z", 0)
                    
                    closest_pos = None
                    min_distance = float('inf')
                    
                    for pos in reachable_positions:
                        # 3D 거리 계산
                        dx = pos.get("x", 0) - target_x
                        dy = pos.get("y", 0) - target_y
                        dz = pos.get("z", 0) - target_z
                        distance = math.sqrt(dx**2 + dy**2 + dz**2)
                        
                        if distance < min_distance:
                            min_distance = distance
                            closest_pos = pos
                    
                    if closest_pos:
                        print(f"  가장 가까운 가능한 위치: ({closest_pos.get('x', 0):.3f}, {closest_pos.get('y', 0):.3f}, {closest_pos.get('z', 0):.3f}) (거리: {min_distance:.3f}m)")
                        
                        # 가장 가까운 위치로 Teleport 시도
                        closest_teleport_result = controller.step(
                            action="Teleport",
                            position=closest_pos,
                            agentId=0
                        )
                        
                        if closest_teleport_result.metadata.get("lastActionSuccess", False):
                            final_pos = closest_teleport_result.metadata.get("agent", {}).get("position", {})
                            print(f"✓ 가장 가까운 위치로 Teleport 성공!")
                            print(f"  최종 위치: ({final_pos.get('x', 0):.3f}, {final_pos.get('y', 0):.3f}, {final_pos.get('z', 0):.3f})")
                            
                            # 카메라 업데이트 및 시각화
                            _update_right_side_camera(controller)
                            if fig:
                                img_display_top, img_display_right, agent_img_display = _update_realtime_visualization(
                                    controller, fig, ax_camera_top, ax_camera_right, ax_agent,
                                    img_display_top, img_display_right, agent_img_display
                                )
                        else:
                            closest_error = closest_teleport_result.metadata.get('errorMessage', 'Unknown error')
                            print(f"  ⚠️ 가장 가까운 위치로도 Teleport 실패: {closest_error}")
                            print("  현재 위치에서 계속 진행합니다.")
                    else:
                        print("  ⚠️ 이동 가능한 위치를 찾을 수 없습니다.")
                        print("  현재 위치에서 계속 진행합니다.")
                else:
                    print("  ⚠️ 이동 가능한 위치가 없습니다.")
                    print("  현재 위치에서 계속 진행합니다.")
            else:
                print("  ⚠️ GetReachablePositions 실패")
                print("  현재 위치에서 계속 진행합니다.")
            
            time.sleep(1)
    else:
        print("\n위치 입력이 없어 현재 위치를 유지합니다.")
    
    # 2. 목표 객체 찾기
    print(f"\n{'=' * 80}")
    print(f"목표 객체 '{object_name}' 찾는 중...")
    
    all_objects = controller.last_event.metadata.get("objects", [])
    target_object_id = None
    
    for obj in all_objects:
        obj_id = obj.get("objectId", "")
        obj_type = obj.get("objectType", "")
        # 객체 이름이나 타입으로 매칭
        if object_name.lower() in obj_id.lower() or object_name.lower() in obj_type.lower():
            target_object_id = obj_id
            obj_pos = obj.get("position", {})
            print(f"✓ 객체 발견: {obj_id}")
            print(f"  위치: ({obj_pos.get('x', 0):.3f}, {obj_pos.get('y', 0):.3f}, {obj_pos.get('z', 0):.3f})")
            break
    
    if not target_object_id:
        print(f"❌ 객체 '{object_name}'를 찾을 수 없습니다.")
        print("\n사용 가능한 객체 목록:")
        obj_types = set()
        for obj in all_objects[:20]:  # 처음 20개만 표시
            obj_type = obj.get("objectType", "Unknown")
            if obj_type not in obj_types:
                obj_types.add(obj_type)
                print(f"  - {obj_type}")
        if len(all_objects) > 20:
            print(f"  ... (총 {len(all_objects)}개 객체)")
        controller.stop()
        return
    
    # 2.5. agentMode를 "arm"으로 변경 (pickup을 위해)
    print(f"\n{'=' * 80}")
    print(f"agentMode를 'arm'으로 변경 중... (pickup을 위해)")
    
    # 현재 위치와 회전 정보 저장
    current_event = controller.last_event
    current_agent_pos = current_event.metadata.get("agent", {}).get("position", {})
    current_agent_rot = current_event.metadata.get("agent", {}).get("rotation", {})
    
    # agentMode를 "arm"으로 변경하기 위해 다시 Initialize 호출
    arm_init_result = controller.step(dict(
        action='Initialize',
        agentMode="arm",  # pickup을 위해 arm 모드로 변경
        snapGrid=False,
        gridSize=grid_size,
        rotateStepDegrees=20,
        visibilityDistance=1.5,
        fieldOfView=120,
        agentCount=1,
        handSphereRadius=0.5  # handSphereRadius를 0.5로 설정
    ))
    
    if arm_init_result.metadata.get("lastActionSuccess", False):
        # Initialize 후 위치가 변경될 수 있으므로 다시 Teleport
        if current_agent_pos:
            restore_teleport = controller.step(
                action="Teleport",
                position=current_agent_pos,
                rotation=current_agent_rot,
                agentId=0
            )
            if restore_teleport.metadata.get("lastActionSuccess", False):
                print(f"✓ agentMode를 'arm'으로 변경 완료 (위치 유지)")
            else:
                print(f"⚠️ agentMode 변경 후 위치 복원 실패, 현재 위치에서 계속 진행")
        else:
            print(f"✓ agentMode를 'arm'으로 변경 완료")
    else:
        error_msg = arm_init_result.metadata.get('errorMessage', 'Unknown error')
        print(f"⚠️ agentMode 변경 실패: {error_msg}")
        print("  default 모드로 계속 진행합니다.")
    
    time.sleep(0.5)
    
    # 3. metadata에서 손의 위치와 목표 객체의 위치 가져오기
    print(f"\n{'=' * 80}")
    print(f"객체 '{object_name}' ({target_object_id}) 집는 중...")
    
    # 최신 metadata 가져오기
    current_event = controller.last_event
    metadata = current_event.metadata
    
    # 목표 객체의 중심 위치 가져오기 (axisAlignedBoundingBox.center)
    target_obj_metadata = None
    for obj in metadata.get("objects", []):
        if obj.get("objectId") == target_object_id:
            target_obj_metadata = obj
            break
    
    if not target_obj_metadata:
        print(f"❌ 객체 '{object_name}'의 metadata를 찾을 수 없습니다.")
        controller.stop()
        return
    
    # 객체의 axisAlignedBoundingBox.center에서 세계 좌표 추출
    bbox = target_obj_metadata.get("axisAlignedBoundingBox", {})
    obj_center = bbox.get("center", {})
    
    if not obj_center or not all(key in obj_center for key in ["x", "y", "z"]):
        print(f"❌ 객체 '{object_name}'의 위치 정보가 유효하지 않습니다.")
        controller.stop()
        return
    
    print(f"  [Object Metadata] 목표 객체 위치: ({obj_center['x']:.3f}, {obj_center['y']:.3f}, {obj_center['z']:.3f})")
    
    # Agent 위치와 회전 가져오기
    agent_pos = metadata["agent"]["position"]
    agent_rot = metadata["agent"]["rotation"]["y"]
    
    print(f"  [Agent] 위치: ({agent_pos['x']:.3f}, {agent_pos['y']:.3f}, {agent_pos['z']:.3f}), 회전: {agent_rot:.1f}°")
    
    # 4. 초기 arm 상태 설정
    print(f"\n  [0/5] Arm 초기 상태 설정 중...")
    
    # MoveArmBase를 0.6으로 설정
    print(f"  [MoveArmBase] 초기화: y=0.7")
    init_armbase_result = controller.step(
        action="MoveArmBase",
        y=0.7,
        normalizedY=True,
        agentId=0
    )
    
    # 카메라 업데이트 및 시각화
    _update_right_side_camera(controller)
    if fig:
        img_display_top, img_display_right, agent_img_display = _update_realtime_visualization(
            controller, fig, ax_camera_top, ax_camera_right, ax_agent,
            img_display_top, img_display_right, agent_img_display
        )
    
    time.sleep(0.2)
    
    if not init_armbase_result.metadata.get('lastActionSuccess', False):
        error_msg = init_armbase_result.metadata.get('errorMessage', 'Unknown error')
        print(f"  ⚠ MoveArmBase 초기화 실패: {error_msg}")
    else:
        print(f"  ✓ MoveArmBase 초기화 완료")
    
    # MoveArm을 (x=0, y=0, z=0)으로 설정
    print(f"  [MoveArm] 초기화: (x=0, y=0, z=0)")
    init_arm_result = controller.step(
        action="MoveArm",
        position={"x": 0, "y": 0, "z": 0},
        coordinateSpace="armBase",
        agentId=0
    )
    
    # 카메라 업데이트 및 시각화
    _update_right_side_camera(controller)
    if fig:
        img_display_top, img_display_right, agent_img_display = _update_realtime_visualization(
            controller, fig, ax_camera_top, ax_camera_right, ax_agent,
            img_display_top, img_display_right, agent_img_display
        )
    
    time.sleep(0.3)
    
    if not init_arm_result.metadata.get('lastActionSuccess', False):
        error_msg = init_arm_result.metadata.get('errorMessage', 'Unknown error')
        print(f"  ⚠ MoveArm 초기화 실패: {error_msg}")
    else:
        print(f"  ✓ MoveArm 초기화 완료")
    
    # 초기화 후 최신 metadata 가져오기
    current_event = controller.last_event
    metadata = current_event.metadata
    agent_pos = metadata["agent"]["position"]
    agent_rot = metadata["agent"]["rotation"]["y"]
    
    # 5. 목표 객체를 정면에서 볼 수 있도록 회전
    print(f"\n  [0/4] 목표 객체를 정면으로 바라보도록 회전 중...")
    
    # Agent와 목표 객체 사이의 각도 계산
    dx = obj_center["x"] - agent_pos["x"]
    dz = obj_center["z"] - agent_pos["z"]
    
    # 목표 방향 각도 계산 (라디안)
    target_angle_rad = math.atan2(dx, dz)
    target_angle_deg = math.degrees(target_angle_rad)
    
    # 각도를 0~360 범위로 정규화
    target_angle_deg = (target_angle_deg + 360) % 360
    
    # 현재 Agent 회전 각도와의 차이 계산
    angle_diff = target_angle_deg - agent_rot
    
    # -180 ~ 180 범위로 정규화 (가장 짧은 회전 경로 선택)
    if angle_diff > 180:
        angle_diff -= 360
    elif angle_diff < -180:
        angle_diff += 360
    
    print(f"  [Rotation] 현재 각도: {agent_rot:.1f}°, 목표 각도: {target_angle_deg:.1f}°, 차이: {angle_diff:.1f}°")
    
    # 5도 이상 차이가 있으면 회전
    while abs(angle_diff) > 5.0:
        # 회전 방향 결정 (양수: 오른쪽, 음수: 왼쪽)
        if angle_diff > 0:
            rotate_action = "RotateAgent"
            rotate_degrees = min(abs(angle_diff), 10.0)  # 최대 90도씩 회전
        else:
            rotate_action = "RotateAgent"
            rotate_degrees = -min(abs(angle_diff), 10.0)  # 최대 90도씩 회전
        
        print(f"  [Rotation] {rotate_action} 실행: {rotate_degrees:.1f}°")
        
        rotate_result = controller.step(
            action=rotate_action,
            degrees=rotate_degrees,
            agentId=0,
            speed=1.0
        )
        
        # 카메라 업데이트 및 시각화
        _update_right_side_camera(controller)
        if fig:
            img_display_top, img_display_right, agent_img_display = _update_realtime_visualization(
                controller, fig, ax_camera_top, ax_camera_right, ax_agent,
                img_display_top, img_display_right, agent_img_display
            )
        
        time.sleep(0.5)
        
        if not rotate_result.metadata.get('lastActionSuccess', False):
            error_msg = rotate_result.metadata.get('errorMessage', 'Unknown error')
            print(f"  ⚠ 회전 실패: {error_msg}")
        else:
            # 회전 후 최신 metadata 업데이트
            current_event = controller.last_event
            metadata = current_event.metadata
            agent_pos = metadata["agent"]["position"]
            agent_rot = metadata["agent"]["rotation"]["y"]
            print(f"  ✓ 회전 완료: 현재 각도 {agent_rot:.1f}°")
            
            # 추가 미세 조정이 필요한지 확인
            dx = obj_center["x"] - agent_pos["x"]
            dz = obj_center["z"] - agent_pos["z"]
            target_angle_rad = math.atan2(dx, dz)
            target_angle_deg = (math.degrees(target_angle_rad) + 360) % 360
            angle_diff = target_angle_deg - agent_rot
            if angle_diff > 180:
                angle_diff -= 360
            elif angle_diff < -180:
                angle_diff += 360
            
            # 5도 이상 차이가 있으면 추가 미세 조정
            if abs(angle_diff) > 10.0:
                if angle_diff > 0:
                    rotate_degrees = min(abs(angle_diff), 30.0)
                else:
                    rotate_degrees = -min(abs(angle_diff), 30.0)
                
                print(f"  [Fine Adjustment] 추가 미세 조정: {rotate_degrees:.1f}°")
                rotate_result = controller.step(
                    action="RotateAgent",
                    degrees=rotate_degrees,
                    agentId=0,
                    speed=1.0
                )
                
                # 카메라 업데이트 및 시각화
                _update_right_side_camera(controller)
                if fig:
                    img_display_top, img_display_right, agent_img_display = _update_realtime_visualization(
                        controller, fig, ax_camera_top, ax_camera_right, ax_agent,
                        img_display_top, img_display_right, agent_img_display
                    )
                
                time.sleep(0.3)
                
                # 최신 metadata 업데이트
                current_event = controller.last_event
                metadata = current_event.metadata
                agent_pos = metadata["agent"]["position"]
                agent_rot = metadata["agent"]["rotation"]["y"]
    else:
        print(f"  ✓ 이미 정면을 향하고 있음 (차이: {angle_diff:.1f}°)")
    
    # 회전 후 최신 metadata 가져오기
    current_event = controller.last_event
    metadata = current_event.metadata
    agent_pos = metadata["agent"]["position"]
    agent_rot = metadata["agent"]["rotation"]["y"]
    
    # 현재 손의 위치 가져오기 (handSphereCenter)
    arm_metadata = metadata.get("arm", {})
    hand_sphere_center = arm_metadata.get("handSphereCenter", {})
    
    if not hand_sphere_center or not all(key in hand_sphere_center for key in ["x", "y", "z"]):
        print(f"  ⚠ Hand position not available in metadata")
        controller.stop()
        return
    
    print(f"  [Hand Metadata] 손의 위치: ({hand_sphere_center['x']:.3f}, {hand_sphere_center['y']:.3f}, {hand_sphere_center['z']:.3f})")
    
    # 6. moveArmBase의 y 파라미터 계산 및 조정
    target_y = obj_center["y"]  # 객체의 세계 좌표 y 값 (높이)
    agent_base_y = agent_pos["y"]  # 로봇 베이스의 세계 좌표 y 값
    
    # 상대적 높이 차이를 0.0~1.0으로 매핑하여 normalizedY 계산
    normalized_y = calculate_armbase_y_normalized(target_y, agent_base_y)
    
    print(f"  [MoveArmBase Calculation] Agent base y: {agent_base_y:.3f}, Target y: {target_y:.3f}")
    print(f"  [MoveArmBase Calculation] Height difference: {target_y - agent_base_y:.3f}m, normalizedY: {normalized_y:.3f}")
    
    print(f"\n  [1/5] MoveArmBase 조정 중... (y={normalized_y:.3f})")
    move_armbase_result = controller.step(
        action="MoveArmBase",
        y=0.7,
        normalizedY=True,
        agentId=0
    )
    
    # 카메라 업데이트 및 시각화
    _update_right_side_camera(controller)
    if fig:
        img_display_top, img_display_right, agent_img_display = _update_realtime_visualization(
            controller, fig, ax_camera_top, ax_camera_right, ax_agent,
            img_display_top, img_display_right, agent_img_display
        )
    
    time.sleep(0.2)
    
    if not move_armbase_result.metadata.get('lastActionSuccess', False):
        error_msg = move_armbase_result.metadata.get('errorMessage', 'Unknown error')
        print(f"  ⚠ MoveArmBase failed: {error_msg}")
    else:
        print(f"  ✓ MoveArmBase 조정 완료")
    
    #7. moveArm을 통해 목표 객체에 손을 가까이 이동 (armBase 기준 좌표 차이 계산)
    print(f"\n  [2/5] MoveArm 실행 중... (armBase 기준 좌표 차이 계산)")
    
    # MoveArmBase 조정 후 최신 metadata 가져오기
    current_event = controller.last_event
    metadata = current_event.metadata
    agent_pos = metadata["agent"]["position"]
    agent_rot = metadata["agent"]["rotation"]["y"]
    
    # 목표 객체의 armBase 좌표 계산
    target_armbase = world_to_armbase_coords(obj_center, agent_pos, agent_rot)
    print(f"  [Target] 목표 객체 armBase 좌표: ({target_armbase['x']:.3f}, {target_armbase['y']:.3f}, {target_armbase['z']:.3f})")
    
    # 현재 손의 위치를 metadata에서 가져오기
    arm_metadata = metadata.get("arm", {})
    hand_sphere_center = arm_metadata.get("handSphereCenter", {})
    
    if not hand_sphere_center or not all(key in hand_sphere_center for key in ["x", "y", "z"]):
        print(f"  ⚠ Hand position not available in metadata")
        controller.stop()
        return
    
    # 현재 손의 armBase 좌표 계산
    hand_armbase = world_to_armbase_coords(hand_sphere_center, agent_pos, agent_rot)
    print(f"  [Current Hand] 현재 손 armBase 좌표: ({hand_armbase['x']:.3f}, {hand_armbase['y']:.3f}, {hand_armbase['z']:.3f})")
    
    # 범위 제한
    move_pos = {
        "x": max(-1.0, min(1.0, target_armbase["x"])),  # x 범위 제한: -1 ~ 1
        "y": max(-1.0, min(1.0, target_armbase["y"])),  # y 범위 제한: -1 ~ 1
        "z": max(0.0, min(1.0, target_armbase["z"]))   # z 범위 제한: 0 ~ 1
    }
    
    move_arm_result = controller.step(
        action="MoveArm",
        position=move_pos,
        coordinateSpace="armBase",
        agentId=0,
        speed=1.0,
        restrictMovement=False
    )
    
    # 카메라 업데이트 및 시각화
    _update_right_side_camera(controller)
    if fig:
        img_display_top, img_display_right, agent_img_display = _update_realtime_visualization(
            controller, fig, ax_camera_top, ax_camera_right, ax_agent,
            img_display_top, img_display_right, agent_img_display
        )
    
    time.sleep(0.5)
    
    if not move_arm_result.metadata.get('lastActionSuccess', False):
        error_msg = move_arm_result.metadata.get('errorMessage', 'Unknown error')
        print(f"  ⚠ MoveArm 이동 실패: {error_msg}")
        
        # 충돌이 발생한 경우 MoveArmBase를 y=0.6으로 설정한 뒤 재시도
        if error_msg and ('collided' in error_msg.lower() or 'collision' in error_msg.lower()):
            print(f"  [Collision Detected] MoveArmBase를 y=0.6으로 재설정 후 재시도...")
            
            # MoveArmBase를 y=0.6으로 설정
            retry_armbase_result = controller.step(
                action="MoveArmBase",
                y=0.6,
                normalizedY=True,
                agentId=0
            )
            time.sleep(0.2)
            
            if not retry_armbase_result.metadata.get('lastActionSuccess', False):
                error_msg_armbase = retry_armbase_result.metadata.get('errorMessage', 'Unknown error')
                print(f"  ⚠ MoveArmBase 재설정 실패: {error_msg_armbase}")
            else:
                print(f"  ✓ MoveArmBase 재설정 완료 (y=0.6)")
                
                # 최신 metadata 업데이트
                current_event = controller.last_event
                metadata = current_event.metadata
                agent_pos = metadata["agent"]["position"]
                agent_rot = metadata["agent"]["rotation"]["y"]
                
                # 목표 객체의 armBase 좌표 재계산
                target_armbase = world_to_armbase_coords(obj_center, agent_pos, agent_rot)
                print(f"  [Retry Target] 목표 객체 armBase 좌표: ({target_armbase['x']:.3f}, {target_armbase['y']:.3f}, {target_armbase['z']:.3f})")
                
                # 현재 손의 armBase 좌표 재계산
                arm_metadata = metadata.get("arm", {})
                hand_sphere_center = arm_metadata.get("handSphereCenter", {})
                if hand_sphere_center and all(key in hand_sphere_center for key in ["x", "y", "z"]):
                    hand_armbase = world_to_armbase_coords(hand_sphere_center, agent_pos, agent_rot)
                    print(f"  [Retry Current Hand] 현재 손 armBase 좌표: ({hand_armbase['x']:.3f}, {hand_armbase['y']:.3f}, {hand_armbase['z']:.3f})")
                    
                    # 기존 방식과 동일: target_armbase를 직접 사용하고 범위 제한 적용
                    move_pos_retry = {
                        "x": max(-1.0, min(1.0, target_armbase["x"])),  # x 범위 제한: -1 ~ 1
                        "y": max(-1.0, min(1.0, target_armbase["y"])),  # y 범위 제한: -1 ~ 1
                        "z": max(0.0, min(1.0, target_armbase["z"]))   # z 범위 제한: 0 ~ 1
                    }
                    
                    print(f"  [Retry MoveArm] 재계산된 위치: ({move_pos_retry['x']:.3f}, {move_pos_retry['y']:.3f}, {move_pos_retry['z']:.3f})")
                    
                    # MoveArm 재시도 (기존 방식과 동일한 파라미터 사용)
                    move_arm_result = controller.step(
                        action="MoveArm",
                        position=move_pos_retry,
                        coordinateSpace="armBase",
                        agentId=0,
                        speed=1.0,
                        restrictMovement=False
                    )
                    
                    # 카메라 업데이트 및 시각화
                    _update_right_side_camera(controller)
                    if fig:
                        img_display_top, img_display_right, agent_img_display = _update_realtime_visualization(
                            controller, fig, ax_camera_top, ax_camera_right, ax_agent,
                            img_display_top, img_display_right, agent_img_display
                        )
                    
                    time.sleep(0.5)
                    
                    if not move_arm_result.metadata.get('lastActionSuccess', False):
                        error_msg_retry = move_arm_result.metadata.get('errorMessage', 'Unknown error')
                        print(f"  ⚠ MoveArm 재시도 실패: {error_msg_retry}")
                    else:
                        print(f"  ✓ MoveArm 재시도 성공")
    else:
        print(f"  ✓ MoveArm 이동 완료")
        
        # 이동 후 실제 손 위치를 metadata에서 재계산
        current_event = controller.last_event
        metadata = current_event.metadata
        agent_pos = metadata["agent"]["position"]
        agent_rot = metadata["agent"]["rotation"]["y"]
        arm_metadata = metadata.get("arm", {})
        hand_sphere_center = arm_metadata.get("handSphereCenter", {})
        if hand_sphere_center and all(key in hand_sphere_center for key in ["x", "y", "z"]):
            # 실제 이동된 손의 armBase 좌표 재계산
            hand_armbase_after = world_to_armbase_coords(hand_sphere_center, agent_pos, agent_rot)
            print(f"  [After MoveArm] 실제 이동된 손 위치: ({hand_armbase_after['x']:.3f}, {hand_armbase_after['y']:.3f}, {hand_armbase_after['z']:.3f})")
            
            # 목표 객체와의 최종 거리 확인
            target_armbase_final = world_to_armbase_coords(obj_center, agent_pos, agent_rot)
            final_distance = ((target_armbase_final["x"] - hand_armbase_after["x"])**2 + 
                             (target_armbase_final["y"] - hand_armbase_after["y"])**2 + 
                             (target_armbase_final["z"] - hand_armbase_after["z"])**2)**0.5
            print(f"  [Final Distance] 손과 목표 객체 간 거리: {final_distance:.3f}m")
    
    # 8. 집기 전에 objectIdCandidates에 목표 객체가 있는지 확인
    print(f"\n  [3/5] PickupObject 실행 전 확인...")
    
    # 최신 metadata에서 objectIdCandidates 확인
    current_event = controller.last_event
    metadata = current_event.metadata
    
    # PickupObject가 접근 가능한 객체 목록 확인
    # metadata에서 직접 확인할 수 있는 방법이 없으므로, 객체가 여전히 존재하는지 확인
    all_objects = metadata.get("objects", [])
    object_exists = False
    for obj in all_objects:
        if obj.get("objectId") == target_object_id:
            object_exists = True
            obj_pos = obj.get("position", {})
            print(f"  ✓ 목표 객체 '{target_object_id}' 확인됨")
            print(f"    현재 위치: ({obj_pos.get('x', 0):.3f}, {obj_pos.get('y', 0):.3f}, {obj_pos.get('z', 0):.3f})")
            break
    
    if not object_exists:
        print(f"  ⚠ 목표 객체 '{target_object_id}'를 찾을 수 없습니다.")
        controller.stop()
        return
    
    # 손의 현재 위치와 객체 위치 확인
    arm_metadata = metadata.get("arm", {})
    hand_sphere_center = arm_metadata.get("handSphereCenter", {})
    if hand_sphere_center:
        hand_x = hand_sphere_center.get("x", 0)
        hand_y = hand_sphere_center.get("y", 0)
        hand_z = hand_sphere_center.get("z", 0)
        obj_x = obj_center.get("x", 0)
        obj_y = obj_center.get("y", 0)
        obj_z = obj_center.get("z", 0)
        
        distance = math.sqrt((hand_x - obj_x)**2 + (hand_y - obj_y)**2 + (hand_z - obj_z)**2)
        print(f"  [Distance] 손과 객체 사이 거리: {distance:.3f}m")
    
    # 9. PickupObject 실행
    print(f"\n  [4/5] PickupObject 실행 중...")
    pickup_result = controller.step(
        action="PickupObject",
        objectIdCandidates=[target_object_id],  # ManipulaTHOR는 리스트로 전달
        agentId=0
        # objectIdCandidates를 사용할 때는 forceAction을 사용할 수 없음
    )
    
    # 카메라 업데이트 및 시각화
    _update_right_side_camera(controller)
    if fig:
        img_display_top, img_display_right, agent_img_display = _update_realtime_visualization(
            controller, fig, ax_camera_top, ax_camera_right, ax_agent,
            img_display_top, img_display_right, agent_img_display
        )
    
    time.sleep(0.1)
    
    if pickup_result.metadata.get("lastActionSuccess", False):
        print(f"✓ 객체 '{object_name}' 집기 성공!")
        
        # 손에 들고 있는 객체 확인
        inventory = pickup_result.metadata.get("inventoryObjects", [])
        if inventory:
            print(f"  손에 든 객체: {inventory}")
    else:
        error_msg = pickup_result.metadata.get('errorMessage', 'Unknown error')
        print(f"❌ 객체 집기 실패: {error_msg}")
        print("\n팁: 객체에 충분히 가까이 있는지, 객체가 집을 수 있는 객체인지 확인하세요.")
    
    print(f"\n{'=' * 80}")
    print("완료! 아무 키나 누르면 종료합니다...")
    input()
    
    controller.stop()
    print("✓ Controller 종료")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Agent 위치로 이동하고 객체 집기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python pickup_object.py -1.000, 0.901, 0.500 apple
  python pickup_object.py --position "-1.0, 0.9, 1.0" --object apple
  python pickup_object.py  # 대화형 모드 (위치와 객체 이름 입력)
        """
    )
    parser.add_argument("pos", nargs="?", type=str, default=None,
                        help="Agent 위치 (x, y, z 형식, 예: '-1.000, 0.901, 0.500')")
    parser.add_argument("obj", nargs="?", type=str, default=None,
                        help="목표 객체 이름 (예: apple, plate, mug)")
    parser.add_argument("--scene", type=str, default="FloorPlan1_physics", 
                        help="씬 이름 (기본값: FloorPlan1_physics)")
    parser.add_argument("--grid-size", type=float, default=0.25, 
                        help="그리드 크기 (기본값: 0.25)")
    parser.add_argument("--position", type=str, default=None,
                        help="Agent 위치 (x, y, z 형식). pos 인자보다 우선순위 낮음")
    parser.add_argument("--object", type=str, default=None,
                        help="목표 객체 이름. obj 인자보다 우선순위 낮음")
    
    args = parser.parse_args()
    
    # 위치 인자 처리: positional argument (pos) > --position 옵션
    position = args.pos if args.pos else args.position
    
    # 객체 이름 인자 처리: positional argument (obj) > --object 옵션
    object_name = args.obj if args.obj else args.object
    
    try:
        pickup_object_test(
            scene_name=args.scene,
            grid_size=args.grid_size,
            position=position,
            object_name=object_name
        )
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
