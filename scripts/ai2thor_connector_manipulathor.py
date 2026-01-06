#!/usr/bin/env python3
"""
AI2-THOR ManipulaTHOR Connector - ProgPrompt 형식의 PLAN을 실행할 수 있는 클래스

ProgPrompt 형식의 코드를 파싱하고 AI2-THOR ManipulaTHOR에서 실행합니다.
예: GoTo('Apple'), Pickup('Apple'), Put('Apple', 'Fridge') 등

ManipulaTHOR는 Arm Agent를 사용하므로 일반 AI2-THOR와 다른 API를 사용합니다.
"""

import cv2
import math
import numpy as np
import os
import re
import shutil
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not installed. Path visualization will be disabled.")

try:
    from ai2thor.controller import Controller
except ImportError:
    print("⚠️  ai2thor not installed. Please install: pip install ai2thor")

try:
    from utils_aug_env import normalize_object_type
except ImportError:
    # Fallback if import fails
    def normalize_object_type(obj_type: str) -> str:
        """Normalize object type (ButterKnife -> Knife)"""
        if obj_type == "ButterKnife":
            return "Knife"
        return obj_type


def distance_pts(pt1, pt2):
    """두 점 사이의 거리 계산"""
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(pt1, pt2)]))


def closest_node(target_pos, reachable_positions, no_agents, clost_node_location):
    """타겟 위치에 가장 가까운 도달 가능한 위치들을 찾기"""
    distances = []
    for pos in reachable_positions:
        dist = distance_pts(target_pos, pos)
        distances.append((dist, pos))
    
    distances.sort(key=lambda x: x[0])
    
    # 각 agent마다 다른 위치 할당
    selected = []
    for i in range(no_agents):
        idx = (clost_node_location[i] + i) % len(distances)
        selected.append(distances[idx][1])
    
    return selected


class ManipulaThorExecutor:
    """ProgPrompt 형식의 PLAN을 AI2-THOR ManipulaTHOR에서 실행하는 클래스"""
    
    def __init__(
        self,
        scene: str = "FloorPlan1",
        agent_id: int = 0,
        headless: bool = False,
        save_images: bool = False,
        image_dir: str = "execution_images",
        save_video: bool = True,
        video_fps: float = 10.0,
        video_dir: str = "execution_videos",
        initial_position: Optional[Tuple[float, float, float]] = None,
        initial_position_strategy: str = "first",  # "first", "random", "center", "custom"
    ):
        """
        Args:
            scene: AI2-THOR scene 이름
            agent_id: Agent ID (기본값: 0)
            headless: 헤드리스 모드 (GUI 없이 실행)
            save_images: 이미지 저장 여부
            image_dir: 이미지 저장 디렉토리
            save_video: 영상 저장 여부 (기본값: True)
            video_fps: 영상 FPS (기본값: 10.0)
            video_dir: 영상 저장 디렉토리
            initial_position: 초기 위치 (x, y, z) 튜플. None이면 strategy에 따라 결정
            initial_position_strategy: 초기 위치 선택 전략
                - "first": 첫 번째 도달 가능한 위치 (기본값)
                - "random": 랜덤 위치
                - "center": 씬의 중심에 가까운 위치
                - "custom": initial_position 사용
        """
        self.scene = scene
        self.agent_id = agent_id
        self.headless = headless
        self.save_images = save_images
        self.image_dir = image_dir
        self.save_video = save_video
        self.video_fps = video_fps
        self.video_dir = video_dir
        self.initial_position = initial_position
        self.initial_position_strategy = initial_position_strategy
        
        self.controller = None
        self.action_queue = []
        self.task_over = False
        self.reachable_positions = []
        self.total_exec = 0
        self.success_exec = 0
        self.execution_log = []
        
        # 비디오 저장 관련
        self.video_writer = None
        self.video_frames = []
        self.video_width = 1000
        self.video_height = 1000
        
        # 실시간 시각화 관련 (Third Party View & Agent View)
        self.show_realtime_view = not headless  # 헤드리스 모드가 아니면 실시간 시각화 활성화
        self.fig = None
        self.ax_camera = None
        self.ax_agent = None
        self.img_display = None
        self.agent_img_display = None
        
        # 이미지 저장 관련
        if self.save_images:
            self._setup_image_dirs()
        
        # 비디오 저장 디렉토리 설정
        if self.save_video:
            os.makedirs(self.video_dir, exist_ok=True)
    
    def _setup_image_dirs(self):
        """이미지 저장 디렉토리 설정"""
        if os.path.exists(self.image_dir):
            shutil.rmtree(self.image_dir)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(os.path.join(self.image_dir, "agent"), exist_ok=True)
        os.makedirs(os.path.join(self.image_dir, "top_view"), exist_ok=True)
    
    def _init_video_writer(self):
        """비디오 writer 초기화"""
        if not self.controller:
            return
        
        # 첫 프레임으로 비디오 크기 결정
        try:
            event = self.controller.last_event
            if hasattr(event, 'cv2img') and event.cv2img is not None:
                frame = event.cv2img
                self.video_height, self.video_width = frame.shape[:2]
            elif hasattr(event, 'frame') and event.frame is not None:
                frame = event.frame
                self.video_height, self.video_width = frame.shape[:2]
            elif hasattr(event, 'events') and len(event.events) > self.agent_id:
                agent_event = event.events[self.agent_id]
                if hasattr(agent_event, 'cv2img') and agent_event.cv2img is not None:
                    frame = agent_event.cv2img
                    self.video_height, self.video_width = frame.shape[:2]
                elif hasattr(agent_event, 'frame') and agent_event.frame is not None:
                    frame = agent_event.frame
                    self.video_height, self.video_width = frame.shape[:2]
            else:
                # 기본값 사용
                self.video_width = 1000
                self.video_height = 1000
        except Exception as e:
            print(f"⚠️ Could not determine video size: {e}, using default 1000x1000")
            self.video_width = 1000
            self.video_height = 1000
        
        # 비디오 파일명 생성 (타임스탬프 포함)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"execution_{self.scene}_{timestamp}.mp4"
        video_path = os.path.join(self.video_dir, video_filename)
        
        # VideoWriter 초기화 (MP4V 코덱 사용)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            video_path,
            fourcc,
            self.video_fps,
            (self.video_width, self.video_height)
        )
        
        if not self.video_writer.isOpened():
            print(f"⚠️ Failed to initialize video writer")
            self.video_writer = None
        else:
            print(f"✓ Video writer initialized: {video_path} (FPS: {self.video_fps})")
            self.video_path = video_path
    
    def _init_realtime_visualization(self):
        """실시간 시각화 초기화 (Third Party View & Agent View)"""
        if not MATPLOTLIB_AVAILABLE or not self.controller:
            return
        
        try:
            # Interactive mode 활성화
            plt.ion()
            
            # 하나의 창에 두 개의 subplot 생성
            self.fig = plt.figure(figsize=(20, 8))
            self.fig.canvas.manager.set_window_title("Agent View & Third Party Camera View")
            
            # 서드파티 카메라 뷰 (왼쪽)
            self.ax_camera = self.fig.add_subplot(121)
            self.ax_camera.axis('off')
            self.ax_camera.set_title("Third Party Camera View", fontsize=14, fontweight='bold')
            
            # 에이전트 시야 뷰 (오른쪽)
            self.ax_agent = self.fig.add_subplot(122)
            self.ax_agent.axis('off')
            self.ax_agent.set_title("Agent View", fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.show(block=False)  # 창을 즉시 표시
            
            # 초기 이미지 표시
            self._update_realtime_visualization()
            
            print(f"  ✓ 실시간 시각화 초기화 완료")
        except Exception as e:
            print(f"  ⚠️ 실시간 시각화 초기화 실패: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_realtime_visualization(self):
        """실시간 시각화 업데이트 (Third Party View & Agent View)"""
        if not self.show_realtime_view or not MATPLOTLIB_AVAILABLE or not self.controller or not self.fig:
            return
        
        try:
            event = self.controller.last_event
            
            # 서드파티 카메라 이미지 업데이트 (third_party_camera_frames 사용)
            if hasattr(event, 'third_party_camera_frames') and event.third_party_camera_frames:
                img_data = event.third_party_camera_frames[0]  # 첫 번째 서드파티 카메라 사용
                if img_data is not None:
                    if isinstance(img_data, np.ndarray):
                        img = img_data
                    else:
                        img = np.array(img_data)
                    
                    # 이미지 형식 확인 및 변환 (RGB 형식이므로 그대로 사용)
                    if len(img.shape) == 3 and img.shape[2] == 4:
                        img = img[:, :, :3]
                    
                    # 서드파티 카메라 이미지 업데이트
                    if self.img_display is None:
                        self.img_display = self.ax_camera.imshow(img)
                        self.ax_camera.set_title("Third Party Camera View", fontsize=14, fontweight='bold')
                    else:
                        self.img_display.set_data(img)
                        self.img_display.set_clim(vmin=img.min(), vmax=img.max())
            
            # 에이전트 시야 이미지 업데이트
            agent_img = event.frame
            if agent_img is not None:
                if isinstance(agent_img, np.ndarray):
                    agent_img_array = agent_img
                else:
                    agent_img_array = np.array(agent_img)
                
                if len(agent_img_array.shape) == 3 and agent_img_array.shape[2] == 4:
                    agent_img_array = agent_img_array[:, :, :3]
                
                if self.agent_img_display is None:
                    self.agent_img_display = self.ax_agent.imshow(agent_img_array)
                    self.ax_agent.set_title("Agent View", fontsize=14, fontweight='bold')
                else:
                    self.agent_img_display.set_data(agent_img_array)
                    self.agent_img_display.set_clim(vmin=agent_img_array.min(), vmax=agent_img_array.max())
            
            # 화면 업데이트
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.01)  # 화면 업데이트
        except Exception as e:
            # 시각화 업데이트 실패는 조용히 무시 (너무 많은 에러 메시지 방지)
            pass
    
    def _capture_frame(self):
        """현재 프레임을 캡처하여 비디오에 추가 및 실시간 시각화 업데이트"""
        # 실시간 시각화 업데이트
        if self.show_realtime_view:
            self._update_realtime_visualization()
        
        # 비디오 저장
        if not self.save_video or not self.controller:
            return
        
        try:
            event = self.controller.last_event
            
            # Agent view 프레임 가져오기
            frame = None
            if hasattr(event, 'cv2img') and event.cv2img is not None:
                frame = event.cv2img.copy()
            elif hasattr(event, 'frame') and event.frame is not None:
                frame = event.frame.copy()
            elif hasattr(event, 'events') and len(event.events) > self.agent_id:
                agent_event = event.events[self.agent_id]
                if hasattr(agent_event, 'cv2img') and agent_event.cv2img is not None:
                    frame = agent_event.cv2img.copy()
                elif hasattr(agent_event, 'frame') and agent_event.frame is not None:
                    frame = agent_event.frame.copy()
            
            if frame is not None:
                # 프레임 크기 조정 (비디오 크기에 맞춤)
                if frame.shape[:2] != (self.video_height, self.video_width):
                    frame = cv2.resize(frame, (self.video_width, self.video_height))
                
                # 비디오에 프레임 추가
                if self.video_writer and self.video_writer.isOpened():
                    self.video_writer.write(frame)
                    self.video_frames.append(frame)
        except Exception as e:
            print(f"⚠️ Error capturing frame: {e}")
    
    def _close_video_writer(self):
        """비디오 writer 종료 및 파일 저장"""
        if self.video_writer and self.video_writer.isOpened():
            self.video_writer.release()
            self.video_writer = None
            if hasattr(self, 'video_path'):
                print(f"✓ Video saved: {self.video_path} ({len(self.video_frames)} frames)")
            self.video_frames = []
    
    def initialize(self):
        """AI2-THOR ManipulaTHOR 환경 초기화"""
        self.controller = Controller(height=1000, width=1000, headless=self.headless)
        self.controller.reset(self.scene)
        
        # Agent 초기화 (ManipulaTHOR는 agentMode="arm" 사용)
        self.controller.step(dict(
            action='Initialize',
            agentMode="arm",  # ManipulaTHOR는 arm 모드
            snapGrid=False,
            gridSize=0.25,
            rotateStepDegrees=20,
            visibilityDistance=1.5,
            fieldOfView=120,
            agentCount=1,
            handSphereRadius=0.2  # handSphereRadius 설정
        ))
        
        # Top view camera 추가
        event = self.controller.step(action="GetMapViewCameraProperties")
        self.controller.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])
        
        # 방 구석에서 로봇을 볼 수 있도록 추가 Third Party Camera 추가 (use_arm_and_armbase.py 방식)
        try:
            # 에이전트 위치 가져오기
            agent_pos = self.controller.last_event.metadata.get("agent", {}).get("position", {})
            agent_x = agent_pos.get("x", 0)
            agent_y = agent_pos.get("y", 0.9)
            agent_z = agent_pos.get("z", 0)
            
            # 방 구석에서 에이전트를 볼 수 있도록 카메라 위치 설정
            # 에이전트 앞쪽에서 위에서 보기
            self.controller.step(
                action="AddThirdPartyCamera",
                position=dict(x=agent_x, y=agent_y + 0.5, z=agent_z - 1.5),  # 에이전트 앞쪽에서 위에서 보기
                rotation=dict(x=15, y=0, z=0),  # 약간 아래를 보도록
                fieldOfView=90
            )
            print(f"  ✓ Third Party Camera 추가 완료 (방 구석 뷰)")
        except Exception as e:
            print(f"  ⚠️ Third Party Camera 추가 실패: {e}")
        
        # 도달 가능한 위치 가져오기 (NavMesh 기반)
        reachable_positions_ = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        self.reachable_positions = [(p["x"], p["y"], p["z"]) for p in reachable_positions_]
        print(f"  ✓ Found {len(self.reachable_positions)} reachable positions")
        
        # Agent 초기 위치 설정
        init_pos = self._select_initial_position()
        if init_pos:
            self.controller.step(dict(action="Teleport", position=init_pos, agentId=self.agent_id))
            print(f"  ✓ Agent initialized at position: ({init_pos['x']:.2f}, {init_pos['y']:.2f}, {init_pos['z']:.2f})")
        else:
            print(f"  ⚠ No valid initial position found, agent will remain at default position")
        
        # Agent 시야 조정
        self.controller.step(action="LookDown", degrees=35, agentId=self.agent_id)
        
        # 비디오 writer 초기화
        if self.save_video:
            self._init_video_writer()
        
        # 실시간 시각화 초기화 (헤드리스 모드가 아닐 때만)
        if self.show_realtime_view and MATPLOTLIB_AVAILABLE:
            self._init_realtime_visualization()
        
        print(f"✓ AI2-THOR ManipulaTHOR initialized: {self.scene}")
    
    def _select_initial_position(self) -> Optional[Dict[str, float]]:
        """
        초기 위치 선택 (사용자 지정 또는 전략 기반)
        
        Returns:
            초기 위치 딕셔너리 {"x": float, "y": float, "z": float} 또는 None
        """
        if not self.reachable_positions:
            return None
        
        # 사용자가 직접 위치를 지정한 경우
        if self.initial_position_strategy == "custom" and self.initial_position:
            # 지정된 위치가 도달 가능한 위치 중에 있는지 확인
            x, y, z = self.initial_position
            # 가장 가까운 도달 가능한 위치 찾기
            closest = min(
                self.reachable_positions,
                key=lambda p: distance_pts([x, y, z], list(p))
            )
            dist = distance_pts([x, y, z], list(closest))
            if dist < 0.5:  # 0.5m 이내면 사용
                return {"x": closest[0], "y": closest[1], "z": closest[2]}
            else:
                print(f"  ⚠ Custom position ({x:.2f}, {y:.2f}, {z:.2f}) is not reachable, using closest: {dist:.2f}m away")
                return {"x": closest[0], "y": closest[1], "z": closest[2]}
        
        # 전략 기반 선택
        if self.initial_position_strategy == "random":
            import random
            pos = random.choice(self.reachable_positions)
            return {"x": pos[0], "y": pos[1], "z": pos[2]}
        
        elif self.initial_position_strategy == "center":
            # 모든 도달 가능한 위치의 중심 계산
            if not self.reachable_positions:
                return None
            
            center_x = sum(p[0] for p in self.reachable_positions) / len(self.reachable_positions)
            center_y = sum(p[1] for p in self.reachable_positions) / len(self.reachable_positions)
            center_z = sum(p[2] for p in self.reachable_positions) / len(self.reachable_positions)
            center = [center_x, center_y, center_z]
            
            # 중심에 가장 가까운 위치 선택
            closest = min(
                self.reachable_positions,
                key=lambda p: distance_pts(center, list(p))
            )
            return {"x": closest[0], "y": closest[1], "z": closest[2]}
        
        else:  # "first" (기본값)
            pos = self.reachable_positions[0]
            return {"x": pos[0], "y": pos[1], "z": pos[2]}
    
    def _find_object_id(self, object_name: str) -> Optional[str]:
        """
        객체 이름, objectId, 또는 좌표로 objectId 찾기
        
        Args:
            object_name: 
                - 객체 이름 (예: "Cabinet")
                - objectId (예: "Cabinet_123")
                - 좌표 포함 형식 (예: "Cabinet|+00.72|+02.02|-02.46")
            
        Returns:
            objectId 또는 None
        """
        all_objects = self.controller.last_event.metadata.get("objects", [])
        objs = [obj["objectId"] for obj in all_objects]
        
        # objectId가 직접 주어진 경우 (정확한 매칭)
        if object_name in objs:
            return object_name
        
        # 좌표 포함 형식 파싱 (예: "Cabinet|+00.72|+02.02|-02.46" 또는 "Cabinet|+0.72|+2.02|-2.46")
        if "|" in object_name:
            parts = object_name.split("|")
            if len(parts) >= 4:  # 타입|좌표1|좌표2|좌표3
                obj_type = parts[0]
                try:
                    # 좌표 문자열 파싱 (+00.72 형식 또는 +0.72 형식 모두 지원)
                    x_str = parts[1].strip()
                    y_str = parts[2].strip()
                    z_str = parts[3].strip()
                    target_x = float(x_str)
                    target_y = float(y_str)
                    target_z = float(z_str)
                    
                    # 좌표로 가장 가까운 객체 찾기
                    closest_obj = None
                    min_distance = float('inf')
                    
                    for obj in all_objects:
                        obj_type_normalized = normalize_object_type(obj.get("objectType", ""))
                        if obj_type_normalized.lower() != obj_type.lower():
                            continue
                        
                        # 객체의 중심 위치 가져오기
                        bbox = obj.get("axisAlignedBoundingBox", {})
                        center = bbox.get("center", {})
                        if not center:
                            continue
                        
                        obj_x = center.get("x", 0)
                        obj_y = center.get("y", 0)
                        obj_z = center.get("z", 0)
                        
                        # 유클리드 거리 계산
                        distance = math.sqrt(
                            (obj_x - target_x)**2 + 
                            (obj_y - target_y)**2 + 
                            (obj_z - target_z)**2
                        )
                        
                        if distance < min_distance:
                            min_distance = distance
                            closest_obj = obj
                    
                    if closest_obj and min_distance < 0.5:  # 0.5m 이내면 매칭
                        print(f"  ✓ Found {obj_type} at coordinates ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})")
                        return closest_obj.get("objectId")
                    elif closest_obj:
                        print(f"  ⚠ Closest {obj_type} found at distance {min_distance:.2f}m from target coordinates")
                        return closest_obj.get("objectId")
                except ValueError:
                    # 좌표 파싱 실패 시 objectId로 처리 (예: "Cabinet|Cabinet_123")
                    object_id_candidate = parts[-1]  # 마지막 부분이 objectId일 수 있음
                    if object_id_candidate in objs:
                        return object_id_candidate
                    # objectId가 아니면 타입 이름만 사용
                    object_name = parts[0]
        
        # 정확한 매칭 먼저 시도
        for obj_id in objs:
            if object_name.lower() in obj_id.lower() or obj_id.lower().startswith(object_name.lower()):
                return obj_id
        
        # 정규식 매칭
        for obj_id in objs:
            if re.match(object_name, obj_id, re.IGNORECASE):
                return obj_id
        
        return None
    
    def _get_object_center(self, object_id: str) -> Optional[Dict[str, float]]:
        """객체의 중심 위치 가져오기"""
        for obj in self.controller.last_event.metadata["objects"]:
            if obj["objectId"] == object_id:
                return obj.get("axisAlignedBoundingBox", {}).get("center")
        return None
    
    def _find_closest_reachable_position_to_target(self, target_pos: Dict[str, float]) -> Tuple[Optional[Dict[str, float]], float]:
        """
        NavMesh에서 목표 위치까지 가장 가까운 이동 가능 위치 찾기
        실제 3D 거리를 계산하여 반환 (goto_object의 거리 검증과 일치)
        
        Args:
            target_pos: 목표 위치 {"x": float, "y": float, "z": float}
            
        Returns:
            (가장 가까운 이동 가능 위치, 실제 3D 거리) 또는 (None, float('inf'))
        """
        if not self.controller:
            return None, float('inf')
        
        try:
            # GetReachablePositions로 최신 이동 가능한 위치들 가져오기
            event = self.controller.step(action="GetReachablePositions")
            reachable_positions = event.metadata.get("actionReturn", [])
            
            if not reachable_positions:
                return None, float('inf')
            
            # 목표 위치 좌표
            target_x = target_pos.get("x", 0)
            target_y = target_pos.get("y", 0)
            target_z = target_pos.get("z", 0)
            target_list = [target_x, target_y, target_z]
            
            closest_pos = None
            min_distance_2d = float('inf')  # x, z 평면에서 가장 가까운 위치 선택
            min_distance_3d = float('inf')  # 실제 3D 거리 (반환값)
            
            for pos in reachable_positions:
                pos_x = pos.get("x", 0)
                pos_y = pos.get("y", 0)
                pos_z = pos.get("z", 0)
                pos_list = [pos_x, pos_y, pos_z]
                
                # x, z 평면에서의 거리 계산 (위치 선택 기준)
                distance_2d = ((pos_x - target_x)**2 + (pos_z - target_z)**2) ** 0.5
                # 실제 3D 거리 계산 (반환값)
                distance_3d = distance_pts(pos_list, target_list)
                
                # x, z 평면에서 가장 가까운 위치 선택 (Agent는 바닥에 있으므로)
                if distance_2d < min_distance_2d:
                    min_distance_2d = distance_2d
                    min_distance_3d = distance_3d
                    closest_pos = {
                        "x": pos_x,
                        "y": pos_y,
                        "z": pos_z
                    }
            
            # 실제 3D 거리 반환 (goto_object의 거리 검증과 일치)
            return closest_pos, min_distance_3d
        except Exception as e:
            print(f"  ⚠ NavMesh에서 이동 가능 위치 찾기 실패: {e}")
            return None, float('inf')
    
    def _retract_arm_for_navigation(self):
        """
        이동을 위해 팔을 접어서 경로를 막지 않도록 함
        MoveArmBase를 낮추고 MoveArm을 몸쪽으로 접기
        """
        try:
            print(f"  Retracting arm for navigation...")
            
            # 1. MoveArmBase를 낮춤 (normalizedY=0.0은 가장 낮은 위치)
            event = self.controller.step(
                action="MoveArmBase",
                y=0.7,
                normalizedY=True,
                agentId=self.agent_id
            )
            self._capture_frame()
            time.sleep(0.2)
            
            if not event.metadata.get('lastActionSuccess', False):
                print(f"  ⚠ MoveArmBase failed: {event.metadata.get('errorMessage', 'Unknown error')}")
            
            # 2. MoveArm을 몸쪽으로 접기 (x=0, y=0, z=0은 몸쪽 위치)
            event = self.controller.step(
                action="MoveArm",
                position={"x": 0, "y": 0, "z": 0},
                coordinateSpace="armBase",
                agentId=self.agent_id
            )
            self._capture_frame()
            time.sleep(0.2)
            
            if not event.metadata.get('lastActionSuccess', False):
                print(f"  ⚠ MoveArm failed: {event.metadata.get('errorMessage', 'Unknown error')}")
            else:
                print(f"  ✓ Arm retracted for navigation")
        except Exception as e:
            print(f"  ⚠ Arm retraction failed: {e}")
    
    def goto_object(self, object_name: str, target_distance: float = 1.0, max_steps: int = 50) -> bool:
        """
        객체로 이동 - NavMesh상 가장 가까운 위치로 이동
        
        Args:
            object_name: 목표 객체 이름
            target_distance: 목표 거리 (미터, 기본값: 1.0m)
            max_steps: 최대 이동 시도 횟수 (기본값: 50)
            
        Returns:
            성공 여부
        """
        # 이동 전에 팔을 접어서 경로를 막지 않도록 함
        self._retract_arm_for_navigation()
        
        object_id = self._find_object_id(object_name)
        if not object_id:
            print(f"✗ Object '{object_name}' not found")
            return False
        
        obj_center = self._get_object_center(object_id)
        if not obj_center or obj_center == {'x': 0.0, 'y': 0.0, 'z': 0.0}:
            print(f"✗ Object '{object_name}' has invalid position")
            return False
        
        print(f"  Finding closest reachable position to '{object_name}' (position: {obj_center})...")
        
        # NavMesh에서 목표 객체까지 가장 가까운 이동 가능 위치 찾기
        closest_pos, distance = self._find_closest_reachable_position_to_target(obj_center)
        
        if closest_pos is None:
            print(f"  ✗ Could not find reachable position near '{object_name}'")
            return False
        
        print(f"  ✓ Found closest reachable position: ({closest_pos['x']:.3f}, {closest_pos['y']:.3f}, {closest_pos['z']:.3f}), distance: {distance:.3f}m")
        
        # 현재 agent 위치
        metadata = self.controller.last_event.metadata
        current_pos = metadata["agent"]["position"]
        current_pos_list = [current_pos['x'], current_pos['y'], current_pos['z']]
        closest_pos_list = [closest_pos['x'], closest_pos['y'], closest_pos['z']]
        
        # 이미 목표 위치에 가까이 있으면 성공
        current_distance = distance_pts(current_pos_list, closest_pos_list)
        if current_distance < 0.1:  # 0.1m 이내면 이미 도착
            print(f"  ✓ Already at closest reachable position (distance: {current_distance:.3f}m)")
            # 객체를 향해 회전
            robot_rot = metadata["agent"]["rotation"]["y"]
            dest_pos = [obj_center['x'], obj_center['y'], obj_center['z']]
            self._rotate_towards_object(dest_pos, current_pos, robot_rot)
            return True
        
        # 가장 가까운 위치까지 실제로 이동 (ObjectNavExpertAction 사용)
        # ObjectNavExpertAction은 AI2-THOR의 내장 경로 탐색으로 NavMesh를 자동으로 따름
        print(f"  Moving to closest reachable position...")
        
        movement_success = self._goto_object_with_objectnav(
            object_name, 
            tuple(closest_pos_list), 
            [obj_center['x'], obj_center['y'], obj_center['z']], 
            target_distance, 
            max_steps
        )
        
        # 최종 결과 반환
        return movement_success
    
    def _is_straight_path(self, current_pos: List[float], goal_pos: List[float], robot_rot: float, angle_threshold: float = 15.0) -> bool:
        """
        현재 위치에서 목표까지의 경로가 직선인지 확인
        
        Args:
            current_pos: 현재 위치 [x, y, z]
            goal_pos: 목표 위치 [x, y, z]
            robot_rot: 로봇의 현재 회전 각도 (도)
            angle_threshold: 직선으로 간주할 최대 각도 차이 (도, 기본값: 15도)
            
        Returns:
            직선 경로면 True, 아니면 False
        """
        # 목표까지의 벡터 계산 (x, z 평면)
        dx = goal_pos[0] - current_pos[0]
        dz = goal_pos[2] - current_pos[2]
        
        if abs(dx) < 0.01 and abs(dz) < 0.01:
            return True  # 이미 목표 위치에 매우 가까움
        
        # 목표 방향의 각도 계산
        target_angle = math.degrees(math.atan2(dx, dz))
        target_angle = (target_angle + 360) % 360
        
        # 로봇의 현재 방향과 목표 방향의 차이
        angle_diff = abs(target_angle - robot_rot)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        # 각도 차이가 임계값 이하면 직선 경로
        return angle_diff <= angle_threshold
    
    def _goto_object_with_objectnav(self, object_name: str, closest_pos: Tuple[float, float, float], dest_pos: List[float], target_distance: float, max_steps: int) -> bool:
        """
        ObjectNavExpertAction을 사용한 이동
        - 직선 경로에서는 직진만 수행 (회전 액션 필터링)
        - 코너에서는 넓게 돌도록 안전거리 확보
        """
        # 제공된 코드 방식: clost_node_location을 리스트로 관리 (단일 agent이므로 [0] 사용)
        clost_node_location = [0]  # 대체 reachable position 인덱스
        goal_thresh = 0.3  # 제공된 코드: goal_thresh = 0.3 (고정값)
        
        obj_label = f"'{object_name}'" if object_name else "target"
        print(f"  Moving towards {obj_label} (goal_thresh: {goal_thresh}m, target_object_distance: {target_distance}m)...")
        
        # 제공된 코드 방식: closest_node 함수 사용하여 reachable position 선택
        # 단일 agent이므로 no_agents=1
        # 현재 위치에서 도달 가능한 reachable position만 사용
        current_metadata = self.controller.last_event.metadata
        current_agent_pos = current_metadata["agent"]["position"]
        current_agent_pos_list = [current_agent_pos['x'], current_agent_pos['y'], current_agent_pos['z']]
        
        # 현재 위치에서 도달 가능한 reachable position 필터링 (너무 먼 위치 제외)
        # NavMesh를 통해 도달 가능한 위치만 사용 (최대 5m 이내)
        reachable_from_current = []
        for rp in self.reachable_positions:
            dist_from_current = distance_pts(current_agent_pos_list, list(rp))
            if dist_from_current <= 5.0:  # 5m 이내의 reachable position만 고려
                reachable_from_current.append(rp)
        
        # 도달 가능한 위치가 없으면 모든 위치 사용
        if not reachable_from_current:
            reachable_from_current = self.reachable_positions
        
        crp_list = closest_node(dest_pos, reachable_from_current, 1, clost_node_location)
        current_closest_pos = crp_list[0] if crp_list else closest_pos
        
        print(f"  Selected reachable position: ({current_closest_pos[0]:.3f}, {current_closest_pos[1]:.3f}, {current_closest_pos[2]:.3f})")
        
        # 거리 추적 (제공된 코드 방식)
        dist_goal = 10.0
        prev_dist_goal = 10.0
        count_since_update = 0
        iteration_count = 0  # 무한 루프 방지
        max_iterations = 200  # 최대 반복 횟수
        
        # 직진 경로 추적을 위한 변수
        consecutive_rotations = 0  # 연속된 회전 횟수
        last_action_type = None  # 마지막 액션 타입
        prev_position = None  # 이전 위치
        
        # 제공된 코드 방식: while all(d > goal_thresh for d in dist_goals)
        # 단일 agent이므로 dist_goal > goal_thresh 조건 사용
        while dist_goal > goal_thresh and iteration_count < max_iterations:
            iteration_count += 1
            # 현재 agent 위치
            metadata = self.controller.last_event.metadata
            robot_pos = metadata["agent"]["position"]
            robot_rot = metadata["agent"]["rotation"]["y"]
            
            current_pos = [robot_pos['x'], robot_pos['y'], robot_pos['z']]
            # 제공된 코드 방식: dist_goals[ia] = distance_pts([location['x'], location['y'], location['z']], crp[ia])
            prev_dist_goal = dist_goal
            dist_goal = distance_pts(current_pos, list(current_closest_pos))
            
            # 제공된 코드 방식: dist_del = abs(dist_goals[ia] - prev_dist_goals[ia])
            dist_del = abs(dist_goal - prev_dist_goal)
            print(f"  Dist to Goal: {dist_goal:.2f}m, change: {dist_del:.2f}m, node_idx: {clost_node_location[0]}")
            
            # 제공된 코드 방식: if dist_del < 0.2: count_since_update[ia] += 1
            if dist_del < 0.2:
                # 로봇이 이동하지 않음
                count_since_update += 1
            else:
                # 로봇이 이동 중
                count_since_update = 0
            
            # 제공된 코드 방식: if count_since_update[ia] < 15: ObjectNavExpertAction
            if count_since_update < 15:
                # 직선 경로인지 확인
                is_straight = self._is_straight_path(current_pos, list(current_closest_pos), robot_rot, angle_threshold=15.0)
                
                # ObjectNavExpertAction 사용 (AI2-THOR의 내장 경로 탐색, NavMesh 자동 따름)
                event = self.controller.step(dict(
                    action='ObjectNavExpertAction',
                    position=dict(x=current_closest_pos[0], y=current_closest_pos[1], z=current_closest_pos[2]),
                    agentId=self.agent_id
                ))
                self._capture_frame()
                
                # ObjectNavExpertAction 성공 여부 확인
                action_success = event.metadata.get('lastActionSuccess', False)
                if not action_success:
                    error_msg = event.metadata.get('errorMessage', 'Unknown error')
                    print(f"  ⚠ ObjectNavExpertAction failed: {error_msg}")
                else:
                    # ObjectNavExpertAction이 반환한 다음 액션 실행
                    next_action = event.metadata.get('actionReturn')
                    if next_action:
                        # 액션 타입 확인
                        action_name = None
                        if isinstance(next_action, str):
                            action_name = next_action
                        elif isinstance(next_action, dict):
                            action_name = next_action.get('action', 'Unknown')
                        
                        # 직선 경로이고 회전 액션이면 스킵 (직진만 수행)
                        if is_straight and action_name in ['RotateLeft', 'RotateRight']:
                            # 직선 경로에서는 회전을 무시하고 MoveAhead만 수행
                            print(f"  → 직선 경로: 회전 액션 스킵, 직진만 수행")
                            move_ahead_event = self.controller.step(
                                action='MoveAhead',
                                agentId=self.agent_id,
                                forceAction=True
                            )
                            self._capture_frame()
                            if move_ahead_event.metadata.get('lastActionSuccess', False):
                                last_action_type = 'MoveAhead'
                                consecutive_rotations = 0
                            else:
                                # MoveAhead 실패 시 원래 액션 실행 (장애물이 있을 수 있음)
                                print(f"  ⚠ MoveAhead 실패, 원래 액션 실행")
                                action_success = self.controller.step(
                                    action=next_action,
                                    agentId=self.agent_id,
                                    forceAction=True
                                )
                                self._capture_frame()
                                if action_name in ['RotateLeft', 'RotateRight']:
                                    last_action_type = 'rotation'
                                    consecutive_rotations += 1
                                else:
                                    last_action_type = action_name
                                    consecutive_rotations = 0
                        else:
                            # 코너를 돌거나 직선이 아닌 경우
                            if action_name in ['RotateLeft', 'RotateRight']:
                                # 코너를 넓게 돌기 위해 회전 각도를 조정
                                if isinstance(next_action, dict):
                                    degrees = next_action.get('degrees', 90)
                                    # 코너를 넓게 돌기 위해 회전 각도를 약간 줄임 (더 부드러운 회전)
                                    adjusted_degrees = max(degrees * 0.8, 10)  # 최소 10도
                                    next_action = dict(
                                        action=action_name,
                                        degrees=adjusted_degrees,
                                        agentId=self.agent_id
                                    )
                                    print(f"  → 코너 회전: {action_name} {adjusted_degrees:.1f}도 (원래: {degrees}도)")
                                else:
                                    print(f"  → 코너 회전: {action_name}")
                                
                                last_action_type = 'rotation'
                                consecutive_rotations += 1
                            else:
                                print(f"  → Executing action: {action_name}")
                                last_action_type = action_name
                                consecutive_rotations = 0
                            
                            # 연속된 회전이 3회 이상이면 경로 재계산 (막혔을 수 있음)
                            if consecutive_rotations >= 3:
                                print(f"  ⚠ 연속 회전 {consecutive_rotations}회 감지, 경로 재계산")
                                count_since_update = 15  # 목표 업데이트 트리거
                                consecutive_rotations = 0
                            else:
                                action_success = self.controller.step(
                                    action=next_action,
                                    agentId=self.agent_id,
                                    forceAction=True
                                )
                                self._capture_frame()
                                
                                if not action_success.metadata.get('lastActionSuccess', False):
                                    error_msg = action_success.metadata.get('errorMessage', 'Unknown error')
                                    print(f"  ⚠ Next action '{action_name}' failed: {error_msg}")
                    else:
                        # ObjectNavExpertAction이 다음 액션을 반환하지 않음
                        if dist_goal <= goal_thresh:
                            break
                        else:
                            count_since_update += 1
                            print(f"  ⚠ ObjectNavExpertAction returned no next action (dist: {dist_goal:.2f}m)")
                
                # 위치 업데이트 (직선 경로 판단용)
                if action_success:
                    new_metadata = self.controller.last_event.metadata
                    new_pos = new_metadata["agent"]["position"]
                    new_pos_list = [new_pos['x'], new_pos['y'], new_pos['z']]
                    if prev_position is None or distance_pts(new_pos_list, prev_position) > 0.1:
                        prev_position = new_pos_list
            else:
                # 제공된 코드 방식: updating goal
                clost_node_location[0] += 1
                count_since_update = 0
                
                # 현재 위치에서 도달 가능한 reachable position 재계산
                current_metadata = self.controller.last_event.metadata
                current_agent_pos = current_metadata["agent"]["position"]
                current_agent_pos_list = [current_agent_pos['x'], current_agent_pos['y'], current_agent_pos['z']]
                
                reachable_from_current = []
                for rp in self.reachable_positions:
                    dist_from_current = distance_pts(current_agent_pos_list, list(rp))
                    if dist_from_current <= 5.0:  # 5m 이내의 reachable position만 고려
                        reachable_from_current.append(rp)
                
                if not reachable_from_current:
                    reachable_from_current = self.reachable_positions
                
                # 새로운 reachable position 선택
                crp_list = closest_node(dest_pos, reachable_from_current, 1, clost_node_location)
                if crp_list:
                    current_closest_pos = crp_list[0]
                    print(f"  ⚠ Stuck, updating goal to reachable position #{clost_node_location[0]}: ({current_closest_pos[0]:.3f}, {current_closest_pos[1]:.3f}, {current_closest_pos[2]:.3f})")
                else:
                    # 모든 reachable position을 시도했으면 처음부터 다시
                    clost_node_location[0] = 0
                    crp_list = closest_node(dest_pos, reachable_from_current, 1, clost_node_location)
                    if crp_list:
                        current_closest_pos = crp_list[0]
                    print(f"  ⚠ All reachable positions tried, resetting to first position...")
            
            # 제공된 코드 방식: time.sleep(0.5)
            time.sleep(0.5)
        
        # 무한 루프 방지: 최대 반복 횟수 도달 시
        if iteration_count >= max_iterations:
            print(f"  ⚠ Maximum iterations ({max_iterations}) reached. Current distance to goal: {dist_goal:.2f}m")
            # 최소한 객체를 향해 회전
            metadata = self.controller.last_event.metadata
            robot_pos = metadata["agent"]["position"]
            robot_rot = metadata["agent"]["rotation"]["y"]
            robot_object_vec = [dest_pos[0] - robot_pos['x'], dest_pos[2] - robot_pos['z']]
            y_axis = [0, 1]
            unit_y = np.array(y_axis) / np.linalg.norm(y_axis)
            unit_vector = np.array(robot_object_vec) / np.linalg.norm(robot_object_vec)
            angle = math.atan2(np.linalg.det([unit_vector, unit_y]), np.dot(unit_vector, unit_y))
            angle = 360 * angle / (2 * math.pi)
            angle = (angle + 360) % 360
            rot_angle = angle - robot_rot
            if rot_angle > 0:
                self.controller.step(action="RotateRight", degrees=abs(rot_angle), agentId=self.agent_id)
            else:
                self.controller.step(action="RotateLeft", degrees=abs(rot_angle), agentId=self.agent_id)
            self._capture_frame()
            return False
        
        # 제공된 코드 방식: align the robot once goal is reached
        # 목표 reachable position에 도달했으므로 객체를 향해 회전
        metadata = self.controller.last_event.metadata
        robot_pos = metadata["agent"]["position"]
        robot_rot = metadata["agent"]["rotation"]["y"]
        current_pos_list = [robot_pos['x'], robot_pos['y'], robot_pos['z']]
        
        # 실제 객체까지의 거리 확인
        distance_to_object = distance_pts(current_pos_list, dest_pos)
        print(f"  Reached reachable position. Distance to object: {distance_to_object:.2f}m (target: {target_distance:.2f}m)")
        
        # 제공된 코드 방식: compute angle between robot heading and object
        robot_object_vec = [dest_pos[0] - robot_pos['x'], dest_pos[2] - robot_pos['z']]
        y_axis = [0, 1]
        unit_y = np.array(y_axis) / np.linalg.norm(y_axis)
        unit_vector = np.array(robot_object_vec) / np.linalg.norm(robot_object_vec)
        
        angle = math.atan2(np.linalg.det([unit_vector, unit_y]), np.dot(unit_vector, unit_y))
        angle = 360 * angle / (2 * math.pi)
        angle = (angle + 360) % 360
        rot_angle = angle - robot_rot
        
        # 제공된 코드 방식: 회전 실행
        if rot_angle > 0:
            self.controller.step(action="RotateRight", degrees=abs(rot_angle), agentId=self.agent_id)
        else:
            self.controller.step(action="RotateLeft", degrees=abs(rot_angle), agentId=self.agent_id)
        self._capture_frame()
        
        # 객체까지의 거리가 target_distance 이내면 성공
        if distance_to_object <= target_distance:
            print(f"  ✓ Reached: {object_name} (distance: {distance_to_object:.2f}m)")
            return True
        else:
            print(f"  ⚠ Reached reachable position but still {distance_to_object:.2f}m away from {object_name}")
            # 객체에 가까워지기 위해 추가 이동 시도
            # 가장 가까운 reachable position으로 추가 이동
            closest_to_object = min(
                self.reachable_positions,
                key=lambda p: distance_pts(dest_pos, list(p))
            )
            closest_dist_to_object = distance_pts(dest_pos, list(closest_to_object))
            
            if closest_dist_to_object < distance_to_object:
                print(f"  → Attempting to move closer to object...")
                # 추가 이동 시도 (최대 10스텝)
                for _ in range(10):
                    event = self.controller.step(dict(
                        action='ObjectNavExpertAction',
                        position=dict(x=closest_to_object[0], y=closest_to_object[1], z=closest_to_object[2]),
                        agentId=self.agent_id
                    ))
                    self._capture_frame()
                    
                    next_action = event.metadata.get('actionReturn')
                    if next_action:
                        self.controller.step(action=next_action, agentId=self.agent_id, forceAction=True)
                        self._capture_frame()
                    
                    # 거리 재확인
                    current_metadata = self.controller.last_event.metadata
                    current_pos = current_metadata["agent"]["position"]
                    current_pos_list = [current_pos['x'], current_pos['y'], current_pos['z']]
                    current_dist = distance_pts(current_pos_list, dest_pos)
                    
                    if current_dist <= target_distance:
                        print(f"  ✓ Reached: {object_name} (distance: {current_dist:.2f}m)")
                        return True
                    
                    time.sleep(0.1)
            
            # 최종 확인
            final_metadata = self.controller.last_event.metadata
            final_pos = final_metadata["agent"]["position"]
            final_pos_list = [final_pos['x'], final_pos['y'], final_pos['z']]
            final_dist = distance_pts(final_pos_list, dest_pos)
            
            if final_dist <= target_distance:
                print(f"  ✓ Reached: {object_name} (distance: {final_dist:.2f}m)")
                return True
            else:
                print(f"  ⚠ Could not reach {object_name} within {target_distance:.2f}m (final distance: {final_dist:.2f}m)")
                return False
    
    def _is_object_visible(self, object_id: str) -> bool:
        """객체가 현재 시야에 보이는지 확인"""
        if not self.controller:
            return False
        
        for obj in self.controller.last_event.metadata.get("objects", []):
            if obj["objectId"] == object_id:
                return obj.get("visible", False)
        return False
    
    def _ensure_object_visible(self, object_id: str, object_name: str, max_rotations: int = 8) -> bool:
        """
        객체가 시야에 정확히 보이도록 회전 (객체를 정확히 향하도록, 효율적인 회전)
        
        Args:
            object_id: 객체 ID
            object_name: 객체 이름 (로그용)
            max_rotations: 최대 회전 횟수 (기본값: 8)
            
        Returns:
            객체가 보이는지 여부
        """
        if not self.controller:
            return False
        
        # 객체 위치 가져오기
        obj_center = self._get_object_center(object_id)
        if not obj_center:
            return False
        
        dest_pos = [obj_center['x'], obj_center['y'], obj_center['z']]
        
        print(f"  Rotating to face '{object_name}'...")
        
        # 먼저 객체가 이미 보이는지 확인
        if self._is_object_visible(object_id):
            # 이미 보이면 정확히 중앙에 오도록 한 번만 미세 조정
            metadata = self.controller.last_event.metadata
            robot_pos = metadata["agent"]["position"]
            robot_rot = metadata["agent"]["rotation"]["y"]
            
            robot_object_vec = [dest_pos[0] - robot_pos['x'], dest_pos[2] - robot_pos['z']]
            if np.linalg.norm(robot_object_vec) > 0.01:
                y_axis = [0, 1]
                unit_y = np.array(y_axis) / np.linalg.norm(y_axis)
                unit_vector = np.array(robot_object_vec) / np.linalg.norm(robot_object_vec)
                
                angle = math.atan2(np.linalg.det([unit_vector, unit_y]), np.dot(unit_vector, unit_y))
                angle = 360 * angle / (2 * math.pi)
                angle = (angle + 360) % 360
                rot_angle = angle - robot_rot
                
                # 각도 정규화
                if rot_angle > 180:
                    rot_angle -= 360
                elif rot_angle < -180:
                    rot_angle += 360
                
                # 5도 이상 차이면 미세 조정
                if abs(rot_angle) > 5:
                    if rot_angle > 0:
                        self.controller.step(action="RotateRight", degrees=min(abs(rot_angle), 20), agentId=self.agent_id)
                    else:
                        self.controller.step(action="RotateLeft", degrees=min(abs(rot_angle), 20), agentId=self.agent_id)
                    self._capture_frame()
                    time.sleep(0.2)
            
            print(f"  ✓ '{object_name}' is already visible")
            return True
        
        # 객체가 보이지 않으면 정확한 각도로 한 번에 회전
        metadata = self.controller.last_event.metadata
        robot_pos = metadata["agent"]["position"]
        robot_rot = metadata["agent"]["rotation"]["y"]
        
        robot_object_vec = [dest_pos[0] - robot_pos['x'], dest_pos[2] - robot_pos['z']]
        if np.linalg.norm(robot_object_vec) < 0.01:
            # 이미 목표 위치에 매우 가까움
            return self._is_object_visible(object_id)
        
        # 객체를 향한 정확한 각도 계산
        y_axis = [0, 1]
        unit_y = np.array(y_axis) / np.linalg.norm(y_axis)
        unit_vector = np.array(robot_object_vec) / np.linalg.norm(robot_object_vec)
        
        angle = math.atan2(np.linalg.det([unit_vector, unit_y]), np.dot(unit_vector, unit_y))
        angle = 360 * angle / (2 * math.pi)
        angle = (angle + 360) % 360
        rot_angle = angle - robot_rot
        
        # 각도 정규화
        if rot_angle > 180:
            rot_angle -= 360
        elif rot_angle < -180:
            rot_angle += 360
        
        # 정확한 각도로 한 번에 회전 (최대 90도씩)
        if abs(rot_angle) > 5:
            # 큰 각도는 여러 번에 나눠서 회전 (최대 90도씩)
            remaining_angle = abs(rot_angle)
            rotation_direction = "RotateRight" if rot_angle > 0 else "RotateLeft"
            
            while remaining_angle > 5 and max_rotations > 0:
                rotation_amount = min(remaining_angle, 90)  # 최대 90도씩
                self.controller.step(action=rotation_direction, degrees=rotation_amount, agentId=self.agent_id)
                self._capture_frame()
                time.sleep(0.2)
                
                # 회전 후 객체가 보이는지 확인
                if self._is_object_visible(object_id):
                    print(f"  ✓ '{object_name}' is now visible")
                    # 미세 조정
                    metadata = self.controller.last_event.metadata
                    robot_rot = metadata["agent"]["rotation"]["y"]
                    rot_angle = angle - robot_rot
                    if rot_angle > 180:
                        rot_angle -= 360
                    elif rot_angle < -180:
                        rot_angle += 360
                    
                    if abs(rot_angle) > 3:
                        if rot_angle > 0:
                            self.controller.step(action="RotateRight", degrees=min(abs(rot_angle), 10), agentId=self.agent_id)
                        else:
                            self.controller.step(action="RotateLeft", degrees=min(abs(rot_angle), 10), agentId=self.agent_id)
                        self._capture_frame()
                        time.sleep(0.15)
                    return True
                
                remaining_angle -= rotation_amount
                max_rotations -= 1
        else:
            # 이미 정확히 향하고 있음
            if self._is_object_visible(object_id):
                print(f"  ✓ '{object_name}' is already visible")
                return True
        
        # 최종 확인
        is_visible = self._is_object_visible(object_id)
        if is_visible:
            print(f"  ✓ '{object_name}' is now visible")
        else:
            print(f"  ⚠ Could not make '{object_name}' visible after rotations")
        
        return is_visible
    
    def _rotate_towards_object(self, dest_pos: List[float], robot_pos: Dict[str, float], robot_rot: float):
        """객체를 향해 회전"""
        robot_object_vec = [dest_pos[0] - robot_pos['x'], dest_pos[2] - robot_pos['z']]
        if np.linalg.norm(robot_object_vec) < 0.01:
            return
        
        y_axis = [0, 1]
        unit_y = np.array(y_axis) / np.linalg.norm(y_axis)
        unit_vector = np.array(robot_object_vec) / np.linalg.norm(robot_object_vec)
        
        angle = math.atan2(np.linalg.det([unit_vector, unit_y]), np.dot(unit_vector, unit_y))
        angle = 360 * angle / (2 * math.pi)
        angle = (angle + 360) % 360
        rot_angle = angle - robot_rot
        
        # 각도 정규화
        if rot_angle > 180:
            rot_angle -= 360
        elif rot_angle < -180:
            rot_angle += 360
        
        if abs(rot_angle) > 5:
            if rot_angle > 0:
                self.controller.step(action="RotateRight", degrees=min(abs(rot_angle), 30), agentId=self.agent_id)
            else:
                self.controller.step(action="RotateLeft", degrees=min(abs(rot_angle), 30), agentId=self.agent_id)
            self._capture_frame()  # 회전 후 프레임 캡처
            time.sleep(0.1)
    
    def _world_to_armbase_coords(self, world_pos: Dict[str, float], agent_pos: Dict[str, float], agent_rot: float) -> Dict[str, float]:
        """
        절대 좌표를 armBase 좌표계로 변환
        
        Args:
            world_pos: 절대 좌표 {"x": float, "y": float, "z": float}
            agent_pos: Agent 절대 좌표 {"x": float, "y": float, "z": float}
            agent_rot: Agent 회전 각도 (y축, 도 단위)
            
        Returns:
            armBase 좌표 {"x": float, "y": float, "z": float}
        """
        # Agent 위치 기준 상대 좌표
        dx = world_pos["x"] - agent_pos["x"]
        dy = world_pos["y"] - agent_pos["y"]
        dz = world_pos["z"] - agent_pos["z"]
        
        # Agent 회전 각도를 라디안으로 변환
        rot_rad = math.radians(agent_rot)
        cos_rot = math.cos(rot_rad)
        sin_rot = math.sin(rot_rad)
        
        # armBase 좌표계로 변환 (Agent의 앞쪽이 +z, 오른쪽이 +x)
        # 회전 변환: Agent가 회전한 만큼 반대로 회전
        armbase_x = dx * cos_rot + dz * sin_rot
        armbase_z = -dx * sin_rot + dz * cos_rot
        armbase_y = dy  # y는 회전과 무관
        
        return {"x": armbase_x, "y": armbase_y, "z": armbase_z}
    
    def _calculate_armbase_y_normalized(self, target_y: float, agent_y: float) -> float:
        """
        목표 y 좌표에 맞춰 MoveArmBase의 normalizedY 값 계산
        
        Args:
            target_y: 목표 객체의 y 좌표 (절대)
            agent_y: Agent의 y 좌표 (절대)
            
        Returns:
            normalizedY 값 (0.0~1.0)
        """
        # armBase의 y 범위는 대략 agent_y 기준 -0.5 ~ +1.5 정도
        # normalizedY는 0.0~1.0 범위
        armbase_y_min = agent_y - 0.5
        armbase_y_max = agent_y + 1.5
        armbase_y_range = armbase_y_max - armbase_y_min
        
        if armbase_y_range <= 0:
            return 0.5  # 기본값
        
        # 목표 y 좌표를 normalized 값으로 변환
        normalized_y = (target_y - armbase_y_min) / armbase_y_range
        # 0.0~1.0 범위로 제한
        normalized_y = max(0.0, min(1.0, normalized_y))
        
        return normalized_y
    
    def pickup_object(self, object_name: str) -> bool:
        """
        객체 집기 (ManipulaTHOR API 사용)
        
        use_arm_and_armbase.py를 참고하여:
        1. 목표 객체의 y 좌표에 맞춰 MoveArmBase 조정
        2. 목표 객체의 좌표를 보고 MoveArm으로 팔 이동
        3. handSphereRadius=0.2 사용
        """
        object_id = self._find_object_id(object_name)
        if not object_id:
            print(f"✗ Object '{object_name}' not found")
            return False
        
        # 객체로 이동
        self.goto_object(object_name)
        time.sleep(1)
        self._capture_frame()  # 프레임 캡처
        
        # 객체가 시야에 보이도록 회전
        self._ensure_object_visible(object_id, object_name)
        time.sleep(0.2)
        self._capture_frame()  # 프레임 캡처
        
        # 객체의 중심 위치 가져오기
        obj_center = self._get_object_center(object_id)
        if not obj_center:
            print(f"✗ Object '{object_name}' has invalid position")
            return False
        
        # Agent 위치와 회전 가져오기
        metadata = self.controller.last_event.metadata
        agent_pos = metadata["agent"]["position"]
        agent_rot = metadata["agent"]["rotation"]["y"]
        
        # 1. 목표 객체의 y 좌표에 맞춰 MoveArmBase 조정
        target_y = obj_center["y"]
        normalized_y = self._calculate_armbase_y_normalized(target_y, agent_pos["y"])
        
        print(f"  Adjusting armBase height: target_y={target_y:.3f}, normalized_y={normalized_y:.3f}")
        event = self.controller.step(
            action="MoveArmBase",
            y=normalized_y,
            normalizedY=True,
            agentId=self.agent_id
        )
        self._capture_frame()
        time.sleep(0.2)
        
        if not event.metadata.get('lastActionSuccess', False):
            print(f"  ⚠ MoveArmBase failed: {event.metadata.get('errorMessage', 'Unknown error')}")
        
        # 2. 목표 객체의 좌표를 armBase 좌표계로 변환하여 MoveArm으로 팔 이동
        armbase_coords = self._world_to_armbase_coords(obj_center, agent_pos, agent_rot)
        
        # armBase 좌표계에서 객체 위치로 팔 이동 (약간 앞쪽으로)
        # z는 앞쪽 방향이므로 약간 양수 값으로 조정
        move_pos = {
            "x": armbase_coords["x"],
            "y": armbase_coords["y"],
            "z": max(0.1, armbase_coords["z"])  # 최소 0.1m 앞으로
        }
        
        print(f"  Moving arm to object: armBase coords={move_pos}")
        event = self.controller.step(
            action="MoveArm",
            position=move_pos,
            coordinateSpace="armBase",
            agentId=self.agent_id
        )
        self._capture_frame()
        time.sleep(0.3)
        
        if not event.metadata.get('lastActionSuccess', False):
            print(f"  ⚠ MoveArm failed: {event.metadata.get('errorMessage', 'Unknown error')}")
        
        # 3. 집기 (ManipulaTHOR는 objectIdCandidates 사용, handSphereRadius는 Initialize에서 설정됨)
        event = self.controller.step(
            action="PickupObject",
            objectIdCandidates=[object_id],  # ManipulaTHOR는 리스트로 전달
            agentId=self.agent_id,
            forceAction=True
        )
        self._capture_frame()  # 액션 실행 후 프레임 캡처
        time.sleep(0.1)  # 프레임 안정화
        
        if event.metadata.get('errorMessage'):
            print(f"✗ Pickup failed: {event.metadata['errorMessage']}")
            return False
        
        if not event.metadata.get('lastActionSuccess', False):
            print(f"✗ Pickup failed: action was not successful")
            return False
        
        self.total_exec += 1
        self.success_exec += 1
        print(f"✓ Picked up: {object_name}")
        return True
    
    def put_object(self, object_name: str, receptacle_name: str) -> bool:
        """
        객체를 수용체에 놓기 (ManipulaTHOR API 사용)
        
        use_arm_and_armbase.py를 참고하여:
        1. 수용체의 y 좌표에 맞춰 MoveArmBase 조정
        2. 수용체의 좌표를 보고 MoveArm으로 팔 이동
        3. handSphereRadius=0.2 사용
        """
        receptacle_id = self._find_object_id(receptacle_name)
        if not receptacle_id:
            print(f"✗ Receptacle '{receptacle_name}' not found")
            return False
        
        # 수용체로 이동
        self.goto_object(receptacle_name)
        time.sleep(1)
        self._capture_frame()  # 프레임 캡처
        
        # 수용체가 시야에 정확히 보이도록 회전 (정확히 수용체를 향하도록)
        self._ensure_object_visible(receptacle_id, receptacle_name)
        time.sleep(0.3)  # 회전 후 안정화 시간
        self._capture_frame()  # 프레임 캡처
        
        # 수용체의 중심 위치 가져오기
        recp_center = self._get_object_center(receptacle_id)
        if not recp_center:
            print(f"✗ Receptacle '{receptacle_name}' has invalid position")
            return False
        
        # Agent 위치와 회전 가져오기
        metadata = self.controller.last_event.metadata
        agent_pos = metadata["agent"]["position"]
        agent_rot = metadata["agent"]["rotation"]["y"]
        
        # 1. 수용체의 y 좌표에 맞춰 MoveArmBase 조정
        target_y = recp_center["y"]
        normalized_y = self._calculate_armbase_y_normalized(target_y, agent_pos["y"])
        
        print(f"  Adjusting armBase height: target_y={target_y:.3f}, normalized_y={normalized_y:.3f}")
        event = self.controller.step(
            action="MoveArmBase",
            y=normalized_y,
            normalizedY=True,
            agentId=self.agent_id
        )
        self._capture_frame()
        time.sleep(0.2)
        
        if not event.metadata.get('lastActionSuccess', False):
            print(f"  ⚠ MoveArmBase failed: {event.metadata.get('errorMessage', 'Unknown error')}")
        
        # 2. 수용체의 좌표를 armBase 좌표계로 변환하여 MoveArm으로 팔 이동
        armbase_coords = self._world_to_armbase_coords(recp_center, agent_pos, agent_rot)
        
        # armBase 좌표계에서 수용체 위치로 팔 이동 (약간 앞쪽으로)
        # z는 앞쪽 방향이므로 약간 양수 값으로 조정
        move_pos = {
            "x": armbase_coords["x"],
            "y": armbase_coords["y"],
            "z": max(0.1, armbase_coords["z"])  # 최소 0.1m 앞으로
        }
        
        print(f"  Moving arm to receptacle: armBase coords={move_pos}")
        event = self.controller.step(
            action="MoveArm",
            position=move_pos,
            coordinateSpace="armBase",
            agentId=self.agent_id
        )
        self._capture_frame()
        time.sleep(0.3)
        
        if not event.metadata.get('lastActionSuccess', False):
            print(f"  ⚠ MoveArm failed: {event.metadata.get('errorMessage', 'Unknown error')}")
        
        # 3. 놓기 (ManipulaTHOR는 objectId에 receptacle ID 전달, handSphereRadius는 Initialize에서 설정됨)
        event = self.controller.step(
            action="PutObject",
            objectId=receptacle_id,  # ManipulaTHOR는 receptacle ID를 objectId로 전달
            agentId=self.agent_id,
            forceAction=True
        )
        self._capture_frame()  # 액션 실행 후 프레임 캡처
        time.sleep(0.1)  # 프레임 안정화
        
        if event.metadata.get('errorMessage'):
            print(f"✗ Put failed: {event.metadata['errorMessage']}")
            return False
        
        if not event.metadata.get('lastActionSuccess', False):
            print(f"✗ Put failed: action was not successful")
            return False
        
        self.total_exec += 1
        self.success_exec += 1
        print(f"✓ Put {object_name} in {receptacle_name}")
        return True
    
    def open_object(self, object_name: str) -> bool:
        """객체 열기"""
        object_id = self._find_object_id(object_name)
        if not object_id:
            print(f"✗ Object '{object_name}' not found")
            return False
        
        # 객체로 이동
        self.goto_object(object_name)
        time.sleep(1)
        self._capture_frame()  # 프레임 캡처
        
        # 객체가 시야에 정확히 보이도록 회전 (정확히 객체를 향하도록)
        self._ensure_object_visible(object_id, object_name)
        time.sleep(0.3)  # 회전 후 안정화 시간
        self._capture_frame()  # 프레임 캡처
        
        # 열기
        event = self.controller.step(
            action="OpenObject",
            objectId=object_id,
            agentId=self.agent_id,
            forceAction=True
        )
        self._capture_frame()  # 액션 실행 후 프레임 캡처
        time.sleep(0.1)  # 프레임 안정화
        
        if event.metadata.get('errorMessage'):
            print(f"✗ Open failed: {event.metadata['errorMessage']}")
            return False
        
        if not event.metadata.get('lastActionSuccess', False):
            print(f"✗ Open failed: action was not successful")
            return False
        
        self.total_exec += 1
        self.success_exec += 1
        print(f"✓ Opened: {object_name}")
        return True
    
    def close_object(self, object_name: str) -> bool:
        """객체 닫기"""
        object_id = self._find_object_id(object_name)
        if not object_id:
            print(f"✗ Object '{object_name}' not found")
            return False
        
        # 객체로 이동
        self.goto_object(object_name)
        time.sleep(1)
        self._capture_frame()  # 프레임 캡처
        
        # 객체가 시야에 정확히 보이도록 회전 (정확히 객체를 향하도록)
        self._ensure_object_visible(object_id, object_name)
        time.sleep(0.3)  # 회전 후 안정화 시간
        self._capture_frame()  # 프레임 캡처
        
        # 닫기
        event = self.controller.step(
            action="CloseObject",
            objectId=object_id,
            agentId=self.agent_id,
            forceAction=True
        )
        self._capture_frame()  # 액션 실행 후 프레임 캡처
        time.sleep(0.1)  # 프레임 안정화
        
        if event.metadata.get('errorMessage'):
            print(f"✗ Close failed: {event.metadata['errorMessage']}")
            return False
        
        if not event.metadata.get('lastActionSuccess', False):
            print(f"✗ Close failed: action was not successful")
            return False
        
        self.total_exec += 1
        self.success_exec += 1
        print(f"✓ Closed: {object_name}")
        return True
    
    def parse_action_line(self, line: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        코드 라인에서 액션 파싱
        
        Args:
            line: 코드 라인 (예: "GoToObject('Apple')", "PutObject('Apple', 'Fridge')")
            
        Returns:
            (action, target_object, receptacle) 튜플
        """
        line = line.strip()
        
        # assert 문이나 else: 문은 무시
        if line.startswith("assert") or line.startswith("else:") or not line:
            return None, None, None
        
        # 주석 제거
        if "#" in line:
            line = line.split("#")[0].strip()
        
        # 함수 호출 패턴 매칭
        match = re.match(r'(\w+)\(([^)]*)\)', line)
        if not match:
            return None, None, None
        
        action = match.group(1)
        params = match.group(2)
        
        if not params:
            return action, None, None
        
        # 파라미터 파싱
        params = [p.strip().strip("'\"") for p in params.split(",")]
        
        if len(params) == 1:
            return action, params[0], None
        elif len(params) == 2:
            return action, params[0], params[1]
        else:
            return action, None, None
    
    def execute_action(self, action: str, target_object: Optional[str] = None, receptacle: Optional[str] = None) -> bool:
        """
        액션 실행
        
        Args:
            action: 액션 이름 (GoToObject, PickupObject, PutObject, OpenObject, CloseObject, etc.)
            target_object: 타겟 객체
            receptacle: 수용체 객체 (PutObject 액션의 경우)
            
        Returns:
            성공 여부
        """
        # 새로운 액션 이름 형식으로 매핑 (이전 이름도 호환성을 위해 지원)
        action_map = {
            # 새로운 액션 이름
            "GoToObject": self.goto_object,
            "PickupObject": self.pickup_object,
            "PutObject": self.put_object,
            "OpenObject": self.open_object,
            "CloseObject": self.close_object,
            # 이전 액션 이름 (하위 호환성)
            "GoTo": self.goto_object,
            "Pickup": self.pickup_object,
            "Put": self.put_object,
            "Open": self.open_object,
            "Close": self.close_object,
        }
        
        if action not in action_map:
            print(f"✗ Unknown action: {action}")
            return False
        
        func = action_map[action]
        
        try:
            # PutObject 또는 Put 액션 처리
            if (action == "PutObject" or action == "Put") and receptacle:
                return func(target_object, receptacle)
            elif target_object:
                return func(target_object)
            else:
                print(f"✗ Missing parameters for action: {action}")
                return False
        except Exception as e:
            print(f"✗ Error executing {action}: {str(e)}")
            return False
    
    def execute_program(self, program_code: str) -> Dict[str, Any]:
        """
        ProgPrompt 형식의 프로그램 실행
        
        Args:
            program_code: 프로그램 코드 문자열
            
        Returns:
            실행 결과 딕셔너리
        """
        if not self.controller:
            self.initialize()
        
        lines = program_code.split("\n")
        executed_actions = []
        failed_actions = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # 빈 라인이나 주석은 스킵
            if not line or line.startswith("#"):
                i += 1
                continue
            
            # 일반 액션 실행
            action, target_obj, receptacle = self.parse_action_line(lines[i])
            if action:
                print(f"\n[Executing] {line}")
                success = self.execute_action(action, target_obj, receptacle)
                
                executed_actions.append({
                    "line": line,
                    "action": action,
                    "target_object": target_obj,
                    "receptacle": receptacle,
                    "success": success,
                    "type": "main"
                })
                
                if not success:
                    failed_actions.append(line)
                
                time.sleep(0.3)
            
            i += 1
        
        # 프로그램 종료 시 최종 프레임 캡처
        if self.save_video:
            self._capture_frame()
        
        result = {
            "total_actions": len(executed_actions),
            "successful_actions": sum(1 for a in executed_actions if a["success"]),
            "failed_actions": len(failed_actions),
            "executed_actions": executed_actions,
            "failed_actions": failed_actions,
            "success_rate": (sum(1 for a in executed_actions if a["success"]) / len(executed_actions) * 100) if executed_actions else 0,
        }
        
        # 비디오 경로 추가
        if self.save_video and hasattr(self, 'video_path'):
            result["video_path"] = self.video_path
        
        return result
    
    def close(self):
        """리소스 정리"""
        # 비디오 writer 종료
        if self.save_video:
            self._close_video_writer()
        
        # 실시간 시각화 종료
        if self.fig and MATPLOTLIB_AVAILABLE:
            try:
                plt.close(self.fig)
            except Exception:
                pass
        
        if self.controller:
            self.controller.stop()
            print("✓ AI2-THOR ManipulaTHOR closed")


def find_latest_plan_file(results_dir: str = "results") -> Optional[str]:
    """
    results 디렉토리에서 가장 최근에 생성된 plan JSON 파일 찾기
    
    Args:
        results_dir: results 디렉토리 경로
        
    Returns:
        가장 최근 plan 파일 경로 또는 None
    """
    import json
    from pathlib import Path
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"✗ Results 디렉토리를 찾을 수 없습니다: {results_dir}")
        return None
    
    # ai2thor_progprompt_*.json 파일 찾기
    json_files = list(results_path.glob("ai2thor_progprompt_*.json"))
    
    if not json_files:
        print(f"✗ Results 디렉토리에 plan 파일을 찾을 수 없습니다: {results_dir}")
        return None
    
    # 수정 시간 기준으로 정렬 (가장 최근 파일)
    json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    latest_file = json_files[0]
    
    print(f"✓ 가장 최근 plan 파일 발견: {latest_file}")
    return str(latest_file)


def load_plan_from_file(plan_file: str) -> Dict[str, str]:
    """
    plan JSON 파일에서 plan 딕셔너리 로드
    
    Args:
        plan_file: plan JSON 파일 경로
        
    Returns:
        {task: program_code} 딕셔너리
    """
    import json
    
    try:
        with open(plan_file, "r", encoding="utf-8") as f:
            plan_dict = json.load(f)
        print(f"✓ Plan 파일 로드 완료: {len(plan_dict)}개의 task")
        return plan_dict
    except Exception as e:
        print(f"✗ Plan 파일 로드 실패: {e}")
        return {}


# 사용 예시
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI2-THOR ManipulaTHOR Plan 실행기")
    parser.add_argument("--scene", type=str, default="FloorPlan1_physics", help="Scene 이름")
    parser.add_argument("--headless", action="store_true", help="헤드리스 모드 (GUI 없이 실행)")
    parser.add_argument("--plan-file", type=str, default=None, help="실행할 plan JSON 파일 경로 (지정하지 않으면 가장 최근 파일 사용)")
    parser.add_argument("--results-dir", type=str, default="results", help="Results 디렉토리 경로")
    parser.add_argument("--task", type=str, default=None, help="실행할 특정 task (지정하지 않으면 모든 task 실행)")
    parser.add_argument("--save-video", action="store_true", default=True, help="비디오 저장 (기본값: True)")
    parser.add_argument("--save-images", action="store_true", help="이미지 저장")
    
    args = parser.parse_args()
    
    # Plan 파일 찾기
    if args.plan_file:
        plan_file = args.plan_file
        if not os.path.exists(plan_file):
            print(f"✗ 지정한 plan 파일을 찾을 수 없습니다: {plan_file}")
            sys.exit(1)
    else:
        plan_file = find_latest_plan_file(args.results_dir)
        if not plan_file:
            print("✗ Plan 파일을 찾을 수 없습니다.")
            sys.exit(1)
    
    # Plan 로드
    plan_dict = load_plan_from_file(plan_file)
    if not plan_dict:
        print("✗ Plan을 로드할 수 없습니다.")
        sys.exit(1)
    
    # 실행할 task 선택
    if args.task:
        if args.task in plan_dict:
            plan_dict = {args.task: plan_dict[args.task]}
            print(f"✓ 특정 task 실행: {args.task}")
        else:
            print(f"✗ 지정한 task를 찾을 수 없습니다: {args.task}")
            print(f"  사용 가능한 tasks: {list(plan_dict.keys())}")
            sys.exit(1)
    
    # Executor 초기화
    executor = ManipulaThorExecutor(
        scene=args.scene,
        headless=args.headless,
        save_video=args.save_video,
        save_images=args.save_images
    )
    executor.initialize()
    
    # 각 task 실행
    all_results = {}
    for task_name, program_code in plan_dict.items():
        print(f"\n{'='*80}")
        print(f"Task 실행: {task_name}")
        print(f"{'='*80}")
        
        result = executor.execute_program(program_code)
        all_results[task_name] = result
        
        print(f"\nExecution Summary for '{task_name}':")
        print(f"  Total Actions: {result['total_actions']}")
        print(f"  Successful: {result['successful_actions']}")
        print(f"  Failed: {result['failed_actions']}")
        print(f"  Success Rate: {result['success_rate']:.1f}%")
        
        if result.get('video_path'):
            print(f"  Video: {result['video_path']}")
        
        print(f"\nExecuted Actions:")
        for action in result['executed_actions']:
            status = "✓" if action['success'] else "✗"
            action_type = action.get('type', 'main')
            print(f"  {status} [{action_type}] {action['line']}")
    
    # 전체 요약
    print(f"\n{'='*80}")
    print(f"전체 실행 요약")
    print(f"{'='*80}")
    total_actions = sum(r['total_actions'] for r in all_results.values())
    total_successful = sum(r['successful_actions'] for r in all_results.values())
    total_failed = sum(r['failed_actions'] for r in all_results.values())
    overall_success_rate = (total_successful / total_actions * 100) if total_actions > 0 else 0
    
    print(f"  Total Tasks: {len(all_results)}")
    print(f"  Total Actions: {total_actions}")
    print(f"  Successful: {total_successful}")
    print(f"  Failed: {total_failed}")
    print(f"  Overall Success Rate: {overall_success_rate:.1f}%")
    
    executor.close()

