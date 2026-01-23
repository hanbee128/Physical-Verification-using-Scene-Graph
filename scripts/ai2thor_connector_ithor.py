#!/usr/bin/env python3
"""
AI2-THOR Connector - ProgPrompt 형식의 PLAN을 실행할 수 있는 클래스

ProgPrompt 형식의 코드를 파싱하고 AI2-THOR에서 실행합니다.
예: GoTo('Apple'), Pickup('Apple'), Put('Apple', 'Fridge') 등
"""

import cv2
import math
import numpy as np
import os
import random
import re
import shutil
import threading
import time
from datetime import datetime
from glob import glob
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
    """두 점 사이의 거리 계산 (X, Z 좌표만 사용, 2D 수평 거리)"""
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[2] - pt2[2]) ** 2)


def closest_node(target_pos, reachable_positions, closest_node_location=0):
    """타겟 위치에 가장 가까운 도달 가능한 위치 찾기 (단일 로봇용)"""
    if not reachable_positions:
        return None
    
    distances = []
    for pos in reachable_positions:
        dist = distance_pts(target_pos, pos)
        distances.append((dist, pos))
    
    distances.sort(key=lambda x: x[0])
    
    # closest_node_location 인덱스로 선택 (단일 로봇)
    idx = closest_node_location % len(distances)
    return distances[idx][1]


class AI2ThorExecutor:
    """ProgPrompt 형식의 PLAN을 AI2-THOR에서 실행하는 클래스"""
    
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
            video_fps: 영상 FPS (기본값: 30.0, 실제보다 빠르게 재생)
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
        
        # 카메라 업데이트 관련 (오른쪽 사이드 뷰용)
        self.last_agent_position = None
        self.last_agent_rotation = None
        self.camera_update_threshold = 0.1  # 0.1m 이상 이동하거나 5도 이상 회전하면 카메라 업데이트
        
        # 실시간 시각화 관련 (3분할: Top View, Right Side View, Agent View)
        self.show_realtime_view = not headless  # 헤드리스 모드가 아니면 실시간 시각화 활성화
        self.fig = None
        self.ax_camera_top = None
        self.ax_camera_right = None
        self.ax_agent = None
        self.img_display_top = None
        self.img_display_right = None
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
            
            # 하나의 창에 두 개의 subplot 생성 (2분할)
            self.fig = plt.figure(figsize=(20, 8))
            self.fig.canvas.manager.set_window_title("Agent View & Top View")
            
            # Top view 카메라 뷰 (왼쪽)
            self.ax_camera_top = self.fig.add_subplot(121)
            self.ax_camera_top.axis('off')
            self.ax_camera_top.set_title("Top View Camera", fontsize=14, fontweight='bold')
            
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
        """실시간 시각화 업데이트 (3분할: Top View, Right Side View, Agent View)"""
        if not self.show_realtime_view or not MATPLOTLIB_AVAILABLE or not self.controller or not self.fig:
            return
        
        try:
            event = self.controller.last_event
            
            # 서드파티 카메라 이미지 업데이트 (third_party_camera_frames 사용)
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
                        
                        if self.img_display_top is None:
                            self.img_display_top = self.ax_camera_top.imshow(img_top)
                            self.ax_camera_top.set_title("Top View Camera", fontsize=14, fontweight='bold')
                        else:
                            self.img_display_top.set_data(img_top)
                            self.img_display_top.set_clim(vmin=img_top.min(), vmax=img_top.max())
            
            # 에이전트 시야 이미지 업데이트
            agent_img = None
            if hasattr(event, 'frame') and event.frame is not None:
                agent_img = event.frame
            elif hasattr(event, 'events') and len(event.events) > self.agent_id:
                agent_event = event.events[self.agent_id]
                if hasattr(agent_event, 'frame') and agent_event.frame is not None:
                    agent_img = agent_event.frame
            
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
        """현재 프레임을 캡처하여 비디오에 추가 및 실시간 시각화 업데이트 (3분할)"""
        if not self.controller:
            return
        
        # Update Tracking Top Camera
        # self._update_top_tracking_camera() # 삭제됨 (고정 Map View 사용)
        
        # 실시간 시각화 업데이트
        if self.show_realtime_view:
            self._update_realtime_visualization()
        
        # 비디오 저장
        if not self.save_video:
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
        """AI2-THOR 환경 초기화"""
        # Controller 초기화 (scene을 직접 전달하여 불필요한 이중 리셋 방지)
        # headless 모드일 때 렌더링 옵션 최적화
        # Controller 초기화 (scene을 직접 전달하여 불필요한 이중 리셋 방지)
        # headless 모드일 때 렌더링 옵션 최적화
        port = random.randint(9000, 10000)
        print(f"Initializing Controller on port {port}...")
        self.controller = Controller(
            scene=self.scene,
            height=800, 
            width=800, 
            headless=self.headless,
            port=port,
            # 타임아웃 증가 (기본값보다 길게 설정)
            server_timeout=300,
        )
        
        # Agent 초기화
        self.controller.step(dict(
            action='Initialize',
            agentMode="default",
            snapGrid=False,
            gridSize=0.25,
            rotateStepDegrees=20,
            visibilityDistance=1.5,
            fieldOfView=120,
            agentCount=1
        ))
        
        # Top view camera 추가 (고정 Map View)
        event = self.controller.step(action="GetMapViewCameraProperties")
        self.controller.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])
        
        # 실시간 시각화 초기화
        if self.show_realtime_view:
            self._init_realtime_visualization()
        
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
        
        print(f"✓ AI2-THOR initialized: {self.scene}")
    
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
                    
                    # Knife를 찾을 때는 ButterKnife 제외
                    exclude_butter = (obj_type.lower() == "knife")
                    
                    for obj in all_objects:
                        obj_type_actual = obj.get("objectType", "")
                        obj_type_normalized = normalize_object_type(obj_type_actual)
                        
                        # Knife를 찾을 때 ButterKnife 제외
                        if exclude_butter:
                            if "butter" in obj_type_actual.lower():
                                continue
                        
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
        
        # 정확한 매칭 먼저 시도 (objectType 기반)
        object_name_lower = object_name.lower()
        
        # Knife를 찾을 때는 ButterKnife 제외
        exclude_butter = (object_name_lower == "knife")
        
        # 정확한 objectType 매칭 우선
        exact_matches = []
        partial_matches = []
        
        for obj in all_objects:
            obj_id = obj.get("objectId", "")
            obj_type = obj.get("objectType", "")
            obj_type_lower = obj_type.lower()
            obj_id_lower = obj_id.lower()
            
            # Knife를 찾을 때 ButterKnife 제외
            if exclude_butter:
                if "butter" in obj_type_lower or "butter" in obj_id_lower:
                    continue
            
            # 정확한 objectType 매칭
            if obj_type_lower == object_name_lower:
                exact_matches.append(obj_id)
            # objectId 매칭
            elif object_name_lower == obj_id_lower or obj_id_lower.startswith(object_name_lower):
                exact_matches.append(obj_id)
            # 부분 매칭 (objectType)
            elif object_name_lower in obj_type_lower:
                if exclude_butter and "butter" in obj_type_lower:
                    continue
                partial_matches.append(obj_id)
            # 부분 매칭 (objectId)
            elif object_name_lower in obj_id_lower:
                if exclude_butter and "butter" in obj_id_lower:
                    continue
                partial_matches.append(obj_id)
        
        # 정확한 매칭이 있으면 반환
        if exact_matches:
            return exact_matches[0]
        
        # 부분 매칭이 있으면 반환
        if partial_matches:
            return partial_matches[0]
        
        # 정규식 매칭 (기존 로직 유지, 하지만 ButterKnife 제외)
        for obj in all_objects:
            obj_id = obj.get("objectId", "")
            obj_type = obj.get("objectType", "")
            
            if exclude_butter:
                if "butter" in obj_type.lower() or "butter" in obj_id.lower():
                    continue
            
            if re.match(object_name, obj_id, re.IGNORECASE):
                return obj_id
        
        return None
    
    def _get_object_center(self, object_id: str) -> Optional[Dict[str, float]]:
        """객체의 중심 위치 가져오기"""
        for obj in self.controller.last_event.metadata["objects"]:
            if obj["objectId"] == object_id:
                return obj.get("axisAlignedBoundingBox", {}).get("center")
        return None
    
    
    def goto_object(self, object_name: str, target_distance: float = 0.5, success_distance: float = 1.0, max_steps: int = 50) -> bool:
        """
        객체로 이동 (SMART-LLM 방식: ObjectNavExpertAction 사용)
        
        Args:
            object_name: 목표 객체 이름
            target_distance: 목표 거리 (미터, 기본값: 0.5m) - 이동 시도 시 목표로 하는 거리
            success_distance: 성공 판정 거리 (미터, 기본값: None=target_distance와 동일) - 이 거리 내에 들어오면 성공으로 간주
            max_steps: 최대 이동 시도 횟수 (기본값: 50)
        
        Returns:
        성공 여부
        """
        if success_distance is None:
            success_distance = target_distance

        print(f"Going to {object_name} (Target: {target_distance}m, Success Thresh: {success_distance}m)")
        
        object_id = self._find_object_id(object_name)
        if not object_id:
            print(f"✗ Object '{object_name}' not found")
            return False
        
        obj_center = self._get_object_center(object_id)
        if not obj_center or obj_center == {'x': 0.0, 'y': 0.0, 'z': 0.0}:
            print(f"✗ Object '{object_name}' has invalid position")
            return False
        
        dest_obj_pos = [obj_center['x'], obj_center['y'], obj_center['z']]
        
        # SMART-LLM 방식으로 이동
        goal_thresh = target_distance
        closest_node_location = 0
        count_since_update = 0
        prev_dist_goal = 10.0
        
        # 가장 가까운 도달 가능한 위치 찾기
        crp = closest_node(dest_obj_pos, self.reachable_positions, closest_node_location)
        if not crp:
            print(f"✗ No reachable position found")
            return False
        
        # agent에서 객체 중심까지의 거리 계산 (crp까지의 거리가 아님)
        current_agent_pos = [
            self.controller.last_event.events[self.agent_id].metadata["agent"]["position"]["x"],
            self.controller.last_event.events[self.agent_id].metadata["agent"]["position"]["y"],
            self.controller.last_event.events[self.agent_id].metadata["agent"]["position"]["z"]
        ]
        dist_goal = distance_pts(current_agent_pos, dest_obj_pos)
        
        step_count = 0
        while dist_goal > goal_thresh and step_count < max_steps:
            # 현재 agent 위치
            metadata = self.controller.last_event.events[self.agent_id].metadata
            location = {
                "x": metadata["agent"]["position"]["x"],
                "y": metadata["agent"]["position"]["y"],
                "z": metadata["agent"]["position"]["z"],
                "rotation": metadata["agent"]["rotation"]["y"]
            }
            
            prev_dist_goal = dist_goal
            # agent에서 객체 중심까지의 거리 계산 (crp까지의 거리가 아님)
            dist_goal = distance_pts([location['x'], location['y'], location['z']], dest_obj_pos)
            
            # 목표 거리에 도달했는지 즉시 확인
            if dist_goal <= success_distance:
                print(f"✓ Reached: {object_name} (Distance: {dist_goal:.2f}m <= Success: {success_distance}m)")
                return True
            
            dist_del = abs(dist_goal - prev_dist_goal)
            
            if dist_del < 0.2:
                # 로봇이 이동하지 않음
                count_since_update += 1
            else:
                # 로봇이 이동 중
                count_since_update = 0
            
            if count_since_update < 15:
                # ObjectNavExpertAction으로 이동
                event = self.controller.step(dict(
                    action='ObjectNavExpertAction',
                    position=dict(x=crp[0], y=crp[1], z=crp[2]),
                    agentId=self.agent_id
                ))
                self._capture_frame()
                
                # 실패 로그 체크 및 실패 시 다른 위치 시도
                if not event.metadata.get('lastActionSuccess', True):
                    error_msg = event.metadata.get('errorMessage', 'Unknown error')
                    print(f"  ⚠ ObjectNavExpertAction failed: {error_msg}")
                    # 실패 시 다른 가까운 위치 시도
                    closest_node_location += 1
                    count_since_update = 0
                    crp = closest_node(dest_obj_pos, self.reachable_positions, closest_node_location)
                    if not crp:
                        break
                    # agent에서 객체 중심까지의 거리 계산
                    dist_goal = distance_pts([location['x'], location['y'], location['z']], dest_obj_pos)
                    step_count += 1
                    time.sleep(0.5)
                    continue
                
                # ObjectNavExpertAction이 반환한 다음 액션 실행
                next_action = event.metadata.get('actionReturn')
                if next_action:
                    next_event = self.controller.step(
                        action=next_action,
                        agentId=self.agent_id,
                        forceAction=True, # 강제 실행으로 전환 (이동 보장)
                        renderImage=True
                    )
                    # Force updated for ThirdPartyCamera (Top View)
                    self.controller.step(action="Pass")
                    self._capture_frame()
                    
                    # 실패 로그 체크
                    if not next_event.metadata.get('lastActionSuccess', True):
                        error_msg = next_event.metadata.get('errorMessage', 'Unknown error')
                        print(f"  ⚠ Action '{next_action}' failed: {error_msg}")
            else:
                # 목표 위치 업데이트 (다른 가까운 위치 시도)
                closest_node_location += 1
                count_since_update = 0
                crp = closest_node(dest_obj_pos, self.reachable_positions, closest_node_location)
                if not crp:
                    break
                # agent에서 객체 중심까지의 거리 계산
                dist_goal = distance_pts([location['x'], location['y'], location['z']], dest_obj_pos)
            
            step_count += 1
            time.sleep(0.1) # 속도 개선을 위해 0.5 -> 0.1로 줄임
        
        # 최종 거리 확인
        final_metadata = self.controller.last_event.events[self.agent_id].metadata
        final_pos = [
            final_metadata["agent"]["position"]["x"],
            final_metadata["agent"]["position"]["y"],
            final_metadata["agent"]["position"]["z"]
        ]
        final_distance = distance_pts(final_pos, dest_obj_pos)
        
        if final_distance <= success_distance:
            print(f"✓ Reached: {object_name} (Distance: {final_distance:.2f}m <= Success: {success_distance}m)")
            return True
        else:
            print(f"⚠ Could not reach '{object_name}' within {success_distance}m (final distance: {final_distance:.2f}m)")
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
            # 이미 보이면 회전하지 않음 (불필요한 회전 방지)
            print(f"  ✓ '{object_name}' is already visible")
            return True
        
        # 객체가 보이지 않으면 정확한 각도로 한 번에 회전
        metadata = self.controller.last_event.events[self.agent_id].metadata
        robot_pos = metadata["agent"]["position"]
        robot_rot = metadata["agent"]["rotation"]["y"]
        current_horizon = metadata["agent"]["cameraHorizon"]
        
        # 수직 각도 조정: 객체가 위/아래에 있는지 확인
        y_diff = dest_pos[1] - robot_pos['y']  # 객체와 agent의 y 좌표 차이
        robot_object_vec = [dest_pos[0] - robot_pos['x'], dest_pos[2] - robot_pos['z']]
        distance_horizontal = np.linalg.norm(robot_object_vec)
        
        # 수직 각도 계산 (pitch)
        if distance_horizontal > 0.01:  # 수평 거리가 충분히 있을 때만
            pitch_angle_rad = math.atan2(y_diff, distance_horizontal)
            pitch_angle_deg = math.degrees(pitch_angle_rad)
            
            # AI2-THOR의 cameraHorizon 동작:
            # - horizon이 양수면 아래를 보고, 음수면 위를 봄
            # - LookUp은 horizon을 감소시킴 (음수 방향)
            # - LookDown은 horizon을 증가시킴 (양수 방향)
            
            # 목표 horizon 각도 계산 (객체를 정확히 보기 위한 각도)
            # pitch_angle_deg > 0 (위를 향함)이면 target_horizon < 0 (위를 봐야 함)
            # pitch_angle_deg < 0 (아래를 향함)이면 target_horizon > 0 (아래를 봐야 함)
            target_horizon = -pitch_angle_deg  # AI2-THOR의 horizon은 반대 방향
            
            # 현재 horizon과 목표 horizon의 차이
            horizon_diff = target_horizon - current_horizon
            
            # 15도 이상 차이가 있으면 조정
            if abs(horizon_diff) > 15:
                if horizon_diff < 0:
                    # horizon_diff < 0이면 target_horizon < current_horizon
                    # 즉, 현재보다 위를 봐야 함 → LookUp (horizon 감소)
                    look_angle = min(15, abs(horizon_diff))
                    look_event = self.controller.step(action="LookUp", degrees=look_angle, agentId=self.agent_id)
                    self._capture_frame()
                    if look_event.metadata.get('lastActionSuccess', True):
                        print(f"  ↑ Looking up {look_angle:.1f}° (object is above, y_diff={y_diff:.3f})")
                    time.sleep(0.1)
                else:
                    # horizon_diff > 0이면 target_horizon > current_horizon
                    # 즉, 현재보다 아래를 봐야 함 → LookDown (horizon 증가)
                    look_angle = min(15, abs(horizon_diff))
                    look_event = self.controller.step(action="LookDown", degrees=look_angle, agentId=self.agent_id)
                    self._capture_frame()
                    if look_event.metadata.get('lastActionSuccess', True):
                        print(f"  ↓ Looking down {look_angle:.1f}° (object is below, y_diff={y_diff:.3f})")
                    time.sleep(0.1)
                
                # 수직 각도 조정 후 다시 보이는지 확인
                if self._is_object_visible(object_id):
                    print(f"  ✓ '{object_name}' is now visible after vertical adjustment")
                    return True
        
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
            consecutive_failures = 0  # 연속 실패 횟수 추적
            
            while remaining_angle > 5 and max_rotations > 0:
                rotation_amount = min(remaining_angle, 90)  # 최대 90도씩
                rot_event = self.controller.step(action=rotation_direction, degrees=rotation_amount, agentId=self.agent_id)
                self._capture_frame()
                
                # 실패 로그 체크
                if not rot_event.metadata.get('lastActionSuccess', True):
                    error_msg = rot_event.metadata.get('errorMessage', 'Unknown error')
                    print(f"  ⚠ Rotation failed: {error_msg}")
                    consecutive_failures += 1
                    # 연속 실패가 3회 이상이면 조기 종료
                    if consecutive_failures >= 3:
                        print(f"  ⚠ Too many rotation failures, stopping")
                        break
                else:
                    consecutive_failures = 0
                
                time.sleep(0.2)
                
                # 회전 후 수직 각도 재조정 (객체가 위/아래에 있는지 다시 확인)
                metadata_after_rot = self.controller.last_event.events[self.agent_id].metadata
                robot_pos_after = metadata_after_rot["agent"]["position"]
                current_horizon_after = metadata_after_rot["agent"]["cameraHorizon"]
                y_diff_after = dest_pos[1] - robot_pos_after['y']
                robot_object_vec_after = [dest_pos[0] - robot_pos_after['x'], dest_pos[2] - robot_pos_after['z']]
                distance_horizontal_after = np.linalg.norm(robot_object_vec_after)
                
                if distance_horizontal_after > 0.01:
                    pitch_angle_rad_after = math.atan2(y_diff_after, distance_horizontal_after)
                    pitch_angle_deg_after = math.degrees(pitch_angle_rad_after)
                    target_horizon_after = -pitch_angle_deg_after
                    horizon_diff_after = target_horizon_after - current_horizon_after
                    
                    if abs(horizon_diff_after) > 15:
                        if horizon_diff_after < 0:
                            # 위를 봐야 함 → LookUp
                            look_angle_after = min(15, abs(horizon_diff_after))
                            look_event_after = self.controller.step(action="LookUp", degrees=look_angle_after, agentId=self.agent_id)
                            self._capture_frame()
                            time.sleep(0.1)
                        else:
                            # 아래를 봐야 함 → LookDown
                            look_angle_after = min(15, abs(horizon_diff_after))
                            look_event_after = self.controller.step(action="LookDown", degrees=look_angle_after, agentId=self.agent_id)
                            self._capture_frame()
                            time.sleep(0.1)
                
                # 회전 후 객체가 보이는지 확인
                if self._is_object_visible(object_id):
                    print(f"  ✓ '{object_name}' is now visible")
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
    
    def pickup_object(self, object_name: str) -> bool:
        """객체 집기"""
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
        
        # 집기
        event = self.controller.step(
            action="PickupObject",
            objectId=object_id,
            agentId=self.agent_id,
            forceAction=False
        )
        self._capture_frame()  # 액션 실행 후 프레임 캡처
        time.sleep(0.1)  # 프레임 안정화
        
        # 실패 로그 체크
        if not event.metadata.get('lastActionSuccess', True):
            error_msg = event.metadata.get('errorMessage', 'Unknown error')
            print(f"✗ Pickup failed: {error_msg}")
            return False
        
        if event.metadata.get('errorMessage'):
            print(f"✗ Pickup failed: {event.metadata['errorMessage']}")
            return False
        
        self.total_exec += 1
        self.success_exec += 1
        print(f"✓ Picked up: {object_name}")
        return True
    
    def put_object(self, object_name: str, receptacle_name: str) -> bool:
        """객체를 수용체에 놓기"""
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
        
        # 놓기
        event = self.controller.step(
            action="PutObject",
            objectId=receptacle_id,
            agentId=self.agent_id,
            forceAction=False
        )
        self._capture_frame()  # 액션 실행 후 프레임 캡처
        time.sleep(0.1)  # 프레임 안정화
        
        # 실패 로그 체크
        if not event.metadata.get('lastActionSuccess', True):
            error_msg = event.metadata.get('errorMessage', 'Unknown error')
            print(f"✗ Put failed: {error_msg}")
            return False
        
        if event.metadata.get('errorMessage'):
            print(f"✗ Put failed: {event.metadata['errorMessage']}")
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
        
        # 실패 로그 체크
        if not event.metadata.get('lastActionSuccess', True):
            error_msg = event.metadata.get('errorMessage', 'Unknown error')
            print(f"✗ Open failed: {error_msg}")
            return False
        
        if event.metadata.get('errorMessage'):
            print(f"✗ Open failed: {event.metadata['errorMessage']}")
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
        
        # 실패 로그 체크
        if not event.metadata.get('lastActionSuccess', True):
            error_msg = event.metadata.get('errorMessage', 'Unknown error')
            print(f"✗ Close failed: {error_msg}")
            return False
        
        if event.metadata.get('errorMessage'):
            print(f"✗ Close failed: {event.metadata['errorMessage']}")
            return False
        
        self.total_exec += 1
        self.success_exec += 1
        print(f"✓ Closed: {object_name}")
        return True
    
    def toggle_on(self, object_name: str) -> bool:
        """객체 켜기"""
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
        
        # 켜기
        event = self.controller.step(
            action="ToggleObjectOn",
            objectId=object_id,
            agentId=self.agent_id,
            forceAction=True
        )
        self._capture_frame()  # 액션 실행 후 프레임 캡처
        time.sleep(0.1)  # 프레임 안정화
        
        # 실패 로그 체크
        if not event.metadata.get('lastActionSuccess', True):
            error_msg = event.metadata.get('errorMessage', 'Unknown error')
            print(f"✗ ToggleOn failed: {error_msg}")
            return False
        
        if event.metadata.get('errorMessage'):
            print(f"✗ ToggleOn failed: {event.metadata['errorMessage']}")
            return False
        
        self.total_exec += 1
        self.success_exec += 1
        print(f"✓ Turned on: {object_name}")
        return True
    
    def toggle_off(self, object_name: str) -> bool:
        """객체 끄기"""
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
        
        # 끄기
        event = self.controller.step(
            action="ToggleObjectOff",
            objectId=object_id,
            agentId=self.agent_id,
            forceAction=True
        )
        self._capture_frame()  # 액션 실행 후 프레임 캡처
        time.sleep(0.1)  # 프레임 안정화
        
        # 실패 로그 체크
        if not event.metadata.get('lastActionSuccess', True):
            error_msg = event.metadata.get('errorMessage', 'Unknown error')
            print(f"✗ ToggleOff failed: {error_msg}")
            return False
        
        if event.metadata.get('errorMessage'):
            print(f"✗ ToggleOff failed: {event.metadata['errorMessage']}")
            return False
        
        self.total_exec += 1
        self.success_exec += 1
        print(f"✓ Turned off: {object_name}")
        return True
    
    def slice_object(self, object_name: str) -> bool:
        """객체 자르기"""
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
        
        # 자르기
        event = self.controller.step(
            action="SliceObject",
            objectId=object_id,
            agentId=self.agent_id,
            forceAction=True
        )
        self._capture_frame()  # 액션 실행 후 프레임 캡처
        time.sleep(0.1)  # 프레임 안정화
        
        # 실패 로그 체크
        if not event.metadata.get('lastActionSuccess', True):
            error_msg = event.metadata.get('errorMessage', 'Unknown error')
            print(f"✗ Slice failed: {error_msg}")
            return False
        
        if event.metadata.get('errorMessage'):
            print(f"✗ Slice failed: {event.metadata['errorMessage']}")
            return False
        
        self.total_exec += 1
        self.success_exec += 1
        print(f"✓ Sliced: {object_name}")
        return True
    
    def clean_object(self, object_name: str) -> bool:
        """객체 청소하기"""
        object_id = self._find_object_id(object_name)
        if not object_id:
            print(f"✗ Object '{object_name}' not found")
            return False
        
        # 객체로 이동
        self.goto_object(object_name)
        time.sleep(1)
        
        # 객체가 시야에 정확히 보이도록 회전 (정확히 객체를 향하도록)
        self._ensure_object_visible(object_id, object_name)
        time.sleep(0.3)  # 회전 후 안정화 시간
        
        # 청소
        event = self.controller.step(
            action="CleanObject",
            objectId=object_id,
            agentId=self.agent_id,
            forceAction=True
        )
        self._capture_frame()  # 액션 실행 후 프레임 캡처
        time.sleep(0.1)  # 프레임 안정화
        
        # 실패 로그 체크
        if not event.metadata.get('lastActionSuccess', True):
            error_msg = event.metadata.get('errorMessage', 'Unknown error')
            print(f"✗ Clean failed: {error_msg}")
            return False
        
        if event.metadata.get('errorMessage'):
            print(f"✗ Clean failed: {event.metadata['errorMessage']}")
            return False
        
        self.total_exec += 1
        self.success_exec += 1
        print(f"✓ Cleaned: {object_name}")
        return True
    
    def break_object(self, object_name: str) -> bool:
        """객체 깨기"""
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
        
        # 깨기
        event = self.controller.step(
            action="BreakObject",
            objectId=object_id,
            agentId=self.agent_id,
            forceAction=True
        )
        self._capture_frame()  # 액션 실행 후 프레임 캡처
        time.sleep(0.1)  # 프레임 안정화
        
        # 실패 로그 체크
        if not event.metadata.get('lastActionSuccess', True):
            error_msg = event.metadata.get('errorMessage', 'Unknown error')
            print(f"✗ Break failed: {error_msg}")
            return False
        
        if event.metadata.get('errorMessage'):
            print(f"✗ Break failed: {event.metadata['errorMessage']}")
            return False
        
        self.total_exec += 1
        self.success_exec += 1
        print(f"✓ Broke: {object_name}")
        return True
    
    def drop_hand(self) -> bool:
        """손에 든 객체 놓기"""
        event = self.controller.step(
            action="DropHandObject",
            agentId=self.agent_id,
            forceAction=True
        )
        self._capture_frame()  # 액션 실행 후 프레임 캡처
        time.sleep(0.1)  # 프레임 안정화
        
        # 실패 로그 체크
        if not event.metadata.get('lastActionSuccess', True):
            error_msg = event.metadata.get('errorMessage', 'Unknown error')
            print(f"✗ DropHand failed: {error_msg}")
            return False
        
        if event.metadata.get('errorMessage'):
            print(f"✗ DropHand failed: {event.metadata['errorMessage']}")
            return False
        
        print(f"✓ Dropped hand")
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
            "ToggleObjectOn": self.toggle_on,
            "ToggleObjectOff": self.toggle_off,
            "SliceObject": self.slice_object,
            "CleanObject": self.clean_object,
            "BreakObject": self.break_object,
            "DropHandObject": self.drop_hand,
            # 이전 액션 이름 (하위 호환성)
            "GoTo": self.goto_object,
            "Pickup": self.pickup_object,
            "Put": self.put_object,
            "Open": self.open_object,
            "Close": self.close_object,
            "ToggleOn": self.toggle_on,
            "ToggleOff": self.toggle_off,
            "Slice": self.slice_object,
            "Clean": self.clean_object,
            "Break": self.break_object,
            "DropHand": self.drop_hand,
        }
        
        if action not in action_map:
            print(f"✗ Unknown action: {action}")
            return False
        
        func = action_map[action]
        
        try:
            # PutObject 또는 Put 액션 처리
            if (action == "PutObject" or action == "Put") and receptacle:
                return func(target_object, receptacle)
            # DropHandObject 또는 DropHand 액션 처리
            elif action == "DropHandObject" or action == "DropHand":
                return func()
            elif target_object:
                return func(target_object)
            else:
                print(f"✗ Missing parameters for action: {action}")
                return False
        except Exception as e:
            print(f"✗ Error executing {action}: {str(e)}")
            return False
    
    def _check_assert_condition(self, assert_line: str) -> bool:
        """
        assert 조건 확인
        
        Args:
            assert_line: assert 문 (예: "assert('close' to 'Book')", "assert('Book' in 'hands')")
            
        Returns:
            조건 만족 여부
        """
        assert_line = assert_line.strip()
        if not assert_line.startswith("assert"):
            return False
        
        # assert('...') 형식에서 조건 추출
        match = re.search(r"assert\(['\"](.*?)['\"]\)", assert_line)
        if not match:
            return False
        
        condition = match.group(1).strip()
        
        # 'close' to 'Object' 형식
        if "'close' to" in condition or '"close" to' in condition:
            obj_match = re.search(r"to\s+['\"](.*?)['\"]", condition)
            if obj_match:
                obj_name = obj_match.group(1)
                return self._check_close_to_object(obj_name)
        
        # 'Object' in 'hands' 형식
        if " in 'hands'" in condition or ' in "hands"' in condition:
            obj_match = re.search(r"['\"](.*?)['\"]\s+in", condition)
            if obj_match:
                obj_name = obj_match.group(1)
                return self._check_object_in_hands(obj_name)
        
        # 'Object' is 'opened' 형식
        if " is 'opened'" in condition or ' is "opened"' in condition:
            obj_match = re.search(r"['\"](.*?)['\"]\s+is", condition)
            if obj_match:
                obj_name = obj_match.group(1)
                return self._check_object_opened(obj_name)
        
        # 'Object' is 'closed' 형식
        if " is 'closed'" in condition or ' is "closed"' in condition:
            obj_match = re.search(r"['\"](.*?)['\"]\s+is", condition)
            if obj_match:
                obj_name = obj_match.group(1)
                return self._check_object_closed(obj_name)
        
        # 'Object' visible 형식
        if " visible" in condition:
            obj_match = re.search(r"['\"](.*?)['\"]\s+visible", condition)
            if obj_match:
                obj_name = obj_match.group(1)
                return self._check_object_visible(obj_name)
        
        return False
    
    def _check_close_to_object(self, object_name: str, threshold: float = 1.5) -> bool:
        """객체와 가까운지 확인 (threshold 미터 이내)"""
        if not self.controller:
            return False
        
        object_id = self._find_object_id(object_name)
        if not object_id:
            return False
        
        obj_center = self._get_object_center(object_id)
        if not obj_center:
            return False
        
        metadata = self.controller.last_event.events[self.agent_id].metadata
        agent_pos = metadata["agent"]["position"]
        
        distance = distance_pts(
            [agent_pos['x'], agent_pos['y'], agent_pos['z']],
            [obj_center['x'], obj_center['y'], obj_center['z']]
        )
        
        return distance <= threshold
    
    def _check_object_in_hands(self, object_name: str) -> bool:
        """손에 객체를 들고 있는지 확인"""
        if not self.controller:
            return False
        
        metadata = self.controller.last_event.events[self.agent_id].metadata
        inventory_objects = metadata.get("inventoryObjects", [])
        
        if not inventory_objects:
            return False
        
        # inventory에 있는 객체의 objectId 확인
        for inv_obj in inventory_objects:
            if object_name.lower() in inv_obj.lower():
                return True
        
        return False
    
    def _check_object_opened(self, object_name: str) -> bool:
        """객체가 열려있는지 확인"""
        if not self.controller:
            return False
        
        object_id = self._find_object_id(object_name)
        if not object_id:
            return False
        
        for obj in self.controller.last_event.metadata.get("objects", []):
            if obj["objectId"] == object_id:
                return obj.get("isOpen", False)
        
        return False
    
    def _check_object_closed(self, object_name: str) -> bool:
        """객체가 닫혀있는지 확인"""
        if not self.controller:
            return False
        
        object_id = self._find_object_id(object_name)
        if not object_id:
            return False
        
        for obj in self.controller.last_event.metadata.get("objects", []):
            if obj["objectId"] == object_id:
                return not obj.get("isOpen", True)  # isOpen이 False이거나 없으면 닫혀있음
        
        return False
    
    def _check_object_visible(self, object_name: str) -> bool:
        """객체가 보이는지 확인"""
        if not self.controller:
            return False
        
        object_id = self._find_object_id(object_name)
        if not object_id:
            return False
        
        for obj in self.controller.last_event.metadata.get("objects", []):
            if obj["objectId"] == object_id:
                return obj.get("visible", False)
        
        return False
    
    def _parse_else_blocks(self, lines: List[str], start_idx: int) -> Tuple[List[str], int]:
        """
        else: 블록(들)의 액션들 추출 (여러 개의 else: 블록 지원)
        
        Args:
            lines: 전체 코드 라인 리스트
            start_idx: 첫 번째 else: 라인의 인덱스
            
        Returns:
            (else 블록의 액션 라인 리스트, 다음 라인 인덱스) 튜플
        """
        else_actions = []
        i = start_idx
        
        # 첫 번째 else: 라인의 들여쓰기 레벨 확인
        first_else_line = lines[i]
        else_indent = len(first_else_line) - len(first_else_line.lstrip())
        
        while i < len(lines):
            line = lines[i]
            line_stripped = line.strip()
            
            # 빈 라인이거나 주석만 있으면 스킵
            if not line_stripped or line_stripped.startswith("#"):
                i += 1
                continue
            
            # else: 라인인 경우
            if line_stripped.startswith("else:"):
                # else: 다음 라인부터 액션 추출
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]
                    next_line_stripped = next_line.strip()
                    
                    # 빈 라인이거나 주석만 있으면 스킵
                    if not next_line_stripped or next_line_stripped.startswith("#"):
                        j += 1
                        continue
                    
                    # 현재 라인의 들여쓰기
                    next_line_indent = len(next_line) - len(next_line.lstrip())
                    
                    # 들여쓰기가 else 블록보다 작거나 같으면 이 else 블록 종료
                    if next_line_indent <= else_indent:
                        break
                    
                    # 액션 라인인지 확인
                    action, _, _ = self.parse_action_line(next_line)
                    if action:
                        else_actions.append(next_line.strip())
                    
                    j += 1
                
                i = j
            else:
                # else:가 아닌 라인이 나오면 종료
                break
        
        return else_actions, i
    
    def execute_program(self, program_code: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        ProgPrompt 형식의 프로그램 실행 (assert/else 처리 포함)
        
        Args:
            program_code: 프로그램 코드 문자열
            max_retries: assert 실패 시 최대 재시도 횟수
            
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
            
            # assert 문 처리
            if line.startswith("assert"):
                print(f"\n[Checking] {line}")
                condition_met = self._check_assert_condition(line)
                
                if condition_met:
                    print(f"  ✓ Condition met, skipping recovery")
                    # assert 다음 라인으로 이동 (else 블록 스킵)
                    i += 1
                    # else: 블록이 있으면 스킵
                    for j in range(i, len(lines)):
                        if lines[j].strip().startswith("else:"):
                            # else 블록의 끝까지 찾기
                            else_indent = len(lines[j]) - len(lines[j].lstrip())
                            k = j + 1
                            while k < len(lines):
                                next_line = lines[k]
                                next_line_stripped = next_line.strip()
                                if next_line_stripped and not next_line_stripped.startswith("#"):
                                    next_line_indent = len(next_line) - len(next_line.lstrip())
                                    if next_line_indent <= else_indent:
                                        break
                                k += 1
                            i = k
                            break
                    continue
                else:
                    print(f"  ✗ Condition not met, executing recovery actions")
                    # 다음 else: 블록 찾기
                    else_idx = None
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip().startswith("else:"):
                            else_idx = j
                            break
                    
                    if else_idx is not None:
                        # else 블록(들)의 액션들 추출
                        else_actions, next_idx = self._parse_else_blocks(lines, else_idx)
                        
                        # else 블록의 액션들 실행
                        for else_action_line in else_actions:
                            action, target_obj, receptacle = self.parse_action_line(else_action_line)
                            if action:
                                print(f"  [Recovery] {else_action_line}")
                                success = self.execute_action(action, target_obj, receptacle)
                                
                                executed_actions.append({
                                    "line": else_action_line,
                                    "action": action,
                                    "target_object": target_obj,
                                    "receptacle": receptacle,
                                    "success": success,
                                    "type": "recovery"
                                })
                                
                                if not success:
                                    failed_actions.append(else_action_line)
                                
                                time.sleep(0.3)
                        
                        # else 블록 다음으로 이동
                        i = next_idx
                        continue
                    else:
                        print(f"  ⚠ No recovery action found")
                        i += 1
                        continue
            
            # else: 문은 assert 처리 중에 이미 처리되므로 스킵
            if line.startswith("else:"):
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
    
    def get_available_objects(self) -> List[Dict[str, Any]]:
        """
        AI2THOR 환경에서 사용 가능한 모든 객체 목록과 속성을 가져오기
        
        Returns:
            객체 목록 (각 객체는 objectType, properties 등을 포함)
        """
        if not self.controller:
            return []
        
        metadata = self.controller.last_event.metadata
        objects = []
        seen_types = set()
        
        for obj in metadata.get("objects", []):
            obj_type = normalize_object_type(obj.get("objectType", ""))
            
            # 중복 제거 (같은 타입의 객체는 한 번만 추가)
            if obj_type and obj_type not in seen_types:
                seen_types.add(obj_type)
                
                # 객체 속성 추출
                properties = []
                if obj.get("pickupable", False):
                    properties.append("Pickupable")
                if obj.get("receptacle", False):
                    properties.append("Receptacle")
                if obj.get("openable", False):
                    properties.append("Openable")
                if obj.get("toggleable", False):
                    properties.append("Toggleable")
                if obj.get("sliceable", False):
                    properties.append("Sliceable")
                if obj.get("breakable", False):
                    properties.append("Breakable")
                if obj.get("fillable", False):
                    properties.append("Fillable")
                if obj.get("dirtyable", False):
                    properties.append("Dirty")
                if obj.get("canFillWithLiquid", False):
                    properties.append("Fillable")
                
                # On/Off 상태 확인
                if obj.get("isToggled", False) or obj.get("isOn", False):
                    properties.append("On")
                else:
                    if obj.get("toggleable", False) or obj.get("isToggled") is not None:
                        properties.append("Off")
                
                objects.append({
                    "objectType": obj_type,
                    "properties": properties,
                    "objectId": obj.get("objectId", ""),
                })
        
        return objects
    
    def get_available_actions(self) -> List[str]:
        """
        AI2THOR 환경에서 사용 가능한 액션 목록 가져오기
        
        Returns:
            액션 목록 (info.txt 형식)
        """
        actions = [
            "GoToObject <obj>              # Navigate close to an object",
            "PickupObject <obj>            # Pick up a pickupable object (agent can hold only ONE object at a time)",
            "PutObject <obj> <recp>        # Place held object inside/on receptacle",
            "OpenObject <obj>              # Open openable container/appliance",
            "CloseObject <obj>             # Close openable object",
            "SliceObject <obj>             # Slice sliceable food while holding a Knife",
            "ToggleObjectOn <obj>          # Turn on toggleable object (Faucet, Microwave, Toaster, Dishwasher, CoffeeMachine, StoveBurner)",
            "ToggleObjectOff <obj>         # Turn off toggleable object",
            "CleanObject <obj>             # Wash/clean an object near Sink/Faucet",
            "BreakObject <obj>             # Break breakable object",
            "DropHandObject                # Drop whatever the agent is holding (required before picking up another object)",
            "ThrowObject <obj>             # Throw an object",
            "PushObject <obj> <obj>        # Push an object",
            "PullObject <obj> <obj>        # Pull an object",
            "UseUpObject <obj>             # Use up an object",
            "FillObjectWithLiquid <obj> <liquid>     # Fill an object with a liquid",
            "EmptyLiquidFromObject <obj>             # Empty an object",
            "DirtyObject <obj>             # Make an object dirty",

        ]
        return actions
    
    def export_to_info_txt_format(self, output_path: Optional[str] = None) -> str:
        """
        AI2THOR 환경의 객체와 액션을 info.txt 형식으로 내보내기
        
        Args:
            output_path: 저장할 파일 경로 (None이면 문자열만 반환)
        
        Returns:
            info.txt 형식의 문자열
        """
        objects = self.get_available_objects()
        actions = self.get_available_actions()
        
        lines = ["ACTION TEMPLATES:", ""]
        
        # 액션 목록 추가 (info.txt 형식으로 변환)
        action_map = {
            "GoToObject": "goto <obj>",
            "PickupObject": "pickup <obj>",
            "PutObject": "put <obj><recp>",
            "OpenObject": "open <obj>",
            "CloseObject": "close <obj>",
            "SliceObject": "slice <obj>",
            "ToggleObjectOn": "toggleon <obj>",
            "ToggleObjectOff": "toggleoff <obj>",
            "CleanObject": "clean <obj>",
            "BreakObject": "break <obj>",
            "DropHandObject": "drop",
            "ThrowObject": "throw <obj>",
            "PushObject": "push <obj> <obj>",
            "PullObject": "pull <obj> <obj>",
            "UseUpObject": "useup <obj>",
            "FillObjectWithLiquid": "fill <obj> <liquid>",
            "EmptyLiquidFromObject": "empty <obj>",
            "DirtyObject": "dirty <obj>",
        }
        
        for action_line in actions:
            action_name = action_line.split()[0]
            if action_name in action_map:
                lines.append(action_map[action_name])
        
        lines.extend(["done", "", "All OBJECTS and PROPERTIES:", ""])
        
        # 객체 목록 추가
        for obj in sorted(objects, key=lambda x: x["objectType"]):
            obj_type = obj["objectType"]
            props = obj["properties"]
            if props:
                props_str = ", ".join(props)
                lines.append(f"{obj_type} : {props_str}")
            else:
                lines.append(obj_type)
        
        result = "\n".join(lines)
        
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"✓ Exported AI2THOR objects and actions to {output_path}")
        
        return result
    
    def get_scene_state(self) -> Dict[str, Any]:
        """현재 scene 상태 가져오기"""
        if not self.controller:
            return {}
        
        metadata = self.controller.last_event.metadata
        return {
            "agent_position": metadata["agent"]["position"],
            "agent_rotation": metadata["agent"]["rotation"],
            "objects": [
                {
                    "objectId": obj["objectId"],
                    "objectType": normalize_object_type(obj.get("objectType", "")),
                    "position": obj.get("position", {}),
                    "visible": obj.get("visible", False),
                    "isPickedUp": obj.get("isPickedUp", False),
                    "isOpen": obj.get("isOpen", False),
                }
                for obj in metadata.get("objects", [])
            ],
        }
    
    def close(self):
        """리소스 정리"""
        # 비디오 writer 종료
        if self.save_video:
            self._close_video_writer()
        
        if self.controller:
            self.controller.stop()
            print("✓ AI2-THOR closed")


# 사용 예시
if __name__ == "__main__":
    executor = AI2ThorExecutor(scene="FloorPlan1", headless=False)
    executor.initialize()
    
    # ProgPrompt 형식 프로그램 (assert/else 포함)
    program = """
    def break_mug():
        GoToObject('Apple')
        PickupObject('Apple')
        GoToObject('Fridge')
        OpenObject('Fridge')
        PutObject('Apple', 'Fridge')
        CloseObject('Fridge')
        GoToObject('Tomato')
        PickupObject('Tomato')
        GoToObject('Fridge')
        OpenObject('Fridge')
        PutObject('Tomato', 'Fridge')
        CloseObject('Fridge')
        GoToObject('Potato')
        PickupObject('Potato')
        GoToObject('Fridge')
        OpenObject('Fridge')
        PutObject('Potato', 'Fridge')
        CloseObject('Fridge')
    """
    
    result = executor.execute_program(program)
    print(f"\nExecution Summary:")
    print(f"  Total Actions: {result['total_actions']}")
    print(f"  Successful: {result['successful_actions']}")
    print(f"  Failed: {result['failed_actions']}")
    print(f"  Success Rate: {result['success_rate']:.1f}%")
    
    print(f"\nExecuted Actions:")
    for action in result['executed_actions']:
        status = "✓" if action['success'] else "✗"
        action_type = action.get('type', 'main')
        print(f"  {status} [{action_type}] {action['line']}")
    
    executor.close()