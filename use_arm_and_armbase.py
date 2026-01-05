#!/usr/bin/env python3
"""
MoveArm과 MoveArmBase 액션을 사용하여 armbase, world, wrist의 움직임을 시각화하는 스크립트
"""
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time


try:
    from ai2thor.controller import Controller
except ImportError:
    print("ai2thor not installed. Please install: pip install ai2thor")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    from PIL import Image
    import io
except ImportError:
    print("matplotlib or numpy or PIL not installed. Please install: pip install matplotlib numpy pillow")
    sys.exit(1)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 한글 폰트 설정 (경고 방지)
try:
    import matplotlib.font_manager as fm
    # 시스템에 설치된 한글 폰트 찾기
    font_list = [f.name for f in fm.fontManager.ttflist]
    korean_fonts = ['NanumGothic', 'Malgun Gothic', 'AppleGothic', 'Noto Sans CJK KR']
    korean_font = None
    for font in korean_fonts:
        if font in font_list:
            korean_font = font
            break
    
    if korean_font:
        plt.rcParams['font.family'] = korean_font
    else:
        # 한글 폰트가 없으면 영어로 변경하거나 경고 무시
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*')
except Exception:
    # 폰트 설정 실패 시 경고 무시
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*')


def get_arm_positions(event) -> Dict[str, Dict[str, float]]:
    """
    이벤트에서 armbase, world, wrist의 위치 정보 추출
    
    Args:
        event: AI2-THOR 이벤트 객체
        
    Returns:
        armbase, world, wrist의 위치 딕셔너리
    """
    arm = event.metadata.get("arm", {})
    
    positions = {}
    
    # ArmBase 위치
    armbase = arm.get("armBase", {})
    if armbase:
        positions["armbase"] = {
            "x": armbase.get("x", 0),
            "y": armbase.get("y", 0),
            "z": armbase.get("z", 0)
        }
    
    return positions


def visualize_arm_movements(
    scene_name: str = "FloorPlan1_physics",
    grid_size: float = 0.25
):
    """
    MoveArm과 MoveArmBase 액션을 실행하고 armbase, world, wrist의 움직임을 시각화
    
    Args:
        scene_name: AI2-THOR 씬 이름
        grid_size: 그리드 크기
    """
    # Controller 초기화 (ManipulaTHOR 모드)
    controller = Controller(
        agentMode="arm",
        scene=scene_name,
        gridSize=grid_size,
        position = {"x": 1.5, "y": 0, "z": 1.5},
        snapToGrid=False,
        rotateStepDegrees=90,
        visibilityDistance=1.5,
        renderInstanceSegmentation=False,
        renderDepthImage=False,
        renderSemanticSegmentation=False,
        width=800,
        height=800,
        fieldOfView=120
    )
    
    logger.info(f"Controller 초기화 완료: {scene_name}")
    
    # 서드파티 카메라 추가 (에이전트를 외부에서 볼 수 있음)
    try:
        # 에이전트 위치 가져오기
        agent_pos = controller.last_event.metadata.get("agent", {}).get("position", {})
        agent_x = agent_pos.get("x", 0)
        agent_y = agent_pos.get("y", 0.9)
        agent_z = agent_pos.get("z", 0)
        
        # 서드파티 카메라 위치 설정 (에이전트 앞쪽에서 보기)
        event = controller.step(
            action="AddThirdPartyCamera",
            position=dict(x=agent_x, y=agent_y + 0.5, z=agent_z - 1.5),  # 에이전트 앞쪽에서 위에서 보기
            rotation=dict(x=15, y=0, z=0),  # 약간 아래를 보도록
            fieldOfView=90
        )
        logger.info("✓ 서드파티 카메라 추가 완료 (에이전트 외부 뷰)")
    except Exception as e:
        logger.warning(f"⚠️ 서드파티 카메라 추가 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 초기 위치 가져오기
    event = controller.last_event
    initial_positions = get_arm_positions(event)
    
    logger.info("초기 위치:")
    for name, pos in initial_positions.items():
        logger.info(f"  {name}: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
    
    dg = 90
    # 로봇 초기 위치에서 왼쪽으로 90도 회전
    logger.info(f"\n로봇을 왼쪽으로 {dg}도 회전 중...")
    try:
        event = controller.step(action="RotateLeft", degrees=dg)
        if event.metadata.get("lastActionSuccess", False):
            logger.info(f"✓ 로봇 회전 완료 (왼쪽 {dg}도)")
        else:
            logger.warning(f"⚠️ 로봇 회전 실패: {event.metadata.get('errorMessage', 'Unknown error')}")
    except Exception as e:
        logger.warning(f"⚠️ 로봇 회전 실패: {e}")
    
    # 위치 추적 리스트
    armbase_trajectory = []
    world_trajectory = []
    wrist_trajectory = []
    
    # 서드파티 카메라 및 에이전트 시야 실시간 표시 설정 (하나의 창에 두 개로 분할)
    plt.ion()  # Interactive mode 활성화
    
    # 하나의 창에 두 개의 subplot 생성
    fig = plt.figure(figsize=(20, 8))
    fig.canvas.manager.set_window_title("Agent View & Third Party Camera View")
    
    # 서드파티 카메라 뷰 (왼쪽)
    ax_camera = fig.add_subplot(121)
    ax_camera.axis('off')
    ax_camera.set_title("Third Party Camera View", fontsize=14, fontweight='bold')
    img_display = None
    
    # 에이전트 시야 뷰 (오른쪽)
    ax_agent = fig.add_subplot(122)
    ax_agent.axis('off')
    ax_agent.set_title("Agent View", fontsize=14, fontweight='bold')
    agent_img_display = None
    
    plt.tight_layout()
    plt.show(block=False)  # 창을 즉시 표시
    
    # 초기 서드파티 카메라 이미지 표시
    try:
        # 서드파티 카메라 이미지를 가져오기 위해 빈 액션 실행
        event = controller.step(action="Pass")
        
        # third_party_camera_frames 사용
        if hasattr(event, 'third_party_camera_frames') and event.third_party_camera_frames:
            img_data = event.third_party_camera_frames[0]
            if img_data is not None:
                if isinstance(img_data, np.ndarray):
                    img = img_data
                else:
                    img = np.array(img_data)
                
                # 이미지 형식 확인 및 변환 (RGB 형식이므로 그대로 사용)
                if len(img.shape) == 3 and img.shape[2] == 4:
                    img = img[:, :, :3]
                
                img_display = ax_camera.imshow(img)
                ax_camera.set_title("Third Party Camera View", fontsize=14, fontweight='bold')
                
                # 에이전트 시야 이미지도 표시
                agent_img = event.frame
                if agent_img is not None:
                    if isinstance(agent_img, np.ndarray):
                        agent_img_array = agent_img
                    else:
                        agent_img_array = np.array(agent_img)
                    
                    if len(agent_img_array.shape) == 3 and agent_img_array.shape[2] == 4:
                        agent_img_array = agent_img_array[:, :, :3]
                    
                    agent_img_display = ax_agent.imshow(agent_img_array)
                    ax_agent.set_title("Agent View", fontsize=14, fontweight='bold')
                
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.1)
                logger.info("✓ 초기 서드파티 카메라 뷰 및 에이전트 시야 뷰 표시")
            else:
                logger.warning("⚠️ 서드파티 카메라 이미지 데이터가 None입니다.")
        else:
            logger.warning("⚠️ 서드파티 카메라 프레임이 없습니다.")
    except Exception as e:
        logger.warning(f"초기 서드파티 카메라 이미지 표시 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 초기 위치 추가
    if "armbase" in initial_positions:
        armbase_trajectory.append(initial_positions["armbase"])
    
    # MoveArmBase 테스트: Y 방향으로만 이동 (MoveArmBase는 y 값만 받음)
    logger.info("\n=== MoveArmBase 테스트 ===")
    # MoveArmBase는 y 값만 받음 (normalizedY=True일 경우 0.0~1.0 범위)
    armbase_movements = [
        0.0,   # 최하단
        0.05,  # 5%
        0.1,   # 10%
        0.15,  # 15%
        0.2,   # 20%
        0.25,  # 25%
        0.3,   # 30%
        0.35,  # 35%
        0.4,   # 40%
        0.45,  # 45%
        0.5,   # 50%
        0.55,  # 55%
        0.6,   # 60%
        0.65,  # 65%
        0.7,   # 70%
        0.75,  # 75%
        0.8,   # 80%
        0.85,  # 85%
        0.9,   # 90%
        0.95,  # 95%
        1.0,   # 최상단
        0.95,  # 95%
        0.9,   # 90%
        0.85,  # 85%
        0.8,   # 80%
        0.75,  # 75%
        0.7,   # 70%
        0.65,  # 65%
        0.6,   # 60%
        0.55,  # 55%
        0.5
    ]
    
    for i, target_y in enumerate(armbase_movements):
        logger.info(f"\nMoveArmBase {i+1}/{len(armbase_movements)}: y={target_y} (normalized)")
        event = controller.step(action="MoveArmBase", y=target_y, normalizedY=True, speed=0.01)  # 매우 느린 속도
        
        if event.metadata.get("lastActionSuccess", False):
            positions = get_arm_positions(event)
            logger.info(f"  성공!")
            for name, pos in positions.items():
                logger.info(f"    {name}: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
                if name == "armbase":
                    armbase_trajectory.append(pos)
            
            # 서드파티 카메라 및 에이전트 시야 이미지 실시간 표시
            try:
                # 서드파티 카메라 이미지 업데이트 (third_party_camera_frames 사용)
                if hasattr(event, 'third_party_camera_frames') and event.third_party_camera_frames:
                    img_data = event.third_party_camera_frames[0]
                    if img_data is not None:
                        if isinstance(img_data, np.ndarray):
                            img = img_data
                        else:
                            img = np.array(img_data)
                        
                        # 이미지 형식 확인 및 변환
                        if len(img.shape) == 3 and img.shape[2] == 4:
                            img = img[:, :, :3]
                        
                        # 서드파티 카메라 이미지 업데이트
                        if img_display is None:
                            img_display = ax_camera.imshow(img)
                            ax_camera.set_title("Third Party Camera View", fontsize=14, fontweight='bold')
                        else:
                            img_display.set_data(img)
                            img_display.set_clim(vmin=img.min(), vmax=img.max())
                
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
                
                # 하나의 figure에 두 개의 subplot이므로 한 번만 업데이트
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.01)  # 화면 업데이트
            except Exception as e:
                logger.warning(f"이미지 표시 실패: {e}")
        else:
            logger.warning(f"  실패: {event.metadata.get('errorMessage', 'Unknown error')}")
    
    # MoveArm 테스트: armBase 좌표계에서 x, y, z 각각 최소~최대로 움직이기
    logger.info("\n=== MoveArm 테스트 (armBase 좌표계) ===")
    
    # armBase 좌표계 범위: x: -0.5 ~ 0.5, y: -0.5 ~ 0.5, z: 0 ~ 0.75
    coord_space = "armBase"
    
    # 초기 위치로 복귀
    logger.info("\n초기 위치로 복귀 중...")
    event = controller.step(
        action="MoveArm",
        position={"x": 0, "y": 0, "z": 0.5},
        speed=0.01,
        coordinateSpace=coord_space
    )
    
    # X 방향 테스트: 최소(-0.5) ~ 최대(0.5)
    logger.info(f"\n{'='*80}")
    logger.info(f"X 방향 테스트: -1 ~ 1")
    logger.info(f"{'='*80}")
    x_movements = []
    for x in np.linspace(-1, 1, 20):
        x_movements.append({"x": x, "y": 0, "z": 0.5})
    
    for i, target_pos in enumerate(x_movements):
        logger.info(f"\nMoveArm (X 방향) {i+1}/{len(x_movements)}: {target_pos}")
        event = controller.step(
            action="MoveArm",
            position=target_pos,
            speed=0.01,  # 매우 느린 속도
            coordinateSpace=coord_space
        )
        
        if event.metadata.get("lastActionSuccess", False):
            positions = get_arm_positions(event)
            logger.info(f"  성공!")
            for name, pos in positions.items():
                logger.info(f"    {name}: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
                if name == "armbase":
                    armbase_trajectory.append(pos)
            
            # 서드파티 카메라 및 에이전트 시야 이미지 실시간 표시
            try:
                # 서드파티 카메라 이미지 업데이트 (third_party_camera_frames 사용)
                if hasattr(event, 'third_party_camera_frames') and event.third_party_camera_frames:
                    img_data = event.third_party_camera_frames[0]
                    if img_data is not None:
                        if isinstance(img_data, np.ndarray):
                            img = img_data
                        else:
                            img = np.array(img_data)
                        
                        # 이미지 형식 확인 및 변환
                        if len(img.shape) == 3 and img.shape[2] == 4:
                            img = img[:, :, :3]
                        
                        # 서드파티 카메라 이미지 업데이트
                        if img_display is None:
                            img_display = ax_camera.imshow(img)
                            ax_camera.set_title("Third Party Camera View", fontsize=14, fontweight='bold')
                        else:
                            img_display.set_data(img)
                            img_display.set_clim(vmin=img.min(), vmax=img.max())
                
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
                
                # 하나의 figure에 두 개의 subplot이므로 한 번만 업데이트
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.01)  # 화면 업데이트
            except Exception as e:
                logger.warning(f"이미지 표시 실패: {e}")
        else:
            logger.warning(f"  실패: {event.metadata.get('errorMessage', 'Unknown error')}")
    
    logger.info("\nX 방향 테스트 완료. 2초 대기 중...")
    time.sleep(2)
    
    # Y 방향 테스트: 최소(-0.5) ~ 최대(0.5)
    logger.info(f"\n{'='*80}")
    logger.info(f"Y 방향 테스트: -0.5 ~ 0.5")
    logger.info(f"{'='*80}")
    y_movements = []
    for y in np.linspace(-0.5, 1, 20):
        y_movements.append({"x": 0, "y": y, "z": 0.5})
    
    for i, target_pos in enumerate(y_movements):
        logger.info(f"\nMoveArm (Y 방향) {i+1}/{len(y_movements)}: {target_pos}")
        event = controller.step(
            action="MoveArm",
            position=target_pos,
            speed=0.01,  # 매우 느린 속도
            coordinateSpace=coord_space
        )
        
        if event.metadata.get("lastActionSuccess", False):
            positions = get_arm_positions(event)
            logger.info(f"  성공!")
            for name, pos in positions.items():
                logger.info(f"    {name}: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
                if name == "armbase":
                    armbase_trajectory.append(pos)
            
            # 서드파티 카메라 및 에이전트 시야 이미지 실시간 표시
            try:
                # 서드파티 카메라 이미지 업데이트 (third_party_camera_frames 사용)
                if hasattr(event, 'third_party_camera_frames') and event.third_party_camera_frames:
                    img_data = event.third_party_camera_frames[0]
                    if img_data is not None:
                        if isinstance(img_data, np.ndarray):
                            img = img_data
                        else:
                            img = np.array(img_data)
                        
                        # 이미지 형식 확인 및 변환
                        if len(img.shape) == 3 and img.shape[2] == 4:
                            img = img[:, :, :3]
                        
                        # 서드파티 카메라 이미지 업데이트
                        if img_display is None:
                            img_display = ax_camera.imshow(img)
                            ax_camera.set_title("Third Party Camera View", fontsize=14, fontweight='bold')
                        else:
                            img_display.set_data(img)
                            img_display.set_clim(vmin=img.min(), vmax=img.max())
                
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
                
                # 하나의 figure에 두 개의 subplot이므로 한 번만 업데이트
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.01)  # 화면 업데이트
            except Exception as e:
                logger.warning(f"이미지 표시 실패: {e}")
        else:
            logger.warning(f"  실패: {event.metadata.get('errorMessage', 'Unknown error')}")
    
    logger.info("\nY 방향 테스트 완료. 2초 대기 중...")
    time.sleep(2)
    
    # Z 방향 테스트: 최소(0) ~ 최대(0.75)
    logger.info(f"\n{'='*80}")
    logger.info(f"Z 방향 테스트: 0 ~ 0.75")
    logger.info(f"{'='*80}")
    z_movements = []
    for z in np.linspace(0, 1, 20):
        z_movements.append({"x": 0, "y": 0, "z": z})
    
    for i, target_pos in enumerate(z_movements):
        logger.info(f"\nMoveArm (Z 방향) {i+1}/{len(z_movements)}: {target_pos}")
        event = controller.step(
            action="MoveArm",
            position=target_pos,
            speed=0.01,  # 매우 느린 속도
            coordinateSpace=coord_space
        )
        
        if event.metadata.get("lastActionSuccess", False):
            positions = get_arm_positions(event)
            logger.info(f"  성공!")
            for name, pos in positions.items():
                logger.info(f"    {name}: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
                if name == "armbase":
                    armbase_trajectory.append(pos)
            
            # 서드파티 카메라 및 에이전트 시야 이미지 실시간 표시
            try:
                # 서드파티 카메라 이미지 업데이트 (third_party_camera_frames 사용)
                if hasattr(event, 'third_party_camera_frames') and event.third_party_camera_frames:
                    img_data = event.third_party_camera_frames[0]
                    if img_data is not None:
                        if isinstance(img_data, np.ndarray):
                            img = img_data
                        else:
                            img = np.array(img_data)
                        
                        # 이미지 형식 확인 및 변환
                        if len(img.shape) == 3 and img.shape[2] == 4:
                            img = img[:, :, :3]
                        
                        # 서드파티 카메라 이미지 업데이트
                        if img_display is None:
                            img_display = ax_camera.imshow(img)
                            ax_camera.set_title("Third Party Camera View", fontsize=14, fontweight='bold')
                        else:
                            img_display.set_data(img)
                            img_display.set_clim(vmin=img.min(), vmax=img.max())
                
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
                
                # 하나의 figure에 두 개의 subplot이므로 한 번만 업데이트
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.01)  # 화면 업데이트
            except Exception as e:
                logger.warning(f"이미지 표시 실패: {e}")
        else:
            logger.warning(f"  실패: {event.metadata.get('errorMessage', 'Unknown error')}")
    
    logger.info("\nZ 방향 테스트 완료. 2초 대기 중...")
    time.sleep(2)
    
    logger.info("\n=== 모든 테스트 완료 ===")
    logger.info("서드파티 카메라 뷰와 에이전트 시야 뷰 창이 열려있습니다.")
    

    repeat_test = input("x, y, z 테스트 반복 : 입력하세요")
    if repeat_test == "x":
        # X 방향 테스트: 최소(-0.5) ~ 최대(0.5)
        logger.info(f"\n{'='*80}")
        logger.info(f"X 방향 테스트: -0.5 ~ 0.5")
        logger.info(f"{'='*80}")
        x_movements = []
        for x in np.linspace(-1, 1, 20):
            x_movements.append({"x": x, "y": 0, "z": 0.5})
        
        for i, target_pos in enumerate(x_movements):
            logger.info(f"\nMoveArm (X 방향) {i+1}/{len(x_movements)}: {target_pos}")
            event = controller.step(
                action="MoveArm",
                position=target_pos,
                speed=0.01,  # 매우 느린 속도
                coordinateSpace=coord_space
            )
            
            if event.metadata.get("lastActionSuccess", False):
                positions = get_arm_positions(event)
                logger.info(f"  성공!")
                for name, pos in positions.items():
                    logger.info(f"    {name}: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
                    if name == "armbase":
                        armbase_trajectory.append(pos)
                
                # 서드파티 카메라 및 에이전트 시야 이미지 실시간 표시
                try:
                    # 서드파티 카메라 이미지 업데이트 (third_party_camera_frames 사용)
                    if hasattr(event, 'third_party_camera_frames') and event.third_party_camera_frames:
                        img_data = event.third_party_camera_frames[0]
                        if img_data is not None:
                            if isinstance(img_data, np.ndarray):
                                img = img_data
                            else:
                                img = np.array(img_data)
                            
                            # 이미지 형식 확인 및 변환
                            if len(img.shape) == 3 and img.shape[2] == 4:
                                img = img[:, :, :3]
                            
                            # 서드파티 카메라 이미지 업데이트
                            if img_display is None:
                                img_display = ax_camera.imshow(img)
                                ax_camera.set_title("Third Party Camera View", fontsize=14, fontweight='bold')
                            else:
                                img_display.set_data(img)
                                img_display.set_clim(vmin=img.min(), vmax=img.max())
                    
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
                    
                    # 하나의 figure에 두 개의 subplot이므로 한 번만 업데이트
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.01)  # 화면 업데이트
                except Exception as e:
                    logger.warning(f"이미지 표시 실패: {e}")
            else:
                logger.warning(f"  실패: {event.metadata.get('errorMessage', 'Unknown error')}")
        
        logger.info("\nX 방향 테스트 완료. 2초 대기 중...")
        time.sleep(2)
    
    if repeat_test == "y":
        # Y 방향 테스트: 최소(-0.5) ~ 최대(0.5)
        logger.info(f"\n{'='*80}")
        logger.info(f"Y 방향 테스트: -0.5 ~ 0.5")
        logger.info(f"{'='*80}")
        y_movements = []
        for y in np.linspace(-0.5, 1, 20):
            y_movements.append({"x": 0, "y": y, "z": 0.5})
        
        for i, target_pos in enumerate(y_movements):
            logger.info(f"\nMoveArm (Y 방향) {i+1}/{len(y_movements)}: {target_pos}")
            event = controller.step(
                action="MoveArm",
                position=target_pos,
                speed=0.01,  # 매우 느린 속도
                coordinateSpace=coord_space
            )
            
            if event.metadata.get("lastActionSuccess", False):
                positions = get_arm_positions(event)
                logger.info(f"  성공!")
                for name, pos in positions.items():
                    logger.info(f"    {name}: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
                    if name == "armbase":
                        armbase_trajectory.append(pos)
                
                # 서드파티 카메라 및 에이전트 시야 이미지 실시간 표시
                try:
                    # 서드파티 카메라 이미지 업데이트 (third_party_camera_frames 사용)
                    if hasattr(event, 'third_party_camera_frames') and event.third_party_camera_frames:
                        img_data = event.third_party_camera_frames[0]
                        if img_data is not None:
                            if isinstance(img_data, np.ndarray):
                                img = img_data
                            else:
                                img = np.array(img_data)
                            
                            # 이미지 형식 확인 및 변환
                            if len(img.shape) == 3 and img.shape[2] == 4:
                                img = img[:, :, :3]
                            
                            # 서드파티 카메라 이미지 업데이트
                            if img_display is None:
                                img_display = ax_camera.imshow(img)
                                ax_camera.set_title("Third Party Camera View", fontsize=14, fontweight='bold')
                            else:
                                img_display.set_data(img)
                                img_display.set_clim(vmin=img.min(), vmax=img.max())
                    
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
                    
                    # 하나의 figure에 두 개의 subplot이므로 한 번만 업데이트
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.01)  # 화면 업데이트
                except Exception as e:
                    logger.warning(f"이미지 표시 실패: {e}")
            else:
                logger.warning(f"  실패: {event.metadata.get('errorMessage', 'Unknown error')}")
        
        logger.info("\nY 방향 테스트 완료. 2초 대기 중...")
        time.sleep(2)
    
    if repeat_test == "z":
        # Z 방향 테스트: 최소(0) ~ 최대(0.75)
        logger.info(f"\n{'='*80}")
        logger.info(f"Z 방향 테스트: 0 ~ 0.75")
        logger.info(f"{'='*80}")
        z_movements = []
        for z in np.linspace(0, 1, 20):
            z_movements.append({"x": 0, "y": 0, "z": z})
        
        for i, target_pos in enumerate(z_movements):
            logger.info(f"\nMoveArm (Z 방향) {i+1}/{len(z_movements)}: {target_pos}")
            event = controller.step(
                action="MoveArm",
                position=target_pos,
                speed=0.01,  # 매우 느린 속도
                coordinateSpace=coord_space
            )
            
            if event.metadata.get("lastActionSuccess", False):
                positions = get_arm_positions(event)
                logger.info(f"  성공!")
                for name, pos in positions.items():
                    logger.info(f"    {name}: ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f})")
                    if name == "armbase":
                        armbase_trajectory.append(pos)
                
                # 서드파티 카메라 및 에이전트 시야 이미지 실시간 표시
                try:
                    # 서드파티 카메라 이미지 업데이트 (third_party_camera_frames 사용)
                    if hasattr(event, 'third_party_camera_frames') and event.third_party_camera_frames:
                        img_data = event.third_party_camera_frames[0]
                        if img_data is not None:
                            if isinstance(img_data, np.ndarray):
                                img = img_data
                            else:
                                img = np.array(img_data)
                            
                            # 이미지 형식 확인 및 변환
                            if len(img.shape) == 3 and img.shape[2] == 4:
                                img = img[:, :, :3]
                            
                            # 서드파티 카메라 이미지 업데이트
                            if img_display is None:
                                img_display = ax_camera.imshow(img)
                                ax_camera.set_title("Third Party Camera View", fontsize=14, fontweight='bold')
                            else:
                                img_display.set_data(img)
                                img_display.set_clim(vmin=img.min(), vmax=img.max())
                    
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
                    
                    # 하나의 figure에 두 개의 subplot이므로 한 번만 업데이트
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.01)  # 화면 업데이트
                except Exception as e:
                    logger.warning(f"이미지 표시 실패: {e}")
            else:
                logger.warning(f"  실패: {event.metadata.get('errorMessage', 'Unknown error')}")
        
        logger.info("\nZ 방향 테스트 완료. 2초 대기 중...")
        time.sleep(2)

    # 창을 열어두기 위해 대기
    input("창을 닫으려면 Enter를 누르세요...")

    controller.stop()
    logger.info("Controller 종료 완료")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MoveArm과 MoveArmBase 액션 시각화")
    parser.add_argument("--scene", type=str, default="FloorPlan1_physics", help="Scene 이름")
    parser.add_argument("--grid-size", type=float, default=0.25, help="Grid 크기")
    
    args = parser.parse_args()
    
    visualize_arm_movements(args.scene, args.grid_size)


if __name__ == "__main__":
    main()
