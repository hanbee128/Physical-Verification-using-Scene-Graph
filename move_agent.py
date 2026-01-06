#!/usr/bin/env python3
"""
좌표 입력하면 해당 좌표로 iTHOR agent 이동 후 360도 회전하는 스크립트
"""
import sys
import time

try:
    from ai2thor.controller import Controller
except ImportError:
    print("ai2thor not installed. Please install: pip install ai2thor")
    sys.exit(1)


def move_and_rotate_agent(
    scene_name: str = "FloorPlan1",
    grid_size: float = 0.25
):
    """
    좌표를 입력받아 해당 좌표로 agent를 이동하고 360도 회전
    
    Args:
        scene_name: AI2-THOR 씬 이름
        grid_size: 그리드 크기
    """
    # 좌표 입력 받기
    print("좌표를 입력하세요 (x, y, z 형식, 예: -1.0, 0.9, 1.0):")
    position_input = input().strip()
    
    try:
        position_parts = [p.strip() for p in position_input.split(",")]
        if len(position_parts) != 3:
            raise ValueError("좌표는 x, y, z 3개의 값이 필요합니다.")
        
        target_position = {
            "x": float(position_parts[0]),
            "y": float(position_parts[1]),
            "z": float(position_parts[2])
        }
    except ValueError as e:
        print(f"❌ 좌표 입력 오류: {e}")
        print("올바른 형식: x, y, z (예: -1.0, 0.9, 1.0)")
        return
    
    print(f"\n목표 좌표: ({target_position['x']:.3f}, {target_position['y']:.3f}, {target_position['z']:.3f})")
    
    # Controller 초기화 (iTHOR 모드 - agentMode 없음)
    print(f"\nController 초기화 중... (씬: {scene_name})")
    controller = Controller(
        scene=scene_name,
        gridSize=grid_size,
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
    
    print("✓ Controller 초기화 완료")
    
    # 현재 agent 위치 확인
    current_event = controller.last_event
    current_pos = current_event.metadata.get("agent", {}).get("position", {})
    print(f"현재 위치: ({current_pos.get('x', 0):.3f}, {current_pos.get('y', 0):.3f}, {current_pos.get('z', 0):.3f})")
    
    # Teleport로 목표 좌표로 이동
    print(f"\n목표 좌표로 이동 중...")
    teleport_result = controller.step(
        action="Teleport",
        position=target_position,
        agentId=0
    )
    
    if teleport_result.metadata.get("lastActionSuccess", False):
        new_pos = teleport_result.metadata.get("agent", {}).get("position", {})
        print(f"✓ Teleport 성공!")
        print(f"  새 위치: ({new_pos.get('x', 0):.3f}, {new_pos.get('y', 0):.3f}, {new_pos.get('z', 0):.3f})")
    else:
        error_msg = teleport_result.metadata.get("errorMessage", "Unknown error")
        print(f"✗ Teleport 실패: {error_msg}")
        controller.stop()
        return
    
    time.sleep(0.5)  # 이동 후 안정화 시간
    
    # 360도 회전
    print(f"\n360도 회전 시작...")
    
    # 회전 각도 설정 (90도씩 4번 회전)
    rotation_degrees = 36
    num_rotations = 10  # 36도 * 10 = 360도
    
    current_rotation = teleport_result.metadata.get("agent", {}).get("rotation", {}).get("y", 0)
    print(f"현재 회전 각도: {current_rotation:.1f}°")
    
    for i in range(num_rotations):
        print(f"  회전 {i+1}/{num_rotations} ({rotation_degrees}도)...")
        
        # 오른쪽으로 회전
        rotate_result = controller.step(
            action="RotateRight",
            degrees=rotation_degrees,
            agentId=0
        )
        
        if rotate_result.metadata.get("lastActionSuccess", False):
            new_rotation = rotate_result.metadata.get("agent", {}).get("rotation", {}).get("y", 0)
            print(f"    ✓ 회전 완료: {new_rotation:.1f}°")
        else:
            error_msg = rotate_result.metadata.get("errorMessage", "Unknown error")
            print(f"    ✗ 회전 실패: {error_msg}")
            break
        
        time.sleep(0.2)  # 회전 간 대기 시간
    
    # 최종 위치 및 회전 확인
    final_event = controller.last_event
    final_pos = final_event.metadata.get("agent", {}).get("position", {})
    final_rotation = final_event.metadata.get("agent", {}).get("rotation", {}).get("y", 0)
    
    print(f"\n{'='*60}")
    print(f"최종 결과:")
    print(f"  위치: ({final_pos.get('x', 0):.3f}, {final_pos.get('y', 0):.3f}, {final_pos.get('z', 0):.3f})")
    print(f"  회전: {final_rotation:.1f}°")
    print(f"{'='*60}")
    
    # 창을 열어두기 위해 대기
    input("\n창을 닫으려면 Enter를 누르세요...")
    
    controller.stop()
    print("✓ Controller 종료 완료")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="iTHOR agent를 좌표로 이동하고 360도 회전")
    parser.add_argument("--scene", type=str, default="FloorPlan1", help="Scene 이름")
    parser.add_argument("--grid-size", type=float, default=0.25, help="Grid 크기")
    
    args = parser.parse_args()
    
    move_and_rotate_agent(args.scene, args.grid_size)


if __name__ == "__main__":
    main()
