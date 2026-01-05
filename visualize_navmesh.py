#!/usr/bin/env python3
"""
NavMesh 시각화 스크립트
FloorPlan1의 NavMesh 이동 가능 영역을 격자로 시각화합니다.
"""

import sys
import logging
from pathlib import Path

try:
    from ai2thor.controller import Controller
except ImportError:
    print("ai2thor not installed. Please install: pip install ai2thor")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
except ImportError:
    print("matplotlib or numpy not installed. Please install: pip install matplotlib numpy")
    sys.exit(1)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_navmesh(scene_name: str = "FloorPlan201", grid_size: float = 0.25):
    """
    NavMesh를 시각화
    
    Args:
        scene_name: AI2THOR 씬 이름
        grid_size: 그리드 크기
    """
    # Controller 초기화
    controller = Controller(scene=scene_name, gridSize=grid_size)
    controller.reset(scene_name)
    controller.step(dict(
        action='Initialize',
        agentMode="arm",  # ManipulaTHOR 모드
        snapGrid=False,
        gridSize=grid_size,
        rotateStepDegrees=20,
        visibilityDistance=1.5,
        fieldOfView=120,
        agentCount=1
    ))
    
    logger.info(f"Getting reachable positions from NavMesh for {scene_name}...")
    
    # GetReachablePositions로 이동 가능한 위치들 가져오기
    try:
        event = controller.step(action="GetReachablePositions")
        positions = event.metadata.get("actionReturn", [])
        
        if not positions:
            logger.warning("No reachable positions found")
            controller.stop()
            return
        
        logger.info(f"Found {len(positions)} reachable positions")
        
        # 위치 데이터 추출
        x_coords = [p["x"] for p in positions]
        y_coords = [p["y"] for p in positions]
        z_coords = [p["z"] for p in positions]
        
        # Agent 위치 가져오기
        agent_pos = event.metadata.get("agent", {}).get("position", {})
        agent_x = agent_pos.get("x", 0)
        agent_y = agent_pos.get("y", 0)
        agent_z = agent_pos.get("z", 0)
        
        # 객체 위치들 가져오기 (선택적)
        objects = event.metadata.get("objects", [])
        object_positions = []
        object_names = []
        for obj in objects[:20]:  # 최대 20개만 표시
            obj_pos = obj.get("position", {})
            if obj_pos:
                object_positions.append({
                    "x": obj_pos.get("x", 0),
                    "y": obj_pos.get("y", 0),
                    "z": obj_pos.get("z", 0)
                })
                object_names.append(obj.get("objectType", "Unknown"))
        
        # 시각화
        fig = plt.figure(figsize=(16, 10))
        
        # 3D 플롯
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.scatter(x_coords, y_coords, z_coords, c='blue', alpha=0.3, s=10, label='Reachable Positions')
        ax1.scatter([agent_x], [agent_y], [agent_z], c='red', s=100, marker='^', label='Agent')
        
        # 객체 위치 표시
        if object_positions:
            obj_x = [p["x"] for p in object_positions]
            obj_y = [p["y"] for p in object_positions]
            obj_z = [p["z"] for p in object_positions]
            ax1.scatter(obj_x, obj_y, obj_z, c='green', s=50, marker='o', alpha=0.7, label='Objects')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'NavMesh 3D View - {scene_name}')
        ax1.legend()
        ax1.view_init(elev=20, azim=45)
        
        # 2D Top View (X-Z 평면)
        ax2 = fig.add_subplot(222)
        ax2.scatter(x_coords, z_coords, c='blue', alpha=0.3, s=10, label='Reachable Positions')
        ax2.scatter([agent_x], [agent_z], c='red', s=100, marker='^', label='Agent')
        
        if object_positions:
            obj_x = [p["x"] for p in object_positions]
            obj_z = [p["z"] for p in object_positions]
            ax2.scatter(obj_x, obj_z, c='green', s=50, marker='o', alpha=0.7, label='Objects')
        
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Z (m)')
        ax2.set_title('Top View (X-Z plane)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # 2D Side View (X-Y 평면)
        ax3 = fig.add_subplot(223)
        ax3.scatter(x_coords, y_coords, c='blue', alpha=0.3, s=10, label='Reachable Positions')
        ax3.scatter([agent_x], [agent_y], c='red', s=100, marker='^', label='Agent')
        
        if object_positions:
            obj_x = [p["x"] for p in object_positions]
            obj_y = [p["y"] for p in object_positions]
            ax3.scatter(obj_x, obj_y, c='green', s=50, marker='o', alpha=0.7, label='Objects')
        
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_title('Side View (X-Y plane)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        
        # 통계 정보
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        
        stats_text = f"""
NavMesh Statistics - {scene_name}

Total Reachable Positions: {len(positions)}
Grid Size: {grid_size}m

Position Ranges:
  X: [{min(x_coords):.2f}, {max(x_coords):.2f}] m
  Y: [{min(y_coords):.2f}, {max(y_coords):.2f}] m
  Z: [{min(z_coords):.2f}, {max(z_coords):.2f}] m

Agent Position:
  ({agent_x:.2f}, {agent_y:.2f}, {agent_z:.2f})

Objects in Scene: {len(objects)}
Objects Displayed: {len(object_positions)}
        """
        
        ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', horizontalalignment='left')
        
        plt.tight_layout()
        
        # 저장
        output_path = Path(f"navmesh_visualization_{scene_name}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"NavMesh visualization saved to {output_path}")
        
        # 표시
        plt.show()
        
        controller.stop()
    except Exception as e:
        logger.error(f"Error visualizing NavMesh: {e}")
        import traceback
        traceback.print_exc()
        controller.stop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize AI2THOR NavMesh")
    parser.add_argument("--scene", type=str, default="FloorPlan2", help="Scene name")
    parser.add_argument("--grid-size", type=float, default=0.25, help="Grid size")
    
    args = parser.parse_args()
    
    visualize_navmesh(args.scene, args.grid_size)

