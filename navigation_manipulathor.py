import math
import time
import numpy as np
from ai2thor.controller import Controller

def get_shortest_path_to_object(controller, object_id, initial_position, initial_rotation, target_pos_vector=None):
    """
    Get the shortest path to an object.
    Tries by objectId first, then by target position if provided.
    """
    # 1. Try by ObjectId
    print(f"Attempting pathfinding to Object ID: {object_id}")
    path_event = controller.step(
        action="GetShortestPath",
        objectId=object_id,
        position=initial_position,
        rotation=initial_rotation
    )
    
    if path_event.metadata["lastActionSuccess"]:
        return path_event.metadata["actionReturn"]["corners"]
    
    print(f"Path by ID failed: {path_event.metadata['errorMessage']}")
    
    # 2. Try by Position (if provided)
    if target_pos_vector:
        print(f"Attempting pathfinding to Position: {target_pos_vector}")
        # We use the object's X, Z and current agent's Y (floor) for safer pathfinding?
        # Actually just pass the object pos, AI2-THOR should project it.
        path_event = controller.step(
            action="GetShortestPath",
            position=initial_position,
            rotation=initial_rotation,
            target=target_pos_vector
        )
        if path_event.metadata["lastActionSuccess"]:
             return path_event.metadata["actionReturn"]["corners"]
        print(f"Path by Position failed: {path_event.metadata['errorMessage']}")
    
    return None

def main():
    # 1. Initialize Controller
    print("Initializing Controller...")
    # Note: agentMode="arm" might have specific navmesh constraints.
    # We'll use start with a general setup.
    controller = Controller(
        agentMode="arm", 
        scene="FloorPlan1", 
        gridSize=0.25,
        snapToGrid=False, # Important for smooth movement/rotation
        visibilityDistance=1.5,
        fieldOfView=90
    )
    
    # 1.5 Ensure valid start position
    if not teleport_to_valid_start(controller):
        print("Could not stabilize agent position. Exiting.")
        return

    # 2. Find Target Object
    target_object_type = "Plate" 
    # Or "Apple", "Book", etc. Change this to what you want to find.
    print(f"Searching for object of type: {target_object_type}")
    
    objects = controller.last_event.metadata["objects"]
    target_obj = next((obj for obj in objects if obj["objectType"] == target_object_type), None)
    
    if not target_obj:
        print(f"Object type {target_object_type} not found in scene.")
        return

    target_id = target_obj["objectId"]
    target_pos = target_obj["position"]
    print(f"Found target: {target_id}")
    print(f"Target position: {target_pos}")

    # 3. Get Shortest Path
    agent_meta = controller.last_event.metadata["agent"]
    start_pos = agent_meta["position"]
    start_rot = agent_meta["rotation"]
    
    # Try finding path
    corners = get_shortest_path_to_object(controller, target_id, start_pos, start_rot, target_pos_vector=target_pos)
    
    if not corners or len(corners) == 0:
        print("Path finding returned no corners. Trying manual navigation approach (Direct Move)...")
        # Fallback: Create a direct path to the object (stopping slightly before)
        # We'll just define 'corners' as a single point near the object.
        # Simple heuristic: Move to 0.5m away from target.
        # But for 'move_to_point', we can just pass the target pos, and it stops if close?
        # move_to_point stops if dist < 0.1. We want to stop at 0.5.
        
        # Let's just calculate one waypoint 0.6m away from target in direction of agent.
        curr_x, curr_z = start_pos["x"], start_pos["z"]
        tgt_x, tgt_z = target_pos["x"], target_pos["z"]
        
        full_dist = math.sqrt((tgt_x - curr_x)**2 + (tgt_z - curr_z)**2)
        stop_dist = 0.6
        
        if full_dist > stop_dist:
            ratio = (full_dist - stop_dist) / full_dist
            new_x = curr_x + (tgt_x - curr_x) * ratio
            new_z = curr_z + (tgt_z - curr_z) * ratio
            corners = [{"x": new_x, "z": new_z}]
            print(f"Fallback path set to direct point: {corners[0]}")
        else:
            print("Already close enough.")
            corners = []

    if corners:
        print(f"Following path with {len(corners)} corners.")
        
        # 4. Follow the path
        for i, point in enumerate(corners):
            # Skip if point is too close to current position
            curr = controller.last_event.metadata["agent"]["position"]
            d = math.sqrt((curr["x"] - point["x"])**2 + (curr["z"] - point["z"])**2)
            if d < 0.1:
                continue
                
            print(f"Moving to point {i+1}/{len(corners)}: {point}")
            success = move_to_point(controller, point)
            if not success:
                print("Navigation interrupted (obstacle encountered).")
                break
            
    # Final adjustment: Look at the object
    print("Rotating to face object...")
    current_pos = controller.last_event.metadata["agent"]["position"]
    dx = target_pos["x"] - current_pos["x"]
    dz = target_pos["z"] - current_pos["z"]
    final_angle = math.degrees(math.atan2(dx, dz))
    controller.step(action="RotateAgent", degrees=final_angle)

    print(f"Navigation Complete.")
    print(f"Final Agent Position: {controller.last_event.metadata['agent']['position']}")
    dist_to_obj = math.sqrt((target_pos["x"] - current_pos["x"])**2 + (target_pos["z"] - current_pos["z"])**2)
    print(f"Distance to object: {dist_to_obj:.3f}m")
    
    # Keep window open briefly
    time.sleep(2)

if __name__ == "__main__":
    main()

def move_to_point(controller, target_point):
    """
    Move the agent to a specific point (x, z).
    This handles rotation and movement.
    """
    current_pos = controller.last_event.metadata["agent"]["position"]
    current_rot = controller.last_event.metadata["agent"]["rotation"]["y"]
    
    dx = target_point["x"] - current_pos["x"]
    dz = target_point["z"] - current_pos["z"]
    dist = math.sqrt(dx**2 + dz**2)
    
    if dist < 0.1: # Already there
        return True

    # Calculate target angle
    target_angle = math.degrees(math.atan2(dx, dz))
    
    # Rotate to target angle
    # We can use RotateAgent with absolute degrees
    # But for better animation we might want to calculate delta, 
    # however Teleport/RotateAgent with degrees is simpler for navigation tasks if we don't need realistic turning animation.
    # Let's use simple RotateAgent to absolute degree.
    controller.step(action="RotateAgent", degrees=target_angle)
    
    # Move ahead
    # We can use MoveAhead with 'moveMagnitude' if supported, or loop small steps.
    # In 'arm' mode or standard, 'MoveAhead' typically moves 'gridSize'.
    # With snapToGrid=False, we can try to use a specific magnitude if possible or just small steps.
    # However, MoveAhead usually doesn't take magnitude in all versions.
    # A robust way is to calculating distance and moving.
    # Since we are essentially "navigating" to a point returned by pathfinding (which are corners),
    # we can try to "Teleport" to each corner if we want strictly following path,
    # OR we can just Rotate and MoveAhead until close.
    
    # Let's try MoveAhead loop.
    steps = int(dist / 0.1) # Move in 0.1m chunks
    remainder = dist % 0.1
    
    for _ in range(steps):
        event = controller.step(action="MoveAhead", moveMagnitude=0.1)
        if not event.metadata["lastActionSuccess"]:
            print("MoveAhead failed (obstacle?)")
            return False
            
    if remainder > 0.01:
        controller.step(action="MoveAhead", moveMagnitude=remainder)
        
    return True

def teleport_to_valid_start(controller):
    """
    Teleport the agent to the nearest reachable position on the NavMesh.
    This helps avoid 'Raycast could not find the floor' errors.
    """
    print("Getting reachable positions...")
    event = controller.step(action="GetReachablePositions")
    if not event.metadata["lastActionSuccess"]:
        print("Failed to get reachable positions.")
        return False
        
    reachable_positions = event.metadata["actionReturn"]
    if not reachable_positions:
        print("No reachable positions found.")
        return False
        
    # Find the nearest reachable position to current position
    current_pos = controller.last_event.metadata["agent"]["position"]
    
    nearest_pos = min(reachable_positions, key=lambda p: 
        (p["x"] - current_pos["x"])**2 + (p["z"] - current_pos["z"])**2
    )
    
    print(f"Teleporting to nearest valid position: {nearest_pos}")
    event = controller.step(action="Teleport", position=nearest_pos)
    return event.metadata["lastActionSuccess"]

def main():
    # 1. Initialize Controller
    print("Initializing Controller...")
    # Note: agentMode="arm" might have specific navmesh constraints.
    # We'll use start with a general setup.
    controller = Controller(
        agentMode="arm", 
        scene="FloorPlan1", 
        gridSize=0.25,
        snapToGrid=False, # Important for smooth movement/rotation
        visibilityDistance=1.5,
        fieldOfView=90
    )
    
    # 1.5 Ensure valid start position
    if not teleport_to_valid_start(controller):
        print("Could not stabilize agent position. Exiting.")
        return

    # 2. Find Target Object
    target_object_type = "Plate" 
    # Or "Apple", "Book", etc. Change this to what you want to find.
    print(f"Searching for object of type: {target_object_type}")
    
    objects = controller.last_event.metadata["objects"]
    target_obj = next((obj for obj in objects if obj["objectType"] == target_object_type), None)
    
    if not target_obj:
        print(f"Object type {target_object_type} not found in scene.")
        return

    target_id = target_obj["objectId"]
    print(f"Found target: {target_id}")
    print(f"Target position: {target_obj['position']}")

    # 3. Get Shortest Path
    agent_meta = controller.last_event.metadata["agent"]
    start_pos = agent_meta["position"]
    start_rot = agent_meta["rotation"]
    
    # Note: We pass the *exact* current position (which we know is valid now)
    corners = get_shortest_path_to_object(controller, target_id, start_pos, start_rot)
    
    if not corners or len(corners) == 0:
        print("Path finding returned no corners. Trying manual navigation approach...")
        # Fallback: Just rotate and move towards it?
        # For now, let's see if the Teleport fix solved the 'Raycast' error.
        return

    print(f"Path found with {len(corners)} corners.")
    
    # 4. Follow the path
    # corners[0] is usually the start position, so we skip it if it's very close
    for i, point in enumerate(corners):
        # Skip if point is too close to current position
        curr = controller.last_event.metadata["agent"]["position"]
        d = math.sqrt((curr["x"] - point["x"])**2 + (curr["z"] - point["z"])**2)
        if d < 0.1:
            continue
            
        print(f"Moving to point {i+1}/{len(corners)}: {point}")
        success = move_to_point(controller, point)
        if not success:
            print("Navigation interrupted.")
            break
            
    # Final adjustment: Look at the object
    # After reaching the last point (which is near the object), face the object.
    target_pos = target_obj["position"]
    current_pos = controller.last_event.metadata["agent"]["position"]
    dx = target_pos["x"] - current_pos["x"]
    dz = target_pos["z"] - current_pos["z"]
    final_angle = math.degrees(math.atan2(dx, dz))
    controller.step(action="RotateAgent", degrees=final_angle)

    print(f"Navigation Complete.")
    print(f"Final Agent Position: {controller.last_event.metadata['agent']['position']}")
    
    # Keep window open briefly
    time.sleep(2)

if __name__ == "__main__":
    main()