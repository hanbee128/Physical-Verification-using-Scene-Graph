#!/usr/bin/env python3
"""
Evaluation script for Physical Guard results.
This script reads the generated plans from `physical_guard_set3_result_*.txt`,
executes them using AI2-THOR, and compares the final state with the expected results
defined in `data/final_test/FloorPlan1.json`.
"""

import os
import sys
import json
import glob
import re
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add the scripts directory to sys.path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Return to AI2ThorExecutor
from ai2thor_connector_ithor import AI2ThorExecutor

def parse_plan_file(file_path: str) -> Dict[str, str]:
    """
    Parses the plan file to extract tasks and their corresponding programs.
    Returns a dictionary mapping task names to program code.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    plans = {}
    # Split by "Task: "
    parts = content.split("Task: ")
    for part in parts:
        if not part.strip():
            continue
        
        # Extract task name (first line)
        lines = part.strip().split('\n')
        task_name = lines[0].strip()
        
        # Extract code (skip separator line if present)
        code_lines = []
        start_collecting = False
        for line in lines[1:]:
            if line.startswith("==="):
                start_collecting = True
                continue
            if line.startswith("Physical Verification Summary:"): # Stop parsing at summary
                break
            if start_collecting or not line.startswith("==="): # If no separator, start collecting immediately or after separator
                 code_lines.append(line)
        
        # Clean up code
        code = '\n'.join(code_lines).strip()
        
        # If code is empty, try to extract from function definition
        if not code:
            # Look for function definition: def function_name():
            for line in lines:
                if line.strip().startswith("def ") and "():" in line:
                    # Extract task name from function name or comment
                    # Try to find "# Task: " comment
                    for comment_line in lines:
                        if "# Task:" in comment_line:
                            task_name_from_comment = comment_line.split("# Task:")[-1].strip()
                            if task_name_from_comment:
                                task_name = task_name_from_comment
                                break
                    break
        
        if code and task_name:
             # Normalize task name to lowercase for robust matching
             normalized_task = task_name.strip().lower()
             plans[normalized_task] = code
             
    return plans

def load_expected_results(json_path: str) -> List[Dict[str, Any]]:
    """Loads expected object states from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def verify_object_state(executor: AI2ThorExecutor, expected_state: Dict[str, Any]) -> Tuple[int, int]:
    """
    Verifies object state and returns (passed_count, total_count).
    """
    object_name = expected_state['name']
    expected_contains = expected_state.get('contains', [])
    expected_status = expected_state.get('state', 'None')
    
    passed_count = 0
    total_count = 0
    
    obj_id = executor._find_object_id(object_name)
    if not obj_id:
        print(f"  ❌ Object '{object_name}' not found in simulator.")
        # If object not found, we count the conditions but they result in 0 passed.
        # Count 'contains' items
        total_count += len(expected_contains)
        # Count 'state' if not None
        if expected_status != 'None':
             total_count += 1
        return 0, total_count

    # Get object metadata from last event
    all_objects = executor.controller.last_event.metadata['objects']
    obj_meta = next((o for o in all_objects if o['objectId'] == obj_id), None)
    
    if not obj_meta:
        print(f"  ❌ Metadata for '{object_name}' ({obj_id}) not found.")
        total_count += len(expected_contains)
        if expected_status != 'None':
             total_count += 1
        return 0, total_count

    # Check containment using receptacleObjectIds
    if expected_contains:
        # Get object IDs of things currently inside the target object using receptacleObjectIds
        receptacle_object_ids = obj_meta.get('receptacleObjectIds', [])
        contained_objects = []
        
        # receptacleObjectIds에 있는 objectId로 실제 객체 정보 찾기
        for contained_obj_id in receptacle_object_ids:
            contained_obj = next((o for o in all_objects if o.get('objectId') == contained_obj_id), None)
            if contained_obj:
                contained_objects.append({
                    'objectId': contained_obj_id,
                    'objectType': contained_obj.get('objectType', '')
                })

        # Check if each expected item is present
        for item in expected_contains:
            total_count += 1
            found = False
            found_object_id = None
            
            for contained in contained_objects:
                contained_type = contained.get('objectType', '')
                # objectType으로 매칭 (예: "Egg", "Apple" 등)
                if item.lower() in contained_type.lower() or contained_type.lower() in item.lower():
                    found = True
                    found_object_id = contained.get('objectId')
                    break
            
            if not found:
                # Debug: find where it actually is
                actual_parent = "None"
                for o in all_objects:
                    obj_type = o.get('objectType', '')
                    if item.lower() in obj_type.lower() or obj_type.lower() in item.lower():
                        # parentReceptacle 또는 parentReceptacles 확인
                        pid = o.get('parentReceptacle')
                        if not pid:
                            parent_receptacles = o.get('parentReceptacles', [])
                            if parent_receptacles:
                                pid = parent_receptacles[0]
                        if pid:
                            # parent receptacle의 objectType 찾기
                            parent_obj = next((p for p in all_objects if p.get('objectId') == pid), None)
                            if parent_obj:
                                actual_parent = parent_obj.get('objectType', pid)
                            else:
                                actual_parent = pid
                            break
                             
                print(f"  ❌ Expected '{item}' in '{object_name}', but not found.")
                print(f"     Found in receptacle: {[c['objectType'] for c in contained_objects]}")
                print(f"     (Debug: '{item}' seems to be in '{actual_parent}')")
            else:
                print(f"  ✅ Verified '{item}' is in '{object_name}' (objectId: {found_object_id}).")
                passed_count += 1

    # Check state
    if expected_status != 'None':
        total_count += 1
        state_passed = False
        
        if expected_status == 'Sliced':
            is_sliced = obj_meta.get('isSliced', False)
            if is_sliced:
                print(f"  ✅ Verified '{object_name}' is Sliced.")
                state_passed = True
            else:
                print(f"  ❌ Expected '{object_name}' to be Sliced, but isSliced={is_sliced}.")
                
        elif expected_status == 'Broken':
            is_broken = obj_meta.get('isBroken', False)
            if is_broken:
                print(f"  ✅ Verified '{object_name}' is Broken.")
                state_passed = True
            else:
                print(f"  ❌ Expected '{object_name}' to be Broken, but isBroken={is_broken}.")
                
        elif expected_status == 'Opened':
            is_open = obj_meta.get('isOpen', False)
            if is_open:
                print(f"  ✅ Verified '{object_name}' is Opened (isOpen={is_open}).")
                state_passed = True
            else:
                print(f"  ❌ Expected '{object_name}' to be Opened, but isOpen={is_open}.")
                
        elif expected_status == 'Closed':
            is_open = obj_meta.get('isOpen', False)
            if not is_open:
                print(f"  ✅ Verified '{object_name}' is Closed (isOpen={is_open}).")
                state_passed = True
            else:
                print(f"  ❌ Expected '{object_name}' to be Closed, but isOpen={is_open}.")
                
        elif expected_status == 'ToggledOn':
            is_toggled = obj_meta.get('isToggled', False)
            if is_toggled:
                print(f"  ✅ Verified '{object_name}' is ToggledOn (isToggled={is_toggled}).")
                state_passed = True
            else:
                print(f"  ❌ Expected '{object_name}' to be ToggledOn, but isToggled={is_toggled}.")
                
        elif expected_status == 'ToggledOff':
            is_toggled = obj_meta.get('isToggled', False)
            if not is_toggled:
                print(f"  ✅ Verified '{object_name}' is ToggledOff (isToggled={is_toggled}).")
                state_passed = True
            else:
                print(f"  ❌ Expected '{object_name}' to be ToggledOff, but isToggled={is_toggled}.")
        else:
            print(f"  ⚠ Unknown state '{expected_status}' for '{object_name}'.")
            # 알 수 없는 상태는 통과로 처리하지 않음
        
        if state_passed:
            passed_count += 1

    return passed_count, total_count

def main():
    parser = argparse.ArgumentParser(description="Evaluate Physical Guard Results")
    parser.add_argument("--plan_file", type=str, help="Path to the result text file (optional, defaults to latest)")
    parser.add_argument("--test_file", type=str, default="data/final_test/FloorPlan1.json", help="Path to the test definition JSON")
    args = parser.parse_args()

    # resolve plan file
    # resolve plan file
    # If standard execution, load ALL plan files in the directory to cover all 5 tasks
    # The existing logic only picked the single latest file.
    
    plan_files = []
    if args.plan_file:
        plan_files = [args.plan_file]
    else:
        # Find ALL files matching pattern
        search_pattern = os.path.join(current_dir, "../results", "physical_guard_set3_result_*.txt")
        plan_files = glob.glob(search_pattern)
        if not plan_files:
             search_pattern = "results/physical_guard_set3_result_*.txt"
             plan_files = glob.glob(search_pattern)
             
    if not plan_files:
        print("❌ No plan files found.")
        return
    
    print(f"ℹ️ Found {len(plan_files)} plan files. Loading...")

    # Load resources
    print("Loading plans...")
    plans = {}
    for pf in plan_files:
        p_data = parse_plan_file(pf)
        # Merge, careful with duplicates (maybe prefer latest?)
        # For now, just update.
        plans.update(p_data)
        
    print(f"Loaded plans for {len(plans)} unique tasks.")
    if plans:
        print("Available plans:")
        for task_key in plans.keys():
            print(f"  - {task_key}")
    
    print("\nLoading expected results...")
    expected_results = load_expected_results(args.test_file)
    print(f"Expected tasks: {len(expected_results)}")
    for task_def in expected_results:
        print(f"  - {task_def['task']}")
    
    # Initialize Executor
    exe = AI2ThorExecutor(
        scene="FloorPlan1",
        headless=False, 
        save_video=False 
    )
    exe.initialize()

    results_summary = []

    for task_def in expected_results: 
        task_name = task_def['task']
        print(f"\n==================================================")
        print(f"Evaluating Task: {task_name}")
        print(f"==================================================")

        # Normalize task name for lookup
        task_key = task_name.strip().lower()
        
        # Try fuzzy matching if exact match fails
        matched_key = None
        if task_key not in plans:
            # Try partial matching
            for plan_key in plans.keys():
                # Check if task name contains plan key or vice versa
                if task_key in plan_key or plan_key in task_key:
                    matched_key = plan_key
                    print(f"  ℹ️ Using fuzzy match: '{plan_key}' for '{task_name}'")
                    break
            
            if not matched_key:
                print(f"⚠️ Plan for task '{task_name}' not available in the result file.")
                print(f"   Available plans: {list(plans.keys())}")
                print(f"   Skipping this task.")
                print(f"   💡 Tip: Run physical_guard.py to generate plans for all tasks.")
                results_summary.append({"task": task_name, "status": "SKIPPED", "reason": "No Plan"})
                continue
        else:
            matched_key = task_key

        program_code = plans[matched_key]
        
        # Reset environment for each task to ensure clean state
        exe.controller.reset(exe.scene)
        
        # Initialize agent again
        exe.controller.step(dict(
            action='Initialize',
            agentMode="default",
            snapGrid=False,
            gridSize=0.25,
            visibilityDistance=1.5,
            fieldOfView=120,
            agentCount=1
        ))
        
        # Also need to re-add cameras if needed, but for evaluation maybe not strictly necessary?
        # ManipulaThorExecutor handles camera update internally during capture.

        # Parse and execute actions
        lines = program_code.split('\n')
        execution_failed = False
        
        # Track executed actions for JSON report
        executed_actions_log = []
        
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # Need to implement Parse logic if ManipulaThorExecutor doesn't have it exposed 
            # ManipulaThorExecutor doesn't seem to have parse_action_line based on my reading earlier.
            # I must check if ManipulaThorExecutor has parse_action_line. 
            # If not, I should copy it from AI2ThorExecutor or implement it here.
            
            # Let's assume for now I need to implement simple parsing
            # Format: Action('Arg1', 'Arg2')
            
            # Simple parsing regex
            match = re.match(r"(\w+)\((.*)\)", line)
            if match:
                action = match.group(1)
                args_str = match.group(2)
                
                # Split args by comma, ignoring commas inside quotes
                # Simple hack: use eval to parse args tuple
                try:
                    # Arg string might be "'Egg', 'Bowl'"
                    # formatted as tuple
                    if args_str.strip():
                        args = eval(f"({args_str},)") # Force tuple
                    else:
                        args = ()
                    
                    target = args[0] if len(args) > 0 else None
                    receptacle = args[1] if len(args) > 1 else None
                    
                    print(f"Running: {line}")
                    
                    # Need to map action names to ManipulaThorExecutor methods if they differ
                    # Or check if ManipulaThorExecutor has execute_action method?
                    # I didn't see execute_action in the view_file output (it ended at line 800).
                    # I should assume I might need to implement mapping.
                    
                    # Let's check if the executor has the method 'action' (lowercase) or similar.
                    # Usually executors have methods like 'pickup(obj)', 'goto(obj)'.
                    
                    # Since I can't verify ManipulaThorExecutor methods right now without viewing file again,
                    # I will rely on standard methods or try to invoke controller step directly if needed.
                    # BUT AI2ThorExecutor had a nice mapping. 
                    
                    # FOR SAFETY: I will use a helper to execute. 
                    success = False
                    
                    # Mapping known plan actions
                    if hasattr(exe, action.lower()): # e.g. exe.pickup(...)
                        pass # TODO
                    
                    # Actually, let's just try to call the method on exe if it exists
                    # snake_case mapping: PickupObject -> pickup_object
                    method_name = re.sub(r'(?<!^)(?=[A-Z])', '_', action).lower()
                    
                    if hasattr(exe, method_name):
                        func = getattr(exe, method_name)
                        try:
                            if receptacle:
                                success = func(target, receptacle)
                            elif target:
                                success = func(target)
                            else:
                                success = func()
                        except Exception as e:
                            print(f"  Error executing {method_name}: {e}")
                            success = False
                    else:
                        # Fallback for some common ones
                        if action == "GoToObject" and hasattr(exe, "goto_object"):
                             success = exe.goto_object(target, target_distance=0.5, success_distance=0.85)
                        elif action == "PutObject" and hasattr(exe, "put_object"):
                             success = exe.put_object(target, receptacle)
                        elif action == "PickupObject" and hasattr(exe, "pickup_object"):
                             success = exe.pickup_object(target)
                        else:
                             print(f"  Warn: Unknown action method '{method_name}' or '{action}'")
                             success = False

                    if not success:
                        print(f"❌ Action failed: {line}")
                        execution_failed = True
                    
                    # Log action execution
                    last_event = exe.controller.last_event
                    last_action_success = last_event.metadata.get('lastActionSuccess', False) if last_event else success
                    error_message = last_event.metadata.get('errorMessage', '') if last_event else ''
                    if not success and not error_message:
                        error_message = "Action method returned False"

                    executed_actions_log.append({
                        "line": line,
                        "action": action,
                        "target_object": target,
                        "receptacle": receptacle,
                        "success": success,
                        "lastActionSuccess": last_action_success,
                        "errorMessage": error_message
                    })

                except Exception as e:
                    print(f"Error parsing/executing line '{line}': {e}")
                    execution_failed = True
                    executed_actions_log.append({
                        "line": line,
                        "error": str(e),
                        "success": False
                    })
            
        # Verify final state
        print("\n" + "="*50)
        print("Verifying object states...")
        print("="*50)
        all_passed = True
        task_passed_conds = 0
        task_total_conds = 0

        for obj_state in task_def['object_states']:
             print(f"\nChecking: {obj_state['name']}")
             p_count, t_count = verify_object_state(exe, obj_state)
             task_passed_conds += p_count
             task_total_conds += t_count
             print(f"  → Passed: {p_count}/{t_count} conditions")
        
        if task_total_conds > 0:
            gcr = (task_passed_conds / task_total_conds) * 100
            if task_passed_conds < task_total_conds:
                all_passed = False
        else:
            gcr = 0.0
            print("  ⚠ No conditions to verify.")
            
        status = "PASSED" if all_passed and not execution_failed else "FAILED"
        if execution_failed:
             status += " (Execution Errors)"
             
        print("\n" + "="*50)
        print(f"Task Result: {status}")
        print(f"GCR (Goal Condition Rate): {gcr:.1f}% ({task_passed_conds}/{task_total_conds} conditions passed)")
        print("="*50)
        results_summary.append({
            "task": task_name, 
            "status": status,
            "gcr": round(gcr, 2),
            "passed_conditions": task_passed_conds,
            "total_conditions": task_total_conds,
            "verification_passed": all_passed,
            "execution_clean": not execution_failed
        })

        # Save execution result to JSON
        safe_task_name = re.sub(r'[^a-zA-Z0-9]', '_', task_name)
        result_json_filename = f"execution_result_{safe_task_name}.json"
        
        final_meta = exe.controller.last_event.metadata.copy() if exe.controller and exe.controller.last_event else {}
        # Remove heavy fields like image data from metadata to save space
        if "thirdPartyCameraFrames" in final_meta: del final_meta["thirdPartyCameraFrames"]
        if "third_party_camera_frames" in final_meta: del final_meta["third_party_camera_frames"]

        execution_data = {
            "episode_id": task_name,
            "instruction": task_name,
            "executed_actions": executed_actions_log,
            "final_metadata": final_meta,
            "failed_guards": [], # Placeholder as we are evaluating execution, not generating plans
            "status": status,
            "gcr": round(gcr, 2),
            "passed_conditions": task_passed_conds,
            "total_conditions": task_total_conds,
            "expected_object_states": task_def.get('object_states', []),
            "verification_passed": all_passed,
            "execution_clean": not execution_failed
        }
        
        with open(result_json_filename, "w", encoding='utf-8') as f:
            json.dump(execution_data, f, indent=2, ensure_ascii=False)
        print(f"Saved execution log to: {result_json_filename}")

    # Print Summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    total_gcr = 0
    count_gcr = 0
    total_passed_conds = 0
    total_conds = 0
    
    for res in results_summary:
        if 'gcr' in res:
            status_icon = "✅" if res['status'] == "PASSED" else "❌"
            print(f"{status_icon} {res['task']}")
            print(f"   Status: {res['status']}")
            print(f"   GCR: {res['gcr']:.1f}%")
            if 'passed_conditions' in res and 'total_conditions' in res:
                print(f"   Conditions: {res.get('passed_conditions', 0)}/{res.get('total_conditions', 0)} passed")
            print()
            total_gcr += res['gcr']
            count_gcr += 1
        else:
            print(f"⚠️  {res['task']}: {res['status']}")
            print()
            
    if count_gcr > 0:
        avg_gcr = total_gcr / count_gcr
        print("-" * 70)
        print(f"AVERAGE GCR: {avg_gcr:.1f}%")
        print(f"Tasks Evaluated: {count_gcr}")
        print("=" * 70)

if __name__ == "__main__":
    main()
