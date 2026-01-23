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

def parse_json_plan_file(file_path: str) -> Dict[str, str]:
    """
    Parses the JSON plan file to extract tasks and their corresponding programs.
    Returns a dictionary mapping task names to program code.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        plan_dict = json.load(f)
    
    plans = {}
    for task_name, program_code in plan_dict.items():
        # Normalize task name to lowercase for robust matching
        normalized_task = task_name.strip().lower()
        plans[normalized_task] = program_code
    
    return plans

def parse_plan_file(file_path: str) -> Dict[str, str]:
    """
    Parses the plan file (txt) to extract tasks and their corresponding programs.
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
                
        elif expected_status == 'ToggledOn' or expected_status == 'On':
            is_toggled = obj_meta.get('isToggled', False)
            if is_toggled:
                print(f"  ✅ Verified '{object_name}' is {'On' if expected_status == 'On' else 'ToggledOn'} (isToggled={is_toggled}).")
                state_passed = True
            else:
                print(f"  ❌ Expected '{object_name}' to be {'On' if expected_status == 'On' else 'ToggledOn'}, but isToggled={is_toggled}.")
                
        elif expected_status == 'ToggledOff' or expected_status == 'Off':
            is_toggled = obj_meta.get('isToggled', False)
            if not is_toggled:
                print(f"  ✅ Verified '{object_name}' is {'Off' if expected_status == 'Off' else 'ToggledOff'} (isToggled={is_toggled}).")
                state_passed = True
            else:
                print(f"  ❌ Expected '{object_name}' to be {'Off' if expected_status == 'Off' else 'ToggledOff'}, but isToggled={is_toggled}.")
        else:
            print(f"  ⚠ Unknown state '{expected_status}' for '{object_name}'.")
            # 알 수 없는 상태는 통과로 처리하지 않음
        
        if state_passed:
            passed_count += 1

    return passed_count, total_count

def execute_and_evaluate_task(
    executor: AI2ThorExecutor,
    task_def: Dict[str, Any],
    program_code: str,
    method_name: str = "Unknown"
) -> Dict[str, Any]:
    """
    Execute a task and evaluate the results.
    Returns evaluation result dictionary.
    """
    task_name = task_def['task']
    print(f"\n[{method_name}] Evaluating Task: {task_name}")
    print("="*50)
    
    # Reset environment for each task to ensure clean state
    executor.controller.reset(executor.scene)
    
    # Initialize agent again
    executor.controller.step(dict(
        action='Initialize',
        agentMode="default",
        snapGrid=False,
        gridSize=0.25,
        visibilityDistance=1.5,
        fieldOfView=120,
        agentCount=1
    ))
    
    # Parse and execute actions
    lines = program_code.split('\n')
    execution_failed = False
    
    # Track executed actions for JSON report
    executed_actions_log = []
    
    for line in lines:
        line = line.strip()
        if not line: 
            continue
        if line.startswith("#"):
            continue  # Skip comments
        
        # Simple parsing regex
        match = re.match(r"(\w+)\((.*)\)", line)
        if match:
            action = match.group(1)
            args_str = match.group(2)
            
            # Split args by comma, ignoring commas inside quotes
            try:
                if args_str.strip():
                    args = eval(f"({args_str},)") # Force tuple
                else:
                    args = ()
                
                target = args[0] if len(args) > 0 else None
                receptacle = args[1] if len(args) > 1 else None
                
                print(f"  Running: {line}")
                
                success = False
                
                # snake_case mapping: PickupObject -> pickup_object
                method_name_action = re.sub(r'(?<!^)(?=[A-Z])', '_', action).lower()
                
                if hasattr(executor, method_name_action):
                    func = getattr(executor, method_name_action)
                    try:
                        if receptacle:
                            success = func(target, receptacle)
                        elif target:
                            success = func(target)
                        else:
                            success = func()
                    except Exception as e:
                        print(f"    Error executing {method_name_action}: {e}")
                        success = False
                else:
                    # Fallback for some common ones
                    if action == "GoToObject" and hasattr(executor, "goto_object"):
                         success = executor.goto_object(target, target_distance=0.5, success_distance=1.0)
                    elif action == "PutObject" and hasattr(executor, "put_object"):
                         success = executor.put_object(target, receptacle)
                    elif action == "PickupObject" and hasattr(executor, "pickup_object"):
                         success = executor.pickup_object(target)
                    elif action == "OpenObject" and hasattr(executor, "open_object"):
                         success = executor.open_object(target)
                    elif action == "CloseObject" and hasattr(executor, "close_object"):
                         success = executor.close_object(target)
                    elif action == "SliceObject" and hasattr(executor, "slice_object"):
                         success = executor.slice_object(target)
                    elif action == "BreakObject" and hasattr(executor, "break_object"):
                         success = executor.break_object(target)
                    elif action == "ToggleObjectOn" and hasattr(executor, "toggle_on"):
                         success = executor.toggle_on(target)
                    elif action == "ToggleObjectOff" and hasattr(executor, "toggle_off"):
                         success = executor.toggle_off(target)
                    else:
                         print(f"    Warn: Unknown action method '{method_name_action}' or '{action}'")
                         success = False

                if not success:
                    print(f"    ❌ Action failed: {line}")
                    execution_failed = True
                
                # Log action execution
                last_event = executor.controller.last_event
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
                print(f"    Error parsing/executing line '{line}': {e}")
                execution_failed = True
                executed_actions_log.append({
                    "line": line,
                    "error": str(e),
                    "success": False
                })
    
    # Verify final state
    print(f"\n[{method_name}] Verifying object states...")
    print("="*50)
    all_passed = True
    task_passed_conds = 0
    task_total_conds = 0

    # object_states 키가 있는지 확인
    object_states = task_def.get('object_states', [])
    if not object_states:
        print(f"  ⚠️  No object_states defined for task '{task_name}'")
        print(f"    → Skipping verification")
        gcr = 0.0
        task_total_conds = 0
        task_passed_conds = 0
    else:
        for obj_state in object_states:
            print(f"  Checking: {obj_state['name']}")
            p_count, t_count = verify_object_state(executor, obj_state)
            task_passed_conds += p_count
            task_total_conds += t_count
            print(f"    → Passed: {p_count}/{t_count} conditions")
        
        if task_total_conds > 0:
            gcr = (task_passed_conds / task_total_conds) * 100
            if task_passed_conds < task_total_conds:
                all_passed = False
        else:
            gcr = 0.0
            print("    ⚠ No conditions to verify.")
        
    status = "PASSED" if all_passed and not execution_failed else "FAILED"
    if execution_failed:
         status += " (Execution Errors)"
         
    print(f"\n[{method_name}] Task Result: {status}")
    print(f"[{method_name}] GCR: {gcr:.1f}% ({task_passed_conds}/{task_total_conds} conditions passed)")
    
    return {
        "task": task_name,
        "method": method_name,
        "status": status,
        "gcr": round(gcr, 2),
        "passed_conditions": task_passed_conds,
        "total_conditions": task_total_conds,
        "verification_passed": all_passed,
        "execution_clean": not execution_failed,
        "executed_actions": executed_actions_log
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Physical Guard and Baseline Results")
    parser.add_argument("--physical_guard_json", type=str, help="Path to physical_guard_result JSON file (optional)")
    parser.add_argument("--baseline_json", type=str, help="Path to baseline_result JSON file (optional)")
    parser.add_argument("--test_file", type=str, default="data/final_test/FloorPlan1.json", help="Path to the test definition JSON")
    parser.add_argument("--folder", type=str, default="Kitchen", 
                       choices=["Kitchen", "LivingRoom", "BedRoom", "BathRoom"],
                       help="Folder name in results/ directory to search for JSON files (default: Kitchen)")
    parser.add_argument("--fp-number", type=str, default=None,
                       help="FP number (e.g., 1, 2, 216, 224, 325, 326, 403, 425). If not provided, will prompt for input.")
    args = parser.parse_args()

    # Get FP number
    fp_number = args.fp_number
    if not fp_number:
        try:
            fp_number = input(f"📋 FP 번호를 입력하세요 (예: 1, 2, 216, 224, 325, 326, 403, 425): ").strip()
            if not fp_number:
                print("❌ FP 번호가 입력되지 않았습니다.")
                return
        except (KeyboardInterrupt, EOFError):
            print("\n❌ 입력이 취소되었습니다.")
            return
    
    # Auto-select test_file based on FP number if not specified
    if not args.test_file or args.test_file == "data/final_test/FloorPlan1.json":
        test_file_path = os.path.join("data/final_test", f"FloorPlan{fp_number}.json")
        if os.path.exists(test_file_path):
            args.test_file = test_file_path
            print(f"📋 FP{fp_number}에 맞는 test_file 자동 선택: {args.test_file}")
        else:
            print(f"⚠️  Warning: {test_file_path} not found, using default: {args.test_file}")
    else:
        print(f"📋 Using specified test file: {args.test_file}")
    
    # Find JSON files in the specified folder/FP{num} directory
    folder_name = args.folder
    fp_folder = f"FP{fp_number}"
    print(f"📂 Searching in results/{folder_name}/{fp_folder} folder...")
    
    # Try absolute path first
    folder_physical_guard_pattern = os.path.join(current_dir, "../results", folder_name, fp_folder, "physical_guard_result_*.json")
    folder_baseline_pattern = os.path.join(current_dir, "../results", folder_name, fp_folder, "baseline_result_*.json")
    
    physical_guard_files = glob.glob(folder_physical_guard_pattern)
    baseline_files = glob.glob(folder_baseline_pattern)
    
    # Try relative path if absolute path didn't work
    if not physical_guard_files:
        folder_physical_guard_pattern = os.path.join("results", folder_name, fp_folder, "physical_guard_result_*.json")
        physical_guard_files = glob.glob(folder_physical_guard_pattern)
    
    if not baseline_files:
        folder_baseline_pattern = os.path.join("results", folder_name, fp_folder, "baseline_result_*.json")
        baseline_files = glob.glob(folder_baseline_pattern)
    
    # Fallback: Check parent folder (results/{folder_name}/) if FP{num} folder has no files
    if not physical_guard_files:
        search_pattern = os.path.join(current_dir, "../results", folder_name, "physical_guard_result_*.json")
        physical_guard_files = glob.glob(search_pattern)
        if not physical_guard_files:
            search_pattern = os.path.join("results", folder_name, "physical_guard_result_*.json")
            physical_guard_files = glob.glob(search_pattern)
    
    if not baseline_files:
        search_pattern = os.path.join(current_dir, "../results", folder_name, "baseline_result_*.json")
        baseline_files = glob.glob(search_pattern)
        if not baseline_files:
            search_pattern = os.path.join("results", folder_name, "baseline_result_*.json")
            baseline_files = glob.glob(search_pattern)
    
    # Final fallback: Check results folder directly
    if not physical_guard_files:
        search_pattern = os.path.join(current_dir, "../results", "physical_guard_result_*.json")
        physical_guard_files = glob.glob(search_pattern)
        if not physical_guard_files:
            search_pattern = "results/physical_guard_result_*.json"
            physical_guard_files = glob.glob(search_pattern)
    
    if not baseline_files:
        search_pattern = os.path.join(current_dir, "../results", "baseline_result_*.json")
        baseline_files = glob.glob(search_pattern)
        if not baseline_files:
            search_pattern = "results/baseline_result_*.json"
            baseline_files = glob.glob(search_pattern)
    
    # Use command line arguments if provided
    if args.physical_guard_json:
        physical_guard_files = [args.physical_guard_json]
    if args.baseline_json:
        baseline_files = [args.baseline_json]
    
    # Get latest files if multiple found
    physical_guard_file = None
    baseline_file = None
    
    if physical_guard_files:
        physical_guard_file = max(physical_guard_files, key=os.path.getmtime)
        print(f"📁 Physical Guard JSON: {physical_guard_file}")
    
    if baseline_files:
        baseline_file = max(baseline_files, key=os.path.getmtime)
        print(f"📁 Baseline JSON: {baseline_file}")
    
    if not physical_guard_file and not baseline_file:
        print("❌ No JSON files found.")
        print(f"   Searched for:")
        print(f"     - results/{folder_name}/{fp_folder}/physical_guard_result_*.json")
        print(f"     - results/{folder_name}/{fp_folder}/baseline_result_*.json")
        print(f"     - results/{folder_name}/physical_guard_result_*.json")
        print(f"     - results/{folder_name}/baseline_result_*.json")
        print(f"     - results/physical_guard_result_*.json")
        print(f"     - results/baseline_result_*.json")
        print(f"\n   Available folders in results/{folder_name}/:")
        folder_dir = os.path.join(current_dir, "../results", folder_name)
        if os.path.exists(folder_dir):
            for item in os.listdir(folder_dir):
                item_path = os.path.join(folder_dir, item)
                if os.path.isdir(item_path):
                    print(f"     - {item}")
        return
    
    # Load plans
    physical_guard_plans = {}
    baseline_plans = {}
    
    if physical_guard_file:
        print(f"\n📖 Loading Physical Guard plans from: {physical_guard_file}")
        physical_guard_plans = parse_json_plan_file(physical_guard_file)
        print(f"   Loaded {len(physical_guard_plans)} tasks")
    
    if baseline_file:
        print(f"\n📖 Loading Baseline plans from: {baseline_file}")
        baseline_plans = parse_json_plan_file(baseline_file)
        print(f"   Loaded {len(baseline_plans)} tasks")
    
    print("\n📋 Loading expected results...")
    expected_results = load_expected_results(args.test_file)
    print(f"   Expected tasks: {len(expected_results)}")
    for task_def in expected_results:
        print(f"     - {task_def['task']}")
    
    # Initialize Executors
    print("\n🤖 Initializing AI2-THOR executors...")
    physical_guard_exe = None
    baseline_exe = None
    
    # Determine scene name from FP number
    scene_name = f"FloorPlan{fp_number}"
    
    if physical_guard_plans:
        physical_guard_exe = AI2ThorExecutor(
            scene=scene_name,
            headless=False, 
            save_video=False 
        )
        physical_guard_exe.initialize()
        print(f"   ✓ Physical Guard executor initialized (Scene: {scene_name})")
    
    if baseline_plans:
        baseline_exe = AI2ThorExecutor(
            scene=scene_name,
            headless=False, 
            save_video=False 
        )
        baseline_exe.initialize()
        print(f"   ✓ Baseline executor initialized (Scene: {scene_name})")
    
    # Evaluate tasks
    physical_guard_results = []
    baseline_results = []
    
    for task_def in expected_results:
        task_name = task_def['task']
        task_key = task_name.strip().lower()
        
        print(f"\n{'='*70}")
        print(f"TASK: {task_name}")
        print(f"{'='*70}")
        
        # Evaluate Physical Guard
        if physical_guard_plans and physical_guard_exe:
            matched_key = None
            if task_key in physical_guard_plans:
                matched_key = task_key
            else:
                # Try fuzzy matching
                for plan_key in physical_guard_plans.keys():
                    if task_key in plan_key or plan_key in task_key:
                        matched_key = plan_key
                        print(f"  ℹ️ Using fuzzy match: '{plan_key}' for Physical Guard")
                        break
            
            if matched_key:
                program_code = physical_guard_plans[matched_key]
                result = execute_and_evaluate_task(
                    physical_guard_exe, task_def, program_code, "Physical Guard"
                )
                physical_guard_results.append(result)
            else:
                print(f"  ⚠️ Physical Guard: Plan not found for '{task_name}'")
                physical_guard_results.append({
                    "task": task_name,
                    "method": "Physical Guard",
                    "status": "SKIPPED",
                    "reason": "No Plan"
                })
        
        # Evaluate Baseline
        if baseline_plans and baseline_exe:
            matched_key = None
            if task_key in baseline_plans:
                matched_key = task_key
            else:
                # Try fuzzy matching
                for plan_key in baseline_plans.keys():
                    if task_key in plan_key or plan_key in task_key:
                        matched_key = plan_key
                        print(f"  ℹ️ Using fuzzy match: '{plan_key}' for Baseline")
                        break
            
            if matched_key:
                program_code = baseline_plans[matched_key]
                result = execute_and_evaluate_task(
                    baseline_exe, task_def, program_code, "Baseline"
                )
                baseline_results.append(result)
            else:
                print(f"  ⚠️ Baseline: Plan not found for '{task_name}'")
                baseline_results.append({
                    "task": task_name,
                    "method": "Baseline",
                    "status": "SKIPPED",
                    "reason": "No Plan"
                })
    
    # Print Comparison Summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY - COMPARISON")
    print("="*70)
    
    # Physical Guard Summary
    if physical_guard_results:
        print("\n📊 PHYSICAL GUARD RESULTS:")
        print("-" * 70)
        total_gcr = 0
        count_gcr = 0
        total_passed_conds = 0
        total_conds = 0
        
        for res in physical_guard_results:
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
                total_passed_conds += res.get('passed_conditions', 0)
                total_conds += res.get('total_conditions', 0)
            else:
                print(f"⚠️  {res['task']}: {res.get('status', 'SKIPPED')}")
                print()
        
        if count_gcr > 0:
            avg_gcr = total_gcr / count_gcr
            print(f"   Average GCR: {avg_gcr:.1f}%")
            print(f"   Overall Conditions: {total_passed_conds}/{total_conds} passed")
    
    # Baseline Summary
    if baseline_results:
        print("\n📊 BASELINE RESULTS:")
        print("-" * 70)
        total_gcr = 0
        count_gcr = 0
        total_passed_conds = 0
        total_conds = 0
        
        for res in baseline_results:
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
                total_passed_conds += res.get('passed_conditions', 0)
                total_conds += res.get('total_conditions', 0)
            else:
                print(f"⚠️  {res['task']}: {res.get('status', 'SKIPPED')}")
                print()
        
        if count_gcr > 0:
            avg_gcr = total_gcr / count_gcr
            print(f"   Average GCR: {avg_gcr:.1f}%")
            print(f"   Overall Conditions: {total_passed_conds}/{total_conds} passed")
    
    # Side-by-side Comparison
    if physical_guard_results and baseline_results:
        print("\n📊 SIDE-BY-SIDE COMPARISON:")
        print("=" * 70)
        print(f"{'Task':<40} {'Physical Guard':<20} {'Baseline':<20}")
        print("-" * 70)
        
        # Create task mapping
        pg_dict = {r['task']: r for r in physical_guard_results}
        bl_dict = {r['task']: r for r in baseline_results}
        
        all_tasks = set(pg_dict.keys()) | set(bl_dict.keys())
        
        for task in sorted(all_tasks):
            pg_res = pg_dict.get(task, {})
            bl_res = bl_dict.get(task, {})
            
            pg_gcr = pg_res.get('gcr', 0) if 'gcr' in pg_res else None
            bl_gcr = bl_res.get('gcr', 0) if 'gcr' in bl_res else None
            
            pg_str = f"{pg_gcr:.1f}%" if pg_gcr is not None else "N/A"
            bl_str = f"{bl_gcr:.1f}%" if bl_gcr is not None else "N/A"
            
            # Highlight winner
            if pg_gcr is not None and bl_gcr is not None:
                if pg_gcr > bl_gcr:
                    pg_str = f"🏆 {pg_str}"
                elif bl_gcr > pg_gcr:
                    bl_str = f"🏆 {bl_str}"
            
            print(f"{task[:38]:<40} {pg_str:<20} {bl_str:<20}")
        
        print("=" * 70)

if __name__ == "__main__":
    main()
