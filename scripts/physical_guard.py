# 논리적 검증 마친 plan을 물리적 검증 수행 -> 논리적 검증 방법처럼 assert/else 사용해서 사전 조건 만족하도록.
# 각 액션별로 필요한 물리적 사전 조건 정의
# GoToObject(object)의 경우: 실제로 NavMesh내 이동 가능한 경로가 있는지를 검증 (NAVIGABLE), object가 실제 Scene에 존재하는가 (EXISTS)
# PickupObject(object)의 경우: 현재 agent가 아무 객체도 hold하고 있지 않은가 (¬HOLDS), object가 실제 Scene에 존재하는가 (EXISTS), object가 pickupable인가 (PICKUPABLE), object와 agent의 손까지의 거리가 1.39m 이내에 있는가 (REACHABLE), object의 parentReceptacle이 존재하는가 (EXISTS), openable한가 (OPENABLE), opened인가 (OPENED)
# PutObject(object, receptacle)의 경우: 현재 agent가 목표 객체를 hold하고 있는가 (HOLDS), receptacle가 실제로 receptacle 타입인가 (RECEPTACLE), receptacle가 openable인가 (OPENABLE), opened되어있는가 (OPENED), 현재 agent 손과 receptacle까지의 거리가 1.39m 이내에 있는가 (REACHABLE)
# OpenObject(object)의 경우: object가 openable한가 (OPENABLE), opened되어있지 않은가 (¬OPENED), object가 실제 Scene에 존재하는가 (EXISTS), object와 agent의 손까지의 거리가 1.39m 이내에 있는가 (REACHABLE)
# CloseObject(object)의 경우: object가 openable한가 (OPENABLE), opened되어있는가 (OPENED), object가 실제 Scene에 존재하는가 (EXISTS), object와 agent의 손까지의 거리가 1.39m 이내에 있는가 (REACHABLE)


#!/usr/bin/env python3
"""
ProgPrompt 스타일의 AI2-THOR FloorPlan1 플래너 (Ollama llama3 사용).

이 스크립트는:
1. 미리 정의된 객체 및 액션 목록을 사용합니다.
2. ProgPrompt 논문에서 영감을 받은 프로그래밍적 프롬프트를 구성합니다
   (코드 스켈레톤 + 어설션 + 자연어 주석 + 복구 단계 포함).
3. OpenAI 호환 HTTP 인터페이스를 통해 Ollama에서 호스팅되는 Llama 3 모델에 쿼리합니다.
4. 각 작업에 대한 Python 스타일 프로그램을 생성하고 results/ 디렉토리에 저장합니다.
"""

# 표준 라이브러리 임포트
import argparse  # 명령줄 인자 파싱을 위한 모듈
import json  # JSON 파일 읽기/쓰기를 위한 모듈
import logging  # 로깅을 위한 모듈
import math  # 수학 함수 (삼각함수 등)
import os  # 운영체제 관련 기능 (디렉토리 생성 등)
import random  # 랜덤 시드 설정을 위한 모듈
import re  # 정규표현식을 위한 모듈
import shutil  # 파일 복사를 위한 모듈
import subprocess  # 외부 스크립트 실행을 위한 모듈
import sys  # 시스템 관련 기능 (프로그램 종료 등)
import textwrap  # 텍스트 들여쓰기 및 포맷팅을 위한 모듈
from datetime import datetime  # 타임스탬프 생성을 위한 날짜/시간 모듈
from pathlib import Path  # 파일 경로 처리를 위한 모듈
from typing import Dict, List, Optional, Tuple, Any  # 타입 힌팅을 위한 모듈

# 로컬 모듈 임포트
from ai2thor_connector_manipulathor import ManipulaThorExecutor  # AI2-THOR 환경 실행을 위한 커넥터

# Scene Graph Extractor 함수들 임포트
try:
    from scene_graph_extractor import (
        find_target_object,
        get_related_edges,
        load_scene_graph as load_scene_graph_extractor
    )
except ImportError:
    # scene_graph_extractor가 같은 디렉토리에 없을 경우를 대비
    find_target_object = None
    get_related_edges = None
    load_scene_graph_extractor = None

# 외부 라이브러리 임포트
from openai import OpenAI  # OpenAI 호환 API 클라이언트 (Ollama와 통신)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_info_txt(info_file_path: str) -> Tuple[List[str], List[str]]:
    """
    info.txt 파일을 파싱하여 액션과 객체 목록을 추출합니다.
    
    매개변수:
        info_file_path: info.txt 파일의 경로
    
    반환값:
        tuple: (액션_리스트, 객체_리스트) 형태의 튜플
    """
    actions = []  # 액션 목록을 저장할 리스트
    objects = []  # 객체 목록을 저장할 리스트
    
    # info.txt 파일을 읽어서 각 줄을 처리
    with open(info_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    current_section = None  # 현재 파싱 중인 섹션 (actions 또는 objects)
    for line in lines:
        line = line.strip()  # 앞뒤 공백 제거
        if not line:  # 빈 줄은 건너뛰기
            continue
        
        # "ACTION TEMPLATES" 섹션 시작을 감지
        if "ACTION TEMPLATES" in line:
            current_section = "actions"
            continue
        # "OBJECT and PROPERTIES" 또는 "OBJECTS and PROPERTIES" 섹션 시작을 감지
        elif "OBJECT and PROPERTIES" in line or "OBJECTS and PROPERTIES" in line:
            current_section = "objects"
            continue
        
        # 액션 섹션 처리
        if current_section == "actions":
            # 빈 줄과 섹션 헤더는 건너뛰기
            if line and not line.startswith("ACTION") and line != "done":
                # 형식 변환: "GoTo <obj>" -> "GoTo <obj>              # 객체 근처로 이동"
                action_name = line.split()[0] if line.split() else ""
                if action_name:
                    # 액션 타입에 따라 설명 추가
                    # 각 액션 타입에 대한 설명 딕셔너리
                    descriptions = {
                        "GoToObject": "Navigate close to an object",
                        "PickupObject": "Pick up a pickupable object (agent can hold only ONE object at a time)",
                        "PutObject": "Place held object inside/on receptacle",
                        "OpenObject": "Open openable container/appliance",
                        "CloseObject": "Close openable object",
                        "ToggleObjectOn": "Turn on a switchable object",
                        "ToggleObjectOff": "Turn off a switchable object",  
                        "BreakObject": "Break a breakable object",
                        "SliceObject": "Slice a sliceable object",
                        "CookObject": "Cook a cookable object"
                    }
                    desc = descriptions.get(action_name, "")
                    if desc:
                        # 액션 이름과 설명을 함께 저장 (30자 너비로 정렬)
                        actions.append(f"{line:<30} # {desc}")
                    else:
                        # 설명이 없으면 원본 라인 그대로 저장
                        actions.append(line)
        
        # 객체 섹션 처리
        elif current_section == "objects":
            # 형식: "Apple: pickupable" -> "Apple" 추출
            if ":" in line:
                obj_name = line.split(":")[0].strip()  # 콜론 앞부분만 추출
                if obj_name:
                    objects.append(obj_name)  # 객체 이름을 리스트에 추가
    
    return actions, objects


# 기본 폴백 값들 (info.txt 파일이 있으면 덮어씌워짐)
# AI2-THOR에서 사용 가능한 액션 목록
AI2THOR_ACTIONS = [
    "GoToObject <obj>              # Navigate close to an object",
    "PickupObject <obj>            # Pick up a pickupable object (agent can hold only ONE object at a time)",
    "PutObject <obj> <recp>        # Place held object inside/on receptacle",
    "OpenObject <obj>              # Open openable container/appliance",
    "CloseObject <obj>             # Close openable object",
    "ToggleObjectOn <obj>          # Turn on a switchable object",
    "ToggleObjectOff <obj>         # Turn off a switchable object",
    "BreakObject <obj>             # Break a breakable object",
    "SliceObject <obj>             # Slice a sliceable object",
    "CookObject <obj>              # Cook a cookable object"
]

# FloorPlan1 씬에서 사용 가능한 기본 객체 목록 (info.txt에서 추출)
DEFAULT_FLOORPLAN1_OBJECTS = [
    "AlarmClock", "Apple", "AppleSliced", "ArmChair", "BaseballBat", "BasketBall", "Bathtub", "BathtubBasin",
    "Bed", "Blinds", "Book", "Boots", "Bottle", "Bowl", "Box", "Bread", "BreadSliced",
    "Cabinet", "Candle", "Cart", "CD", "CellPhone", "Chair", "Cloth", "CoffeeMachine", "CoffeTable",
    "CounterTop", "CreditCard", "Cup", "Curtains", "Desk", "DeskLamp", "DiningTable", "DishSponge",
    "Drawer", "Dresser", "Egg", "EggCracked", "Faucet", "FloorLamp", "Footstool", "Fork", "Fridge",
    "GarbageCan", "HandTowel", "HanTowelHolder", "HousePlant", "Kettle", "KeyChain", "Knife", "Ladle",
    "Laptop", "LaundryHamper", "LaundryHamperLid", "Lettuce", "LettuceSliced", "LightSwitch", "Microwave",
    "Mirror", "Mug", "Newspaper", "Ottoman", "Painting", "Pan", "PaperTowel", "Pen", "Pencil",
    "PepperShaker", "Pillow", "Plate", "Plunger", "Poster", "Pot", "Potato", "PotatoSliced",
    "RemoteControl", "Safe", "SaltShaker", "ScrubBrush", "Shelf", "ShowerCurtain", "ShowerDoor",
    "ShowerGlass", "ShowerHead", "SideTable", "Sink", "SinkBasin", "SoapBar", "SoapBottle", "Sofa",
    "Spatula", "Spoon", "SprayBottle", "Statue", "StoveBurner", "StoveKnob", "TeddyBear", "Television",
    "TennisRacket", "TissueBox", "Toaster", "Toilet", "ToiletPaper", "ToiletPaperHanger", "Tomato",
    "TomatoSliced", "Towel", "TowelHolder", "TVStand", "Vase", "Watch", "WateringCan", "Window",
    "WineBottle",
]

# LLM 프롬프트에 포함될 예제 프로그램들 (작업별로 Python 스타일 코드)
# 각 예제는 ProgPrompt 형식을 따르며, 어설션과 복구 단계를 포함한 완전한 프로그램 구조를 보여줌
DEFAULT_EXAMPLES = {
    # 예제 1: 사과를 냉장고에 넣기
    "put_apple_in_fridge": textwrap.dedent(
        """\
        def put_apple_in_fridge():
        \t# Step 1: Find an apple and pick it up
        \tassert('Apple' visible)
        \t\telse: GoToObject('Apple')
        \tGoToObject('Apple')
        \tassert('close' to 'Apple')
        \t\telse: GoToObject('Apple')
        \tPickupObject('Apple')
        \t# Step 2: Move to the fridge and open it
        \tGoToObject('Fridge')
        \tassert('close' to 'Fridge')
        \t\telse: GoToObject('Fridge')
        \tassert('Fridge' is 'closed')
        \t\telse: CloseObject('Fridge')
        \tOpenObject('Fridge')
        \t# Step 3: Place the apple inside and close the door
        \tassert('Apple' in 'hands')
        \t\telse: GoToObject('Apple')
        \t\telse: PickupObject('Apple')
        \tassert('Fridge' is 'opened')
        \t\telse: OpenObject('Fridge')
        \tPutObject('Apple', 'Fridge')
        \tCloseObject('Fridge')
        """
    ),
    # 예제 2: 빵 자르기
    "slice_bread": textwrap.dedent(
        """\
        def slice_bread():
        \t# Step 1: Retrieve the knife
        \tGoToObject('Knife')
        \tassert('close' to 'Knife')
        \t\telse: GoToObject('Knife')
        \tPickupObject('Knife')
        \t# Step 2: Locate the bread loaf
        \tGoToObject('Bread')
        \t# Step 3: Slice the bread with precondition checks
        \tassert('Knife' in 'hands')
        \t\telse: GoToObject('Knife')
        \t\telse: PickupObject('Knife')
        \tassert('close' to 'Bread')
        \t\telse: GoToObject('Bread')
        \tSliceObject('Bread')
        """
    ),
    # 예제 3: 머그잔 씻기
    "wash_mug": textwrap.dedent(
        """\
        def wash_mug():
        \t# Step 1: Find and pick up the mug
        \tGoToObject('Mug')
        \tassert('close' to 'Mug')
        \t\telse: GoToObject('Mug')
        \tPickupObject('Mug')
        \t# Step 2: Move to the sink and drop the mug inside
        \tGoToObject('Sink')
        \tassert('close' to 'Sink')
        \t\telse: GoToObject('Sink')
        \tPutObject('Mug', 'SinkBasin')
        \t# Step 3: Turn on the faucet and clean
        \tGoToObject('Faucet')
        \tassert('Faucet' is 'off')
        \t\telse: ToggleObjectOff('Faucet')
        \tToggleObjectOn('Faucet')
        \tGoToObject('Mug')
        \tCleanObject('Mug')
        \t# Step 4: Turn off the faucet
        \tGoToObject('Faucet')
        \tToggleObjectOff('Faucet')
        """
    ),
    # 예제 4: Mug를 깨뜨리기
    "break_mug": textwrap.dedent(
        """\
        def break_mug():
        \t# Step 1: Find and pick up the mug
        \tGoToObject('Mug')
        \tassert('close' to 'Mug')
        \t\telse: GoToObject('Mug')
        \tPickupObject('Mug')
        \t# Step 2: Break the mug
        \tassert('Mug' in 'hands')
        \t\telse: GoToObject('Mug')
        \t\telse: PickupObject('Mug')
        \tBreakObject('Mug')
        """
    ),
    # 예제 5: Cook the Egg
    "cook_egg": textwrap.dedent(
        """\
        def cook_egg():
        \t# Step 1: Find and pick up the egg
        \tGoToObject('Egg')
        \tPickupObject('Egg')
        \t# Step 2: Put the egg in the pan
        \tGoToObject('Pan')
        \tPutObject('Egg', 'Pan')
        \t# Step 3: Break the egg
        \tBreakObject('Egg')
        \t# Step 4: Pick up the pan and put it on the stove burner
        \tPickupObject('Pan')
        \tGoToObject('StoveBurner')
        \tPutObject('Pan', 'StoveBurner')
        \t# Step 5: Turn on the stove knob and cook the egg
        \tToggleObjectOn('StoveKnob')
        \tCookObject('EggCracked')
        \tToggleObjectOff('StoveKnob')
        """
    )
}


def load_task_list(task_file: Optional[str], inline_tasks: List[str]) -> List[str]:
    """
    파일이나 인라인 리스트에서 작업 목록을 로드합니다.
    
    매개변수:
        task_file: 작업 목록이 담긴 JSON/JSONL 파일 경로 (선택사항)
        inline_tasks: 명령줄에서 직접 제공된 작업 목록
    
    반환값:
        List[str]: 모든 작업을 포함한 리스트
    """
    # 인라인으로 제공된 작업들을 먼저 리스트에 추가
    tasks = list(inline_tasks)
    
    # 파일이 제공되지 않았으면 인라인 작업만 반환
    if not task_file:
        return tasks

    # 파일 전체를 읽어 JSON 파싱 시도
    try:
        with open(task_file, "r", encoding="utf-8") as f:
            whole_data = json.load(f)
            
            if isinstance(whole_data, list):
                for item in whole_data:
                    if isinstance(item, dict) and "task" in item:
                        tasks.append(item["task"])
                    elif isinstance(item, str):
                        tasks.append(item)
                    # Handle other cases if necessary
                return tasks
            elif isinstance(whole_data, dict):
                # If dict, maybe keys are tasks or specific field
                # For now retain legacy behavior: keys as tasks
                tasks.extend(whole_data.keys())
                return tasks
    except json.JSONDecodeError:
        # Not a valid single JSON file, try line-by-line (JSONL or Text)
        pass

    # 파일에서 작업 목록 읽기 (Line-by-line fallback)
    with open(task_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # 앞뒤 공백 제거
            if not line:  # 빈 줄은 건너뛰기
                continue
            
            # Skip lines that are just JSON array brackets if checking line-by-line failed
            if line in ['[', ']']: 
                continue

            try:
                # JSON 형식으로 파싱 시도
                data = json.loads(line)
            except json.JSONDecodeError:
                # JSON이 아니면 일반 텍스트로 처리하여 작업 목록에 추가
                # Remove trailing commas if it looks like a JSON list item
                if line.endswith(','):
                    line = line[:-1]
                tasks.append(line)
                continue
            
            # 파싱된 데이터 타입에 따라 처리
            if isinstance(data, dict):
                if "task" in data:
                     tasks.append(data["task"])
                else:
                     # 딕셔너리인 경우 키들을 작업으로 사용 (Legacy)
                     tasks.extend(data.keys())
            elif isinstance(data, list):
                # 리스트인 경우 모든 요소를 작업으로 사용
                tasks.extend(data)
            elif isinstance(data, str):
                # 문자열인 경우 그대로 작업으로 추가
                tasks.append(data)
    return tasks


def format_actions_section(actions: List[str]) -> str:
    """
    액션 목록을 프롬프트에 포함될 형식으로 포맷팅합니다.
    
    매개변수:
        actions: 사용 가능한 액션 목록
    
    반환값:
        str: 포맷팅된 액션 섹션 문자열
    """
    # LLM에게 전달할 액션 사용 지침 구성
    instructions = [
        "Use ONLY the following high-level APIs (AI2-THOR compatible):",  # 허용된 API만 사용하라는 지시
        *actions,  # 액션 목록을 펼쳐서 추가
        "",
        "Your program must be executable pseudo-code:",  # 실행 가능한 의사코드 형식 요구
        "  • Stage 1 (Program Skeleton): declare `def <task_name>():` and keep code block indentation.",  # 1단계: 함수 선언 및 들여쓰기
        "  • Stage 2 (Structured Comments): precede each block with a natural-language `# comment` that describes",  # 2단계: 구조화된 주석
        "    the intent of that block (e.g., `# Step 2: Open the fridge`).",  # 각 블록의 의도를 설명하는 주석 예시
        "  • Stage 3 (Assertions + Recovery): before every environment action, assert the relevant precondition.",  # 3단계: 어설션 및 복구 단계
        "    Use the syntax:",  # 사용할 문법
        "        assert('Fridge' is 'opened')",  # 어설션 예시: 냉장고가 열려있는지 확인
        "            else: OpenObject('Fridge')",  # 조건이 거짓이면 냉장고 열기
        "    or distance checks such as:",  # 또는 거리 확인 예시
        "        assert('close' to 'Apple')",  # 사과에 가까운지 확인
        "            else: GoToObject('Apple')",  # 조건이 거짓이면 사과로 이동
        "    CRITICAL: Before EVERY Pickup action, you MUST assert that hands are empty:",  # 중요: 모든 Pickup 액션 전에 손이 비어있는지 확인
        "        assert('hands' is 'empty')",  # 손이 비어있는지 확인
        "            else: GoToObject('CounterTop')",  # 손에 뭔가 들고 있으면 카운터로 이동
        "            else: PutObject(<current_object_in_hands>, 'CounterTop')",  # 들고 있는 객체를 카운터에 놓기
        "        where <current_object_in_hands> is the object currently in hands. Replace it with the actual object name.",  # 현재 손에 든 객체 이름으로 대체
        "        Example: If holding 'Knife' and picking up 'Apple', use: PutObject('Knife', 'CounterTop')",  # 예시: 칼을 들고 있을 때 사과를 집으려면
        "        If CounterTop is not nearby the target object, use another nearby receptacle like 'Table', 'Shelf', or 'Floor'.",  # 카운터가 가깝지 않으면 다른 수용체 사용
        "        First GoTo the receptacle, then Put the held object there.",  # 먼저 수용체로 이동한 후 객체를 놓기
        "    Recovery steps MUST be concrete actions from the allowed list.",  # 복구 단계는 허용된 액션 목록에서 구체적인 액션이어야 함
        "",
        
    ]
    return "\n".join(instructions)  # 줄바꿈으로 연결하여 반환


def build_prompt(objects: List[str], actions: List[str], example_programs: Dict[str, str], max_examples: int) -> str:
    """
    ProgPrompt 규칙을 인코딩한 시스템 프롬프트를 구성합니다.
    
    매개변수:
        objects: 사용 가능한 객체 목록
        actions: 사용 가능한 액션 목록
        example_programs: 예제 프로그램 딕셔너리 (작업명: 프로그램 코드)
        max_examples: 프롬프트에 포함할 최대 예제 개수
    
    반환값:
        str: 완성된 시스템 프롬프트 문자열
    """
    # 객체 목록을 따옴표로 감싸서 쉼표로 구분된 문자열로 변환
    obj_listing = ", ".join(f"\"{obj}\"" for obj in objects)
    
    # 예제 프로그램 중 최대 개수만큼 선택
    selected_examples = list(example_programs.items())[:max_examples]

    # 예제 블록 구성
    example_block_lines = []
    for task_name, code in selected_examples:
        example_block_lines.append(f"# Example: {task_name}")  # 예제 제목 추가
        example_block_lines.append(code)  # 예제 코드 추가
        example_block_lines.append("")  # 빈 줄 추가 (가독성)
    example_block_section = "\n".join(example_block_lines)  # 줄바꿈으로 연결

    prompt = f"""
                You are ProgPrompt, an AI2-THOR task planner operating in scene FloorPlan1.
                The user will describe a household task. You must return a PYTHON-LIKE PROGRAM that:
                • Uses ONLY the whitelisted actions listed below.
                • Includes structured natural-language comments before each action block (Stage 2).
                • Inserts assertion + recovery pairs before every interaction to guarantee preconditions (Stage 3).
                • Keeps actions grouped logically so the plan can be executed sequentially without missing steps.

                Forbidden outputs:
                ✗ Free-form natural language
                ✗ JSON, bullet lists, or code without asserts/comments
                ✗ Actions outside the whitelist

                Environment knowledge:
                • Simulator: AI2-THOR
                • Scene: FloorPlan1
                • Available objects: {obj_listing}
                • IMPORTANT: The agent can hold only ONE object at a time. Before picking up a new object, place the currently held object on a nearby receptacle.

                {format_actions_section(actions)}

                # Program template you MUST follow
                def task_name():
                \t# Step description
                \tassert('<condition>')
                \t\telse: <recovery action>
                \t<ActionCall('Object')>

                # Additional rules
                • If a precondition already holds, the `else:` recovery must be skipped.
                • CRITICAL: The agent can hold only ONE object at a time. 
                • CRITICAL: Before EVERY Pickup action, you MUST add an assert to check if hands are empty:
                    assert('hands' is 'empty')
                        else: GoToObject('CounterTop')
                        else: PutObject(<current_object_in_hands>, 'CounterTop')
                    where <current_object_in_hands> is the object currently in hands. Replace it with the actual object name.
                    If CounterTop is not nearby the target object, use another nearby receptacle like 'Table', 'Shelf', or 'Floor'.
                    First GoTo the receptacle, then Put the held object there.
                    This ensures that if the agent is already holding an object, it will place it on a nearby surface before picking up a new one.
                • When storing object A inside B, ensure B is open (assert + recovery).
                • When slicing or using tools, assert the correct tool is in hand.
                • After finishing tool use, drop or place the tool somewhere sensible.
                • Keep the number of steps minimal but never skip mandatory transitions.
                • If you want to slice the object, you don't need to pickup the object. Just bring Knife to the object and slice it.
                - Agent는 하나의 객체만 hold할 수 있음. 만약 2개 이상의 객체를 pickup해야 한다면, 먼저 하나를 pickup하고 해야 하는 액션을 수행 후, 다음 객체를 pickup 해야 함.
                - 객체를 pickup하기 전에는 무조건 객체가 pickupable한지, parentReceptacle 안에 존재하는 객체인지 확인 후, IN edge가 존재하면 openobject를 먼저 수행해야 함.
                

                Below are exemplar programs showing the required layout:
{textwrap.indent(example_block_section, "")}
"""
    # 들여쓰기 제거 및 앞뒤 공백 제거 후 반환
    return textwrap.dedent(prompt).strip()


def generate_program(
    client: OpenAI,
    model: str,
    base_prompt: str,
    task: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """
    Ollama (OpenAI 호환)를 쿼리하여 단일 작업에 대한 프로그램을 생성합니다.
    
    매개변수:
        client: OpenAI 호환 API 클라이언트 인스턴스
        model: 사용할 모델 이름 (예: "llama3")
        base_prompt: 시스템 프롬프트 (ProgPrompt 규칙 포함)
        task: 수행할 작업 설명
        temperature: 샘플링 온도 (0.0 = 결정적, 높을수록 다양함)
        max_tokens: 생성할 최대 토큰 수
    
    반환값:
        str: 생성된 Python 스타일 프로그램 코드
    """
    # LLM에 전달할 메시지 구성
    messages = [
        {"role": "system", "content": base_prompt},  # 시스템 프롬프트 (규칙 및 예제)
        {
            "role": "user",  # 사용자 메시지
            "content": (
                "Task instruction:\n"  # 작업 지시사항 헤더
                f"{task}\n\n"  # 실제 작업 설명
                "Produce ONLY the Python-like program that satisfies the rules."  # 규칙을 만족하는 Python 스타일 프로그램만 생성하라는 지시
            ),
        },
    ]
    
    # LLM API 호출하여 프로그램 생성
    response = client.chat.completions.create(
        model=model,  # 사용할 모델
        messages=messages,  # 메시지 목록
        temperature=temperature,  # 샘플링 온도
        max_tokens=max_tokens,  # 최대 토큰 수
    )
    
    # 응답에서 생성된 프로그램 코드 추출 (앞뒤 공백 제거)
    return response.choices[0].message.content.strip()


# ============================================================================
# Scene Graph 기반 물리적 검증 함수들
# ============================================================================

def parse_program_to_actions(program_code: str) -> List[Dict[str, Any]]:
    """
    프로그램 코드를 파싱하여 액션 리스트로 변환
    
    Args:
        program_code: 프로그램 코드 (def 형태)
        
    Returns:
        액션 리스트 [{"type": str, "args": dict, "line": str}, ...]
    """
    lines = program_code.split("\n")
    plan = []
    current_task = None
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("Task:"):
            current_task = line.replace("Task:", "").strip()
            continue
        
        if line.startswith("def "):
            current_task = line.split("(")[0].replace("def ", "").strip()
            continue
        
        # assert, else:, 주석 제거
        if line.startswith("assert") or line.startswith("else:") or line.startswith("#"):
            continue
        
        # 액션 파싱
        match = re.match(r'(\w+)\(([^)]*)\)', line)
        if not match:
            continue
        
        action = match.group(1)
        params = match.group(2)
        
        if not params:
            continue
        
        params = [p.strip().strip("'\"") for p in params.split(",")]
        
        # 액션 타입 정규화
        action_type = action
        if action == "GoTo":
            action_type = "GoToObject"
        elif action == "Pickup":
            action_type = "PickupObject"
        elif action == "Put":
            action_type = "PutObject"
        elif action == "Open":
            action_type = "OpenObject"
        elif action == "Close":
            action_type = "CloseObject"
        elif action == "ToggleOn":
            action_type = "ToggleObjectOn"
        elif action == "ToggleOff":
            action_type = "ToggleObjectOff"
        elif action == "Slice":
            action_type = "SliceObject"
        elif action == "Break":
            action_type = "BreakObject"

        
        if len(params) == 1:
            plan.append({
                "type": action_type,
                "args": {"o": params[0]},
                "line": line,
                "is_original": True,  # 원본 액션 표시
                "is_recovery": False
            })
        elif len(params) == 2:
            plan.append({
                "type": action_type,
                "args": {"o": params[0], "r": params[1]},
                "line": line,
                "is_original": True,  # 원본 액션 표시
                "is_recovery": False
            })
    
    return plan


def load_scene_graph(scene_graph_path: str) -> Dict[str, Any]:
    """Scene Graph JSON 파일 로드"""
    try:
        with open(scene_graph_path, "r", encoding="utf-8") as f:
            scene_graph = json.load(f)
        logger.info(f"Scene Graph 로드 완료: {scene_graph_path}")
        return scene_graph
    except Exception as e:
        logger.error(f"Scene Graph 로드 실패: {e}")
        return {"nodes": {"agent": {}, "objects": []}, "edges": []}


def get_relevant_scene_context(
    scene_graph: Dict[str, Any],
    action_type: str,
    object_name: Optional[str] = None,
    receptacle_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Scene Graph에서 관련 노드와 엣지 정보 추출
    
    Args:
        scene_graph: Scene Graph 딕셔너리
        action_type: 액션 타입
        object_name: 타겟 객체 이름
        receptacle_name: 수용체 이름
        
    Returns:
        관련 노드와 엣지 정보
    """
    nodes = scene_graph.get("nodes", {})
    edges = scene_graph.get("edges", [])
    agent_node = nodes.get("agent", {})
    object_nodes = nodes.get("objects", [])
    
    # 관련 객체 노드 찾기
    relevant_objects = []
    target_object_node = None
    receptacle_object_node = None
    
    # 정확한 매칭과 부분 매칭을 분리하여 수집
    exact_matches = []
    partial_matches = []
    
    if object_name:
        object_name_lower = object_name.lower()
        
        for obj_node in object_nodes:
            obj_type = obj_node.get("objectType", "")
            obj_id = obj_node.get("nodeId", "")
            obj_type_lower = obj_type.lower()
            
            # nodeId 형식인지 확인 (예: "Drawer|-01.56|+00.84|-00.20")
            if "|" in object_name and len(object_name.split("|")) >= 4:
                # nodeId 형식이면 정확히 일치하는지 확인
                if obj_id == object_name:
                    exact_matches.append(obj_node)
            else:
                # "Knife"를 찾을 때 "ButterKnife"는 무조건 제외
                if object_name_lower == "knife":
                    if "butter" in obj_type_lower or "butter" in obj_id.lower():
                        continue
                
                # 정확한 매칭 우선 (예: "Knife"는 "Knife"와만 매칭)
                if obj_type_lower == object_name_lower:
                    exact_matches.append(obj_node)
                # 정확한 매칭이 없으면 부분 매칭 시도
                elif object_name_lower in obj_type_lower:
                    # 추가 확인: "knife"를 찾을 때 "butterknife"는 제외 (이중 체크)
                    if object_name_lower == "knife" and ("butter" in obj_type_lower or "butter" in obj_id.lower()):
                        continue
                    partial_matches.append(obj_node)
        
        # "Knife"를 찾을 때는 정확한 매칭만 사용 (부분 매칭 사용 안 함)
        if object_name_lower == "knife":
            if exact_matches:
                target_object_node = exact_matches[0]  # 첫 번째 정확한 매칭 사용
                relevant_objects.extend(exact_matches)
            # 정확한 매칭이 없으면 None (ButterKnife는 선택 안 함)
        else:
            # 정확한 매칭이 있으면 그것만 사용, 없으면 부분 매칭 사용
            if exact_matches:
                target_object_node = exact_matches[0]  # 첫 번째 정확한 매칭 사용
                relevant_objects.extend(exact_matches)
            elif partial_matches:
                target_object_node = partial_matches[0]  # 첫 번째 부분 매칭 사용
                relevant_objects.extend(partial_matches)
    
    # 수용체 찾기 (별도 처리)
    if receptacle_name:
        receptacle_exact_matches = []
        receptacle_partial_matches = []
        receptacle_name_lower = receptacle_name.lower()
        
        for obj_node in object_nodes:
            obj_type = obj_node.get("objectType", "")
            obj_id = obj_node.get("nodeId", "")
            obj_type_lower = obj_type.lower()
            
            # nodeId 형식인지 확인
            if "|" in receptacle_name and len(receptacle_name.split("|")) >= 4:
                # nodeId 형식이면 정확히 일치하는지 확인
                if obj_id == receptacle_name:
                    receptacle_exact_matches.append(obj_node)
            else:
                # 일반 이름 형식이면 타입으로 매칭
                if obj_type_lower == receptacle_name_lower:
                    receptacle_exact_matches.append(obj_node)
                elif receptacle_name_lower in obj_type_lower:
                    receptacle_partial_matches.append(obj_node)
        
        # 정확한 매칭이 있으면 그것만 사용, 없으면 부분 매칭 사용
        if receptacle_exact_matches:
            receptacle_object_node = receptacle_exact_matches[0]
            relevant_objects.extend(receptacle_exact_matches)
        elif receptacle_partial_matches:
            receptacle_object_node = receptacle_partial_matches[0]
            relevant_objects.extend(receptacle_partial_matches)
    
    # 관련 엣지 찾기
    relevant_edges = []
    for edge in edges:
        source = edge.get("source", "")
        target = edge.get("target", "")
        edge_type = edge.get("edgeType", "")
        
        # Agent와 관련된 엣지
        if source == "agent_0" or target == "agent_0":
            if object_name and (object_name.lower() in target.lower() or object_name.lower() in source.lower()):
                relevant_edges.append(edge)
        
        # 타겟 객체와 관련된 엣지
        if target_object_node:
            obj_id = target_object_node.get("nodeId", "")
            if obj_id in [source, target]:
                relevant_edges.append(edge)
        
        # 수용체와 관련된 엣지
        if receptacle_object_node:
            recp_id = receptacle_object_node.get("nodeId", "")
            if recp_id in [source, target]:
                relevant_edges.append(edge)
    
    return {
        "agent": agent_node,
        "targetObject": target_object_node,
        "receptacleObject": receptacle_object_node,
        "relevantObjects": relevant_objects,
        "relevantEdges": relevant_edges
    }


def find_closest_reachable_position(
    controller: Optional[Any],
    target_pos: Dict[str, float],
    agent_pos: Optional[Dict[str, float]] = None,
    return_closest_only: bool = False
) -> Tuple[Optional[Dict[str, float]], float, Optional[Dict[str, float]]]:
    """
    NavMesh에서 목표 위치까지 가장 가까운 이동 가능 위치 찾기
    목표 객체와 정면으로 마주보는 위치를 우선적으로 선택
    
    Args:
        controller: AI2-THOR Controller (None이면 None 반환)
        target_pos: 목표 위치 {"x": float, "y": float, "z": float}
        agent_pos: 현재 Agent 위치 (None이면 controller에서 가져옴)
        return_closest_only: True이면 거리만 고려한 가장 가까운 위치도 반환
        
    Returns:
        (최적 이동 가능 위치, 거리, 거리만 고려한 가장 가까운 위치) 또는 (None, float('inf'), None)
        return_closest_only가 False이면 세 번째 값은 None
    """
    if controller is None:
        return None, float('inf'), None
    
    try:
        # GetReachablePositions로 이동 가능한 위치들 가져오기
        event = controller.step(action="GetReachablePositions")
        reachable_positions = event.metadata.get("actionReturn", [])
        
        if not reachable_positions:
            return None, float('inf'), None
        
        # Agent 위치 가져오기
        if agent_pos is None:
            agent_metadata = event.metadata.get("agent", {})
            agent_pos = agent_metadata.get("position", {})
            if not agent_pos:
                # agent_pos가 없으면 기본값 사용
                agent_pos = {"x": 0, "y": 0, "z": 0}
        
        target_x = target_pos.get("x", 0)
        target_z = target_pos.get("z", 0)
        agent_x = agent_pos.get("x", 0)
        agent_z = agent_pos.get("z", 0)
        
        # 목표 객체를 향한 방향 벡터 (Agent 기준)
        target_direction_x = target_x - agent_x
        target_direction_z = target_z - agent_z
        target_direction_norm = math.sqrt(target_direction_x**2 + target_direction_z**2)
        
        if target_direction_norm < 0.01:
            # Agent가 이미 목표 위치에 매우 가까움
            target_direction_x = 1.0
            target_direction_z = 0.0
            target_direction_norm = 1.0
        
        # 정규화된 방향 벡터
        target_dir_x = target_direction_x / target_direction_norm
        target_dir_z = target_direction_z / target_direction_norm
        
        best_pos = None
        best_score = float('inf')
        closest_by_distance = None  # 거리만 고려한 가장 가까운 위치
        closest_distance = float('inf')
        
        # 각 도달 가능한 위치에 대해 점수 계산 (거리 + 각도 고려)
        for pos in reachable_positions:
            pos_x = pos.get("x", 0)
            pos_z = pos.get("z", 0)
            
            # 거리 계산 (x, z 평면)
            distance = math.sqrt((pos_x - target_x)**2 + (pos_z - target_z)**2)
            
            # 거리만 고려한 가장 가까운 위치 추적
            if distance < closest_distance:
                closest_distance = distance
                closest_by_distance = {
                    "x": pos_x,
                    "y": pos.get("y", 0),
                    "z": pos_z
                }
            
            # 위치에서 목표 객체를 향한 방향 벡터
            pos_to_target_x = target_x - pos_x
            pos_to_target_z = target_z - pos_z
            pos_to_target_norm = math.sqrt(pos_to_target_x**2 + pos_to_target_z**2)
            
            if pos_to_target_norm < 0.01:
                # 위치가 목표와 거의 같음
                angle_score = 0.0
            else:
                # 정규화된 방향 벡터
                pos_dir_x = pos_to_target_x / pos_to_target_norm
                pos_dir_z = pos_to_target_z / pos_to_target_norm
                
                # 내적을 사용하여 각도 계산 (1.0 = 같은 방향, -1.0 = 반대 방향)
                dot_product = target_dir_x * pos_dir_x + target_dir_z * pos_dir_z
                # 각도 차이 (0~180도)
                angle_diff = math.acos(max(-1.0, min(1.0, dot_product))) * 180.0 / math.pi
                # 각도 점수: 0도에 가까울수록 낮은 점수 (우선순위 높음)
                angle_score = angle_diff
            # 종합 점수: 거리 + 각도 가중치 (거리 1m = 각도 10도와 동일한 가중치)
            # 거리가 가까울수록, 각도가 작을수록(정면) 낮은 점수 = 우선순위 높음
            score = distance + (angle_score / 10.0)
            
            if score < best_score:
                best_score = score
                best_pos = {
                    "x": pos_x,
                    "y": pos.get("y", 0),
                    "z": pos_z
                }
        
        if best_pos:
            final_distance = math.sqrt((best_pos["x"] - target_x)**2 + (best_pos["z"] - target_z)**2)
            if return_closest_only:
                return best_pos, final_distance, closest_by_distance
            else:
                return best_pos, final_distance, None
        else:
            return None, float('inf'), None
    except Exception as e:
        logger.warning(f"NavMesh에서 이동 가능 위치 찾기 실패: {e}")
        return None, float('inf'), None


def verify_guard_with_scene_graph(
    guard_name: str,
    scene_context: Dict[str, Any],
    action_type: str,
    object_name: Optional[str],
    receptacle_name: Optional[str],
    controller: Optional[Any] = None,
    scene_graph: Optional[Dict[str, Any]] = None,
    agent_position: Optional[Dict[str, float]] = None
) -> Tuple[bool, str]:
    """
    Scene Graph를 직접 확인하여 개별 가드 검증
    
    Args:
        guard_name: 가드 이름 (예: "VISIBLE(agent, object)", "HOLDS(agent, *)")
        scene_context: Scene Graph에서 추출한 관련 정보
        action_type: 액션 타입
        object_name: 타겟 객체 이름
        receptacle_name: 수용체 이름
        controller: AI2-THOR Controller (NavMesh 검증용, 선택사항)
        
    Returns:
        (검증 통과 여부, 이유)
    """
    agent_node = scene_context.get("agent", {})
    target_obj = scene_context.get("targetObject")
    receptacle_obj = scene_context.get("receptacleObject")
    relevant_edges = scene_context.get("relevantEdges", [])
    
    guard_upper = guard_name.upper()
    
    # EXISTS(object) 검증
    if "EXISTS" in guard_upper:
        if target_obj is None:
            return False, f"객체 '{object_name}'가 Scene Graph에 존재하지 않음"
        return True, f"객체 '{object_name}' 존재 확인"
    
    # REACHABLE(agent, object) 검증 - armBase 좌표계 기준 (손 위치 기준)
    if "REACHABLE" in guard_upper:
        # PutObject의 경우 receptacle을 타겟으로 사용
        if "RECEPTACLE" in guard_upper or action_type == "PutObject":
            check_obj = receptacle_obj if receptacle_obj else target_obj
            obj_name_for_log = receptacle_name if receptacle_name else object_name
        else:
            check_obj = target_obj
            obj_name_for_log = object_name
        
        if check_obj is None:
            return False, f"타겟 객체가 없음 ({'receptacle' if ('RECEPTACLE' in guard_upper or action_type == 'PutObject') else 'object'})"
        
        obj_pos = check_obj.get("position", {})
        if not obj_pos:
            return False, "객체 위치 정보가 없음"
        
        # Agent 위치: 파라미터로 받은 위치를 우선 사용, 없으면 agent_node에서 가져오기
        if agent_position is not None:
            agent_pos = agent_position
        else:
            agent_pos = agent_node.get("position", {})
        
        agent_rot = agent_node.get("rotation", {})
        
        if not agent_pos:
            return False, "Agent 위치 정보가 없음"
        
        # Hand 위치: Agent의 절대 좌표를 Hand 좌표로 사용 (절대 좌표로 직접 비교)
        agent_x = agent_pos.get("x", 0) # - 0.3289999
        agent_y = agent_pos.get("y", 0) # + 0.2250000
        agent_z = agent_pos.get("z", 0) # - 0.5699999
        
        # 객체의 절대 좌표
        obj_x = obj_pos.get("x", 0)
        obj_y = obj_pos.get("y", 0)
        obj_z = obj_pos.get("z", 0)
        
        # 손 위치를 기준으로 한 도달 가능한 범위 (절대 좌표 기준)
        # use_arm_and_armbase.py에서 테스트한 범위 기준:
        # x 범위: -1 ~ 1 (좌우)
        # y 범위: -0.35 ~ 1 (상하)
        # z 범위: 0 ~ 1 (앞뒤)
        # 손 위치에서 ± 범위로 계산
        x_range_min = -1 -0.5  #
        x_range_max = +1 +0.5 #
        y_range_min = -0.901     # 아래로 -0.35
        y_range_max = 0.6  # 위로 1.0
        z_range_min = -1 - 0.5 # 뒤로 0 (뒤) - 기본 1.5에 handSpereRadius=0.2로 했을 때
        z_range_max = +1 + 0.5 # 앞으로 1.0 (앞) - 기본 0.5에 handSpereRadius=0.2로 했을 때
        
        # 손 위치 기준으로 범위 계산 (절대 좌표)
        x_min = agent_x + x_range_min
        x_max = agent_x + x_range_max
        y_min = agent_y + y_range_min
        y_max = agent_y + y_range_max
        z_min = agent_z + z_range_min
        z_max = agent_z + z_range_max
        
        # 목표 객체가 손 위치 기준 범위 내에 있는지 확인 (절대 좌표로 비교)
        in_range = (x_min <= obj_x <= x_max and 
                   y_min <= obj_y <= y_max and 
                   z_min <= obj_z <= z_max)
        
        # 손에서 객체까지의 거리 계산 (절대 좌표)
        dx = obj_x - agent_x
        dy = obj_y - agent_y
        dz = obj_z - agent_z
        distance_3d = math.sqrt(dx**2 + dy**2 + dz**2)
        
        if in_range:
            return True, f"\n Agent 위치 기준 범위 내 (agent 위치: x={agent_x:.3f}, y={agent_y:.3f}, z={agent_z:.3f}, 객체 위치: x={obj_x:.3f}, y={obj_y:.3f}, z={obj_z:.3f}, 거리={distance_3d:.3f}m)"
        else:
            # 범위를 벗어난 경우
            out_of_range_axis = []
            if obj_x < x_min or obj_x > x_max:
                out_of_range_axis.append(f"x={obj_x:.3f} (범위: {x_min:.3f}~{x_max:.3f})")
            if obj_y < y_min or obj_y > y_max:
                out_of_range_axis.append(f"y={obj_y:.3f} (범위: {y_min:.3f}~{y_max:.3f})")
            if obj_z < z_min or obj_z > z_max:
                out_of_range_axis.append(f"z={obj_z:.3f} (범위: {z_min:.3f}~{z_max:.3f})")
            
            return False, f"\n Agent 위치 기준 범위 밖 (agent 위치: x={agent_x:.3f}, y={agent_y:.3f}, z={agent_z:.3f}, 객체 위치: x={obj_x:.3f}, y={obj_y:.3f}, z={obj_z:.3f}, 거리={distance_3d:.3f}m, 벗어난 축: {', '.join(out_of_range_axis)})"
    
    # Proximity(agent, object) 검증 - agent와 목표 객체 간 거리가 2m 이내인지 확인
    if "PROXIMITY" in guard_upper:
        # PutObject의 경우 receptacle을 타겟으로 사용
        if action_type == "PutObject":
            check_obj = receptacle_obj if receptacle_obj else target_obj
            obj_name_for_log = receptacle_name if receptacle_name else object_name
        else:
            check_obj = target_obj
            obj_name_for_log = object_name
        
        if check_obj is None:
            return False, f"타겟 객체가 없음 ({'receptacle' if action_type == 'PutObject' else 'object'})"
        
        obj_pos = check_obj.get("position", {})
        if not obj_pos:
            return False, "객체 위치 정보가 없음"
        
        # Agent 위치: 파라미터로 받은 위치를 우선 사용, 없으면 agent_node에서 가져오기
        if agent_position is not None:
            agent_pos = agent_position
        else:
            agent_pos = agent_node.get("position", {})
        
        if not agent_pos:
            return False, "Agent 위치 정보가 없음"
        
        # Agent와 객체 간 거리 계산 (3D 유클리드 거리)
        agent_x = agent_pos.get("x", 0)
        agent_y = agent_pos.get("y", 0)
        agent_z = agent_pos.get("z", 0)
        
        obj_x = obj_pos.get("x", 0)
        obj_y = obj_pos.get("y", 0)
        obj_z = obj_pos.get("z", 0)
        
        dx = obj_x - agent_x
        dy = obj_y - agent_y
        dz = obj_z - agent_z
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # 2m 이내이면 통과
        proximity_threshold = 1.5
        if distance <= proximity_threshold:
            return True, f"Agent와 객체 간 거리: {distance:.3f}m <= {proximity_threshold}m (agent 위치: ({agent_x:.3f}, {agent_y:.3f}, {agent_z:.3f}), 객체 위치: ({obj_x:.3f}, {obj_y:.3f}, {obj_z:.3f}))"
        else:
            return False, f"Agent와 객체 간 거리: {distance:.3f}m > {proximity_threshold}m (agent 위치: ({agent_x:.3f}, {agent_y:.3f}, {agent_z:.3f}), 객체 위치: ({obj_x:.3f}, {obj_y:.3f}, {obj_z:.3f}))"
    
    # NAVIGABLE(agent, object) 검증
    if "NAVIGABLE" in guard_upper:
        if target_obj is None:
            return False, "타겟 객체가 없음"
        
        obj_pos = target_obj.get("position", {})
        if not obj_pos:
            return False, "객체 위치 정보가 없음"
        
        # Agent 위치 가져오기 (정면으로 마주보는 위치 선택을 위해)
        agent_node = scene_context.get("agent", {}) if scene_context else {}
        agent_pos_for_nav = agent_position if agent_position else agent_node.get("position", {})
        
        # NavMesh를 사용하여 목표 객체까지 거리만 고려한 가장 가까운 이동 가능 위치 찾기
        closest_pos, distance, closest_by_distance = find_closest_reachable_position(
            controller, obj_pos, agent_pos_for_nav, return_closest_only=True
        )
        
        if closest_pos is None:
            return False, "NavMesh 정보를 가져올 수 없음 (Controller 필요)"
        
        # 거리만 고려한 가장 가까운 위치를 사용 (closest_by_distance가 있으면 사용)
        if closest_by_distance:
            closest_pos = closest_by_distance
            # 거리 재계산
            obj_x = obj_pos.get("x", 0)
            obj_z = obj_pos.get("z", 0)
            closest_x = closest_pos.get("x", 0)
            closest_z = closest_pos.get("z", 0)
            distance = math.sqrt((closest_x - obj_x)**2 + (closest_z - obj_z)**2)
        
        # 목표 객체 위치와 가장 가까운 이동 가능 위치의 실제 거리 계산
        obj_x = obj_pos.get("x", 0)
        obj_y = obj_pos.get("y", 0)
        obj_z = obj_pos.get("z", 0)
        closest_x = closest_pos.get("x", 0)
        closest_y = closest_pos.get("y", 0)
        closest_z = closest_pos.get("z", 0)
        
        # 실제 3D 거리 계산 (목표 객체 위치와 이동 가능 위치 사이)
        actual_distance_3d = math.sqrt(
            (closest_x - obj_x)**2 + 
            (closest_y - obj_y)**2 + 
            (closest_z - obj_z)**2
        )
        
        # 가장 가까운 이동 가능 위치 좌표 정보
        closest_pos_str = f"({closest_x:.3f}, {closest_y:.3f}, {closest_z:.3f})"
        obj_pos_str = f"({obj_x:.3f}, {obj_y:.3f}, {obj_z:.3f})"
        
        # 거리가 1.3m 이내이면 통과
        navigable_threshold = 1.3
        if distance <= navigable_threshold:
            return True, f"목표 객체 위치: {obj_pos_str}, 가장 가까운 이동 가능 위치: {closest_pos_str} (NavMesh 거리: {distance:.2f}m, 실제 3D 거리: {actual_distance_3d:.2f}m) <= {navigable_threshold}m"
        else:
            return False, f"목표 객체 위치: {obj_pos_str}, 가장 가까운 이동 가능 위치: {closest_pos_str} (NavMesh 거리: {distance:.2f}m, 실제 3D 거리: {actual_distance_3d:.2f}m) > {navigable_threshold}m"
    
    # HOLDS(agent, 'Knife') 검증 (특정 객체를 들고 있는지 확인) - HOLDS(agent, object)보다 먼저 확인
    if "HOLDS" in guard_upper and "'" in guard_name:
        # guard_name에서 객체 이름 추출 (예: "HOLDS(agent, 'Knife')")
        import re
        match = re.search(r"'([^']+)'", guard_name)
        if match:
            required_object_name = match.group(1)
            
            # 제외할 객체 타입 리스트 (예: 'Knife'를 찾을 때 'ButterKnife'는 제외)
            exclude_types = []
            if required_object_name.lower() == "knife":
                exclude_types = ["butterknife"]
            
            # scene_graph를 우선적으로 사용 (업데이트된 정보 반영)
            if scene_graph:
                # Agent 노드에서 직접 확인
                agent_node_updated = scene_graph.get("nodes", {}).get("agent", {})
                if agent_node_updated.get("isHolding", False):
                    held_object_id = agent_node_updated.get("heldObjectId")
                    if held_object_id:
                        # heldObjectId에서 객체 타입 추출
                        object_nodes = scene_graph.get("nodes", {}).get("objects", [])
                        for obj_node in object_nodes:
                            if obj_node.get("nodeId") == held_object_id:
                                obj_type = obj_node.get("objectType", "")
                                obj_type_lower = obj_type.lower()
                                required_name_lower = required_object_name.lower()
                                
                                # 제외할 타입인지 확인
                                if any(exclude.lower() in obj_type_lower for exclude in exclude_types):
                                    break
                                
                                # 정확한 타입 매칭 또는 정확한 타입으로 시작하는지 확인
                                if obj_type_lower == required_name_lower or obj_type_lower.startswith(required_name_lower + "|"):
                                    return True, f"Agent가 '{required_object_name}'를 들고 있음 (업데이트된 Scene Graph)"
                                break
                
                # HOLDS 엣지 확인
                edges = scene_graph.get("edges", [])
                for edge in edges:
                    if edge.get("edgeType") == "HOLDS":
                        held_obj_id = edge.get("target")
                        if held_obj_id:
                            object_nodes = scene_graph.get("nodes", {}).get("objects", [])
                            for obj_node in object_nodes:
                                if obj_node.get("nodeId") == held_obj_id:
                                    obj_type = obj_node.get("objectType", "")
                                    obj_type_lower = obj_type.lower()
                                    required_name_lower = required_object_name.lower()
                                    
                                    # 제외할 타입인지 확인
                                    if any(exclude.lower() in obj_type_lower for exclude in exclude_types):
                                        break
                                    
                                    # 정확한 타입 매칭 또는 정확한 타입으로 시작하는지 확인
                                    if obj_type_lower == required_name_lower or obj_type_lower.startswith(required_name_lower + "|"):
                                        return True, f"Agent가 '{required_object_name}'를 들고 있음 (HOLDS 엣지)"
                                    break
            
            # scene_graph에서 확인 실패 시 scene_context 사용 (fallback)
            holds_edges = [e for e in relevant_edges if e.get("edgeType") == "HOLDS"]
            if holds_edges:
                held_obj_id = holds_edges[0].get("target")
                # 실제로 들고 있는 객체의 타입 확인
                held_obj_type = None
                if scene_graph:
                    object_nodes = scene_graph.get("nodes", {}).get("objects", [])
                    for obj_node in object_nodes:
                        if obj_node.get("nodeId") == held_obj_id:
                            held_obj_type = obj_node.get("objectType", "")
                            break
                
                if held_obj_type:
                    return False, f"Agent가 '{required_object_name}'를 들고 있지 않음 (현재 '{held_obj_type}'를 들고 있음)"
                else:
                    return False, f"Agent가 '{required_object_name}'를 들고 있지 않음 (다른 객체를 들고 있음)"
            
            # Agent 노드의 isHolding 확인
            if agent_node.get("isHolding", False):
                held_object_id = agent_node.get("heldObjectId")
                if held_object_id:
                    # 실제로 들고 있는 객체의 타입 확인
                    held_obj_type = None
                    if scene_graph:
                        object_nodes = scene_graph.get("nodes", {}).get("objects", [])
                        for obj_node in object_nodes:
                            if obj_node.get("nodeId") == held_object_id:
                                held_obj_type = obj_node.get("objectType", "")
                                break
                    
                    if held_obj_type:
                        return False, f"Agent가 '{required_object_name}'를 들고 있지 않음 (현재 '{held_obj_type}'를 들고 있음)"
                    else:
                        return False, f"Agent가 '{required_object_name}'를 들고 있지 않음 (다른 객체를 들고 있음)"
            
            return False, f"Agent가 '{required_object_name}'를 들고 있지 않음"
    
    # HOLDS(agent, object) 검증
    if "HOLDS" in guard_upper and "¬" not in guard_name:
        if target_obj is None:
            return False, "타겟 객체가 없음"
        
        target_obj_id = target_obj.get("nodeId")
        
        # scene_graph를 우선적으로 사용 (업데이트된 정보 반영)
        if scene_graph:
            # Agent 노드에서 직접 확인
            agent_node_updated = scene_graph.get("nodes", {}).get("agent", {})
            if agent_node_updated.get("isHolding", False) and agent_node_updated.get("heldObjectId") == target_obj_id:
                return True, f"Agent가 '{object_name}'를 들고 있음 (업데이트된 Scene Graph)"
            
            # HOLDS 엣지 확인
            edges = scene_graph.get("edges", [])
            for edge in edges:
                if edge.get("edgeType") == "HOLDS" and edge.get("target") == target_obj_id:
                    return True, f"HOLDS 엣지 존재 (Agent가 '{object_name}'를 들고 있음)"
        
        # scene_graph에서 확인 실패 시 scene_context 사용 (fallback)
        # HOLDS 엣지 확인
        holds_edge = None
        for edge in relevant_edges:
            if edge.get("edgeType") == "HOLDS" and edge.get("target") == target_obj_id:
                holds_edge = edge
                break
        
        if holds_edge:
            return True, f"HOLDS 엣지 존재 (Agent가 '{object_name}'를 들고 있음)"
        
        # Agent 노드의 isHolding 확인
        if agent_node.get("isHolding", False) and agent_node.get("heldObjectId") == target_obj_id:
            return True, f"Agent가 '{object_name}'를 들고 있음"
        
        return False, f"Agent가 '{object_name}'를 들고 있지 않음"
    
    # ¬HOLDS(agent, *) 검증 (손이 비어있어야 함)
    if "HOLDS" in guard_upper and "¬" in guard_name:
        # scene_graph를 우선적으로 사용 (업데이트된 정보 반영)
        if scene_graph:
            # Agent 노드에서 직접 확인
            agent_node_updated = scene_graph.get("nodes", {}).get("agent", {})
            if agent_node_updated.get("isHolding", False):
                held_obj_id = agent_node_updated.get("heldObjectId")
                return False, f"Agent가 이미 객체를 들고 있음 (heldObjectId: {held_obj_id}, 업데이트된 Scene Graph)"
            
            # HOLDS 엣지 확인
            edges = scene_graph.get("edges", [])
            holds_edges = [e for e in edges if e.get("edgeType") == "HOLDS"]
            if holds_edges:
                held_obj_id = holds_edges[0].get("target")
                return False, f"Agent가 이미 객체를 들고 있음 (heldObjectId: {held_obj_id}, 업데이트된 Scene Graph)"
        
        # scene_graph에서 확인 실패 시 scene_context 사용 (fallback)
        # HOLDS 엣지 확인
        holds_edges = [e for e in relevant_edges if e.get("edgeType") == "HOLDS"]
        
        if holds_edges:
            held_obj_id = holds_edges[0].get("target")
            return False, f"Agent가 이미 객체를 들고 있음 (heldObjectId: {held_obj_id})"
        
        # Agent 노드의 isHolding 확인
        if agent_node.get("isHolding", False):
            held_obj_id = agent_node.get("heldObjectId")
            return False, f"Agent가 이미 객체를 들고 있음 (heldObjectId: {held_obj_id})"
        
        return True, "Agent의 손이 비어있음"
    
    # IN(object, receptacle) 검증
    if "IN" in guard_upper and "¬" not in guard_name:
        if target_obj is None:
            return False, "타겟 객체가 없음"
        
        # IN 엣지 확인
        in_edges = [e for e in relevant_edges if e.get("edgeType") == "IN" and e.get("source") == target_obj.get("nodeId")]
        
        if in_edges:
            receptacle_id = in_edges[0].get("target")
            return True, f"IN 엣지 존재 (객체가 수용체 '{receptacle_id}' 안에 있음)"
        
        # parentReceptacles 확인
        parent_receptacles = target_obj.get("parentReceptacles", [])
        if parent_receptacles:
            return True, f"parentReceptacles에 수용체가 있음: {parent_receptacles}"
        
        return False, "객체가 수용체 안에 있지 않음"
    
    # ¬IN(object, closed_receptacle) 검증
    if "IN" in guard_upper and "¬" in guard_name:
        if target_obj is None:
            return True, "타겟 객체가 없지만 전제조건이므로 통과"
        
        # parentReceptacles 확인
        parent_receptacles = target_obj.get("parentReceptacles", [])
        if not parent_receptacles:
            return True, "객체가 수용체 안에 있지 않음"
        
        # 모든 부모 수용체가 열려있는지 확인
        all_open = True
        closed_receptacles = []
        
        for recp_id in parent_receptacles:
            # 수용체 노드 찾기
            recp_node = None
            
            # scene_graph를 우선적으로 사용 (업데이트된 정보 반영)
            if scene_graph:
                object_nodes = scene_graph.get("nodes", {}).get("objects", [])
                for obj_node in object_nodes:
                    if obj_node.get("nodeId") == recp_id:
                        recp_node = obj_node
                        break
            
            # scene_graph에서 못 찾으면 relevantObjects에서 찾기 (fallback)
            if not recp_node:
                for obj in scene_context.get("relevantObjects", []):
                    if obj.get("nodeId") == recp_id:
                        recp_node = obj
                        break
            
            if recp_node:
                is_openable = recp_node.get("openable", False)
                is_open = recp_node.get("isOpen", False)
                openness = recp_node.get("openness", 0.0)
                
                logger.info(f"      수용체 '{recp_id}': openable={is_openable}, isOpen={is_open}, openness={openness}")
                
                if is_openable and not is_open:
                    all_open = False
                    closed_receptacles.append(recp_id)
                    logger.warning(f"      → 수용체 '{recp_id}'가 닫혀있음 (openable=True, isOpen=False)")
                else:
                    logger.info(f"      → 수용체 '{recp_id}'가 열려있거나 openable이 아님 (openable={is_openable}, isOpen={is_open})")
            else:
                # 수용체 노드를 찾지 못한 경우, openable이 아니라고 가정하고 통과
                logger.warning(f"      수용체 노드를 찾을 수 없음: {recp_id}")
        
        if all_open:
            return True, "모든 부모 수용체가 열려있거나 openable이 아님"
        else:
            return False, f"닫힌 수용체가 있음: {closed_receptacles}"
    
    # OPENED(object) 검증
    if "OPENED" in guard_upper and "¬" not in guard_name:
        target = receptacle_obj if receptacle_obj else target_obj
        if target is None:
            return False, "대상 객체가 없음"
        
        # openable이 False이면 OPENED 검증 불필요 (자동 통과)
        is_openable = target.get("openable", False)
        if not is_openable:
            return True, "객체가 openable이 아니므로 OPENED 검증 불필요"
        
        is_open = target.get("isOpen", False)
        if is_open:
            return True, f"객체가 열려있음 (openness: {target.get('openness', 'N/A')})"
        else:
            return False, "객체가 닫혀있음"
    
    # ¬OPENED(object) 검증
    if "OPENED" in guard_upper and "¬" in guard_name:
        target = receptacle_obj if receptacle_obj else target_obj
        if target is None:
            return True, "대상 객체가 없지만 전제조건이므로 통과"
        
        # openable이 False이면 OPENED 검증 불필요 (자동 통과)
        is_openable = target.get("openable", False)
        if not is_openable:
            return True, "객체가 openable이 아니므로 OPENED 검증 불필요"
        
        is_open = target.get("isOpen", False)
        if not is_open:
            return True, "객체가 닫혀있음"
        else:
            return False, f"객체가 이미 열려있음 (openness: {target.get('openness', 'N/A')})"
    
    # pickupable(object) 검증
    if "PICKUPABLE" in guard_upper:
        if target_obj is None:
            return False, "타겟 객체가 없음"
        
        is_pickupable = target_obj.get("pickupable", False)
        if is_pickupable:
            return True, "객체가 pickupable임"
        else:
            return False, "객체가 pickupable이 아님"
    
    # receptacle(receptacle) 검증
    if "RECEPTACLE" in guard_upper:
        target = receptacle_obj if receptacle_obj else target_obj
        if target is None:
            return False, "수용체 객체가 없음"
        
        is_receptacle = target.get("receptacle", False)
        if is_receptacle:
            return True, "객체가 receptacle임"
        else:
            return False, "객체가 receptacle이 아님"
    
    # openable(object) 검증
    if "OPENABLE" in guard_upper:
        target = receptacle_obj if receptacle_obj else target_obj
        if target is None:
            return False, "대상 객체가 없음"
        
        is_openable = target.get("openable", False)
        if is_openable:
            return True, "객체가 openable임"
        else:
            return False, "객체가 openable이 아님"
    
    # toggleable(object) 검증
    if "TOGGLEABLE" in guard_upper:
        if target_obj is None:
            return False, "타겟 객체가 없음"
        
        is_toggleable = target_obj.get("toggleable", False)
        if is_toggleable:
            return True, "객체가 toggleable임"
        else:
            return False, "객체가 toggleable이 아님"
    
    # isToggled(object) 검증
    if "ISTOGGLED" in guard_upper and "¬" not in guard_name:
        if target_obj is None:
            return False, "타겟 객체가 없음"
        
        is_toggled = target_obj.get("isToggled", False)
        if is_toggled:
            return True, "객체가 켜져있음"
        else:
            return False, "객체가 꺼져있음"
    
    # ¬isToggled(object) 검증
    if "ISTOGGLED" in guard_upper and "¬" in guard_name:
        if target_obj is None:
            return True, "타겟 객체가 없지만 전제조건이므로 통과"
        
        is_toggled = target_obj.get("isToggled", False)
        if not is_toggled:
            return True, "객체가 꺼져있음"
        else:
            return False, "객체가 이미 켜져있음"
    
    # sliceable(object) 검증
    if "SLICEABLE" in guard_upper:
        if target_obj is None:
            return False, "타겟 객체가 없음"
        
        is_sliceable = target_obj.get("sliceable", False)
        if is_sliceable:
            return True, "객체가 sliceable임"
        else:
            return False, "객체가 sliceable이 아님"
    
    # ¬isSliced(object) 검증
    if "ISSLICED" in guard_upper and "¬" in guard_name:
        if target_obj is None:
            return True, "타겟 객체가 없지만 전제조건이므로 통과"
        
        is_sliced = target_obj.get("isSliced", False)
        if not is_sliced:
            return True, "객체가 아직 자르지 않음"
        else:
            return False, "객체가 이미 잘림"
    
    # breakable(object) 검증
    if "BREAKABLE" in guard_upper:
        if target_obj is None:
            return False, "타겟 객체가 없음"
        
        is_breakable = target_obj.get("breakable", False)
        if is_breakable:
            return True, "객체가 breakable임"
        else:
            return False, "객체가 breakable이 아님"
    
    # ¬isBroken(object) 검증
    if "ISBROKEN" in guard_upper and "¬" in guard_name:
        if target_obj is None:
            return True, "타겟 객체가 없지만 전제조건이므로 통과"
        
        is_broken = target_obj.get("isBroken", False)
        if not is_broken:
            return True, "객체가 아직 깨지지 않음"
        else:
            return False, "객체가 이미 깨짐"
    
    # 알 수 없는 가드
    return False, f"알 수 없는 가드 타입: {guard_name}"


def verify_action_with_scene_graph(
    action: Dict[str, Any],
    scene_graph: Dict[str, Any],
    controller: Optional[Any] = None
) -> Tuple[bool, str, List[str], List[Dict[str, Any]]]:
    """
    Scene Graph를 활용하여 액션의 물리적 검증 수행
    
    Args:
        action: 액션 딕셔너리
        scene_graph: Scene Graph 딕셔너리
        controller: AI2-THOR Controller (NavMesh 검증용, 선택사항)
        
    Returns:
        (검증 통과 여부, 이유, 실패한 가드 리스트)
    """
    action_type = action.get("type", "")
    args = action.get("args", {})
    object_name = args.get("o")
    receptacle_name = args.get("r")
    
    logger.info(f"  → 물리적 검증: {action_type}('{object_name}'" + 
                (f", '{receptacle_name}')" if receptacle_name else ")"))
    
    # Scene Graph Extractor를 활용하여 에이전트와 목표 객체 정보 출력
    if find_target_object and object_name:
        logger.info("  📊 에이전트 및 목표 객체 정보:")
        
        # 에이전트 정보 출력
        agent_node = scene_graph.get("nodes", {}).get("agent", {})
        if agent_node:
            agent_pos = agent_node.get("position", {})
            is_holding = agent_node.get("isHolding", False)
            held_object_id = agent_node.get("heldObjectId", None)
            logger.info(f"    🤖 에이전트 위치: ({agent_pos.get('x', 0):.3f}, {agent_pos.get('y', 0):.3f}, {agent_pos.get('z', 0):.3f})")
            logger.info(f"    🤖 에이전트 회전: {agent_node.get('rotation', {}).get('y', 0):.1f}°")
            logger.info(f"    🤖 손에 들고 있는 객체: {held_object_id if is_holding else 'None'}")
        
        # 목표 객체 정보 찾기 및 출력
        matched_objects = find_target_object(scene_graph, object_name)
        if matched_objects:
            target_obj = matched_objects[0]  # 첫 번째 매칭 객체 사용
            obj_pos = target_obj.get("position", {})
            obj_type = target_obj.get("objectType", "N/A")
            distance = target_obj.get("distance", 0)
            
            logger.info(f"    📦 목표 객체: {obj_type}")
            logger.info(f"    📦 객체 위치: ({obj_pos.get('x', 0):.3f}, {obj_pos.get('y', 0):.3f}, {obj_pos.get('z', 0):.3f})")
            logger.info(f"    📦 거리: {distance:.3f}m")
            
            # 속성 정보 (True인 것만)
            attributes = []
            if target_obj.get('pickupable', False):
                attributes.append("pickupable")
            if target_obj.get('openable', False):
                attributes.append("openable")
            if target_obj.get('receptacle', False):
                attributes.append("receptacle")
            if target_obj.get('toggleable', False):
                attributes.append("toggleable")
            if target_obj.get('visible', False):
                attributes.append("visible")
            if attributes:
                logger.info(f"    📦 속성: {', '.join(attributes)}")
            
            # 상태 정보
            states = []
            if target_obj.get('isOpen', False):
                states.append("isOpen")
            if target_obj.get('isToggled', False):
                states.append("isToggled")
            if target_obj.get('isPickedUp', False):
                states.append("isPickedUp")
            if states:
                logger.info(f"    📦 상태: {', '.join(states)}")
            
            # 부모 수용체 정보
            parent_receptacles = target_obj.get("parentReceptacles", [])
            if parent_receptacles:
                logger.info(f"    📦 부모 수용체: {', '.join(parent_receptacles)}")
                logger.info(f"    📦 부모 수용체 속성: Openable={target_obj.get('openable', False)}, isOpen={target_obj.get('isOpen', False)}, openness={target_obj.get('openness', 0.0)}")
            
            # 관련 엣지 정보
            if get_related_edges:
                related_edges = get_related_edges(scene_graph, target_obj)
                if related_edges:
                    edge_types = [e.get("edgeType", "UNKNOWN") for e in related_edges]
                    logger.info(f"    📦 관련 엣지: {', '.join(set(edge_types))}")
        
        # 수용체 정보 (PutObject의 경우)
        if receptacle_name and find_target_object:
            matched_receptacles = find_target_object(scene_graph, receptacle_name)
            if matched_receptacles:
                recp_obj = matched_receptacles[0]
                recp_pos = recp_obj.get("position", {})
                recp_type = recp_obj.get("objectType", "N/A")
                recp_distance = recp_obj.get("distance", 0)
                is_open = recp_obj.get("isOpen", False)
                
                logger.info(f"    📦 수용체: {recp_type}")
                logger.info(f"    📦 수용체 위치: ({recp_pos.get('x', 0):.3f}, {recp_pos.get('y', 0):.3f}, {recp_pos.get('z', 0):.3f})")
                logger.info(f"    📦 수용체 거리: {recp_distance:.3f}m")
                logger.info(f"    📦 수용체 열림 상태: {is_open}")
    
    # Scene Graph에서 관련 정보 추출
    scene_context = get_relevant_scene_context(
        scene_graph, action_type, object_name, receptacle_name
    )
    
    # Agent 위치 가져오기 (REACHABLE 검증에 사용)
    agent_node = scene_graph.get("nodes", {}).get("agent", {})
    agent_position = agent_node.get("position", {}) if agent_node else None
    
    # 액션별 기본 가드 설정
    if action_type == "GoToObject":
        guards = ["EXISTS(object)", "NAVIGABLE(agent, object)"]
    elif action_type == "PickupObject":
        guards = ["EXISTS(object)", "Proximity(agent, object)", "pickupable(object)", 
                 "REACHABLE(agent, object)", "¬HOLDS(agent, *)", "¬IN(object, closed_receptacle)"]
    elif action_type == "PutObject":
        # 수용체의 openable 여부 확인하여 OPENED 가드 추가 여부 결정
        guards = ["EXISTS(object)", "EXISTS(receptacle)", "Proximity(agent, receptacle)", 
                 "HOLDS(agent, object)", "receptacle(receptacle)",
                 "REACHABLE(agent, receptacle)", "openable(receptacle)"]
        # openable이 True인 경우에만 OPENED 검증 추가
        if receptacle_name and find_target_object:
            matched_receptacles = find_target_object(scene_graph, receptacle_name)
            if matched_receptacles:
                recp_obj = matched_receptacles[0]
                if recp_obj.get("openable", False):
                    guards.append("OPENED(receptacle)")
    elif action_type == "OpenObject":
        guards = ["EXISTS(object)", "Proximity(agent, object)", "openable(object)", "¬OPENED(object)", 
                 "REACHABLE(agent, object)"]
    elif action_type == "CloseObject":
        guards = ["EXISTS(object)", "Proximity(agent, object)", "openable(object)", "OPENED(object)",
                 "REACHABLE(agent, object)"]
    elif action_type == "ToggleObjectOn":
        guards = ["EXISTS(object)", "Proximity(agent, object)", "toggleable(object)", 
        "REACHABLE(agent, object)", "¬isToggled(object)", "¬IN(object, closed_receptacle)"]
    elif action_type == "ToggleObjectOff":
        guards = ["EXISTS(object)", "Proximity(agent, object)", "toggleable(object)", 
        "REACHABLE(agent, object)", "isToggled(object)"]
    elif action_type == "SliceObject":
        guards = ["EXISTS(object)", "Proximity(agent, object)", "sliceable(object)", 
        "¬isSliced(object)", "REACHABLE(agent, object)", "HOLDS(agent, 'Knife')", "¬IN(object, closed_receptacle)"]
    elif action_type == "BreakObject":
        guards = ["EXISTS(object)", "Proximity(agent, object)", "breakable(object)", 
        "¬isBroken(object)", "REACHABLE(agent, object)", "¬IN(object, closed_receptacle)"]
    else:
        guards = ["EXISTS(object)"]
    
    logger.info(f"  → 검증할 Guards: {guards}")
    
    # 각 가드를 Scene Graph로 직접 검증
    failed_guards = []
    all_passed = True
    object_not_exists = False  # EXISTS 가드 실패 플래그
    target_position = None  # GoToObject 검증 통과 시 이동할 좌표
    
    for guard in guards:
        passed, reason = verify_guard_with_scene_graph(
            guard, scene_context, action_type, object_name, receptacle_name, controller, scene_graph, agent_position
        )
        
        if passed:
            logger.info(f"    ✓ {guard}: {reason}")
            
            # GoToObject의 NAVIGABLE 가드 통과 시 이동할 좌표 저장
            if action_type == "GoToObject" and "NAVIGABLE" in guard.upper():
                target_obj = scene_context.get("targetObject")
                if target_obj and controller:
                    obj_pos = target_obj.get("position", {})
                    if obj_pos:
                        # Agent 위치 전달하여 거리만 고려한 가장 가까운 위치 선택
                        closest_pos, distance, closest_by_distance = find_closest_reachable_position(
                            controller, obj_pos, agent_position, return_closest_only=True
                        )
                        # 거리만 고려한 가장 가까운 위치 사용
                        if closest_by_distance:
                            closest_pos = closest_by_distance
                            # 거리 재계산
                            obj_x = obj_pos.get("x", 0)
                            obj_z = obj_pos.get("z", 0)
                            closest_x = closest_pos.get("x", 0)
                            closest_z = closest_pos.get("z", 0)
                            distance = math.sqrt((closest_x - obj_x)**2 + (closest_z - obj_z)**2)
                        if closest_pos:
                            target_position = closest_pos
                            logger.info(f"    📍 이동할 좌표 (가장 가까운 위치): ({closest_pos.get('x', 0):.3f}, {closest_pos.get('y', 0):.3f}, {closest_pos.get('z', 0):.3f}) (거리: {distance:.3f}m)")
        else:
            logger.warning(f"    ✗ {guard}: {reason}")
            failed_guards.append(guard)
            all_passed = False
            
            # EXISTS 가드 실패 시 즉시 중단 플래그 설정
            if "EXISTS" in guard:
                object_not_exists = True
                break  # EXISTS 실패 시 나머지 가드 검증 중단
    
    # 복구 액션 생성 (특정 가드 실패 시)
    recovery_actions = []
    
    if not all_passed:
        # 타겟 객체 노드 가져오기 (복구 액션 생성에 필요)
        target_obj = scene_context.get("targetObject")
        receptacle_obj = scene_context.get("receptacleObject")
        
        # PickupObject: ¬IN(object, closed_receptacle) 실패 → 부모 수용체 열기
        if action_type == "PickupObject" and "¬IN(object, closed_receptacle)" in failed_guards:
            # 부모 수용체 찾기
            if target_obj:
                parent_receptacles = target_obj.get("parentReceptacles", [])
                for recp_id in parent_receptacles:
                    # 수용체 노드 찾기
                    recp_node = None
                    if scene_graph:
                        object_nodes = scene_graph.get("nodes", {}).get("objects", [])
                        for obj_node in object_nodes:
                            if obj_node.get("nodeId") == recp_id:
                                recp_node = obj_node
                                break
                    
                    if recp_node and recp_node.get("openable", False) and not recp_node.get("isOpen", False):
                        # 수용체를 여는 액션 생성
                        recp_type = recp_node.get("objectType", "")
                        # nodeId 전체를 사용하여 정확한 수용체 지정 (좌표 포함)
                        # recp_id는 이미 정확한 nodeId (예: "Drawer|-01.56|+00.84|-00.20")
                        
                        recovery_action = {
                            "type": "OpenObject",
                            "args": {"o": recp_id},  # nodeId 전체 사용
                            "line": f"OpenObject('{recp_id}')",  # nodeId 전체 포함
                            "nodeId": recp_id,  # nodeId를 별도로 저장
                            "reason": f"부모 수용체 '{recp_id}' 열기 (객체 '{object_name}' 접근을 위해)",
                            "is_original": False,
                            "is_recovery": True,
                            "failed_guards": ["¬IN(object, closed_receptacle)"],
                            "recovery_reason": f"닫힌 수용체 '{recp_id}' 내부 객체 접근을 위해 수용체 열기"
                        }
                        recovery_actions.append(recovery_action)
                        logger.info(f"    → 복구 액션 생성: OpenObject('{recp_id}')")
                        break  # 첫 번째 닫힌 수용체만 처리
        
        # PutObject: HOLDS(agent, object) 실패 → GoToObject와 PickupObject 추가
        if action_type == "PutObject" and "HOLDS(agent, object)" in failed_guards:
            # 목표 객체 찾기
            if object_name and target_obj:
                obj_type = target_obj.get("objectType", "")
                obj_id = target_obj.get("nodeId", "")

                # GoToObject 복구 액션 생성 (먼저 실행되어야 함)
                goto_recovery = {
                    "type": "GoToObject",
                    "args": {"o": object_name},
                    "line": f"GoToObject('{object_name}')",
                    "reason": f"객체 '{object_name}'로 이동 (PickupObject를 위해)",
                    "is_original": False,
                    "is_recovery": True,
                    "failed_guards": ["HOLDS(agent, object)"],
                    "recovery_reason": f"PutObject를 위해 객체 '{object_name}'로 이동"
                }
                recovery_actions.append(goto_recovery)
                logger.info(f"    → 복구 액션 생성: GoToObject('{object_name}')")

                # PickupObject 복구 액션 생성 (GoToObject 이후 실행)
                pickup_recovery = {
                    "type": "PickupObject",
                    "args": {"o": object_name},
                    "line": f"PickupObject('{object_name}')",
                    "reason": f"객체 '{object_name}' 집기 (PutObject를 위해)",
                    "is_original": False,
                    "is_recovery": True,
                    "failed_guards": ["HOLDS(agent, object)"],
                    "recovery_reason": f"PutObject를 위해 객체 '{object_name}' 집기"
                }
                recovery_actions.append(pickup_recovery)
                logger.info(f"    → 복구 액션 생성: PickupObject('{object_name}')")
                
                
        
        # Proximity 가드 실패 → GoToObject 추가 (NavMesh 상 목표 객체를 정면으로 보는 위치로 이동)
        if any("Proximity" in g for g in failed_guards):
            # 목표 객체 찾기
            target_for_proximity = None
            target_name_for_proximity = None
            
            if action_type == "PutObject":
                # PutObject의 경우 receptacle을 타겟으로 사용
                if receptacle_obj:
                    target_for_proximity = receptacle_obj
                    target_name_for_proximity = receptacle_name if receptacle_name else object_name
                elif receptacle_name and scene_graph:
                    # receptacle_obj가 None이면 scene_graph에서 직접 찾기
                    if find_target_object:
                        matched_receptacles = find_target_object(scene_graph, receptacle_name)
                        if matched_receptacles:
                            target_for_proximity = matched_receptacles[0]
                            target_name_for_proximity = receptacle_name
                    # find_target_object가 없으면 receptacle_name만 사용
                    if not target_for_proximity:
                        target_name_for_proximity = receptacle_name
                else:
                    target_for_proximity = target_obj
                    target_name_for_proximity = object_name
            else:
                target_for_proximity = target_obj
                target_name_for_proximity = object_name
            
            # target_name_for_proximity가 있으면 복구 액션 생성 (target_for_proximity가 None이어도 이름으로 이동 가능)
            if target_name_for_proximity:
                # NavMesh 상에서 목표 객체를 정면으로 보는 가장 가까운 위치 계산
                target_position = None
                if controller and target_for_proximity:
                    obj_pos = target_for_proximity.get("position", {})
                    if obj_pos:
                        # Agent 위치 전달하여 목표 객체를 정면으로 보는 위치 선택
                        closest_pos, distance, closest_by_distance = find_closest_reachable_position(
                            controller, obj_pos, agent_position, return_closest_only=True
                        )
                        if closest_pos:
                            target_position = closest_pos
                            logger.info(f"    📍 Proximity 복구: 목표 객체 '{target_name_for_proximity}'를 정면으로 보는 위치 계산 완료: ({closest_pos.get('x', 0):.3f}, {closest_pos.get('y', 0):.3f}, {closest_pos.get('z', 0):.3f}) (거리: {distance:.3f}m)")
                
                goto_proximity_recovery = {
                    "type": "GoToObject",
                    "args": {"o": target_name_for_proximity},
                    "line": f"GoToObject('{target_name_for_proximity}')",
                    "reason": f"객체 '{target_name_for_proximity}'로 이동 (Proximity 가드 위반)",
                    "is_original": False,
                    "is_recovery": True,
                    "failed_guards": ["Proximity"],
                    "recovery_reason": f"Agent와 객체 '{target_name_for_proximity}' 간 거리가 2m를 초과하여 이동 필요",
                    "target_position": target_position  # NavMesh 상 계산된 위치 저장
                }
                recovery_actions.append(goto_proximity_recovery)
                if target_position:
                    logger.info(f"    → 복구 액션 생성: GoToObject('{target_name_for_proximity}') (Proximity 가드 위반, 목표 위치: ({target_position.get('x', 0):.3f}, {target_position.get('y', 0):.3f}, {target_position.get('z', 0):.3f}))")
                else:
                    logger.info(f"    → 복구 액션 생성: GoToObject('{target_name_for_proximity}') (Proximity 가드 위반, Controller 없음으로 위치 계산 불가)")
        
        # REACHABLE 가드 실패 시 복구 불가능 - 검증 종료 처리 (아래에서 처리)
        
        # SliceObject: HOLDS(agent, 'Knife') 실패 → Knife로 이동하고 집기
        if action_type == "SliceObject" and "HOLDS(agent, 'Knife')" in failed_guards:
            # Knife로 이동하고 집기
            goto_knife_recovery = {
                "type": "GoToObject",
                "args": {"o": "Knife"},
                "line": "GoToObject('Knife')",
                "reason": "Knife로 이동 (SliceObject를 위해)",
                "is_original": False,
                "is_recovery": True,
                "failed_guards": ["HOLDS(agent, 'Knife')"],
                "recovery_reason": "SliceObject를 위해 Knife로 이동"
            }
            recovery_actions.append(goto_knife_recovery)
            logger.info(f"    → 복구 액션 생성: GoToObject('Knife')")
            
            pickup_knife_recovery = {
                "type": "PickupObject",
                "args": {"o": "Knife"},
                "line": "PickupObject('Knife')",
                "reason": "Knife 집기 (SliceObject를 위해)",
                "is_original": False,
                "is_recovery": True,
                "failed_guards": ["HOLDS(agent, 'Knife')"],
                "recovery_reason": "SliceObject를 위해 Knife 집기"
            }
            recovery_actions.append(pickup_knife_recovery)
            logger.info(f"    → 복구 액션 생성: PickupObject('Knife')")
        
        # SliceObject, BreakObject: ¬IN(object, closed_receptacle) 실패 → 부모 수용체 열기
        if action_type in ["SliceObject", "BreakObject"] and "¬IN(object, closed_receptacle)" in failed_guards:
            # 부모 수용체 찾기
            if target_obj:
                parent_receptacles = target_obj.get("parentReceptacles", [])
                for recp_id in parent_receptacles:
                    # 수용체 노드 찾기
                    recp_node = None
                    if scene_graph:
                        object_nodes = scene_graph.get("nodes", {}).get("objects", [])
                        for obj_node in object_nodes:
                            if obj_node.get("nodeId") == recp_id:
                                recp_node = obj_node
                                break
                    
                    if recp_node and recp_node.get("openable", False) and not recp_node.get("isOpen", False):
                        recovery_action = {
                            "type": "OpenObject",
                            "args": {"o": recp_id},
                            "line": f"OpenObject('{recp_id}')",
                            "nodeId": recp_id,
                            "reason": f"부모 수용체 '{recp_id}' 열기 (객체 '{object_name}' 접근을 위해)",
                            "is_original": False,
                            "is_recovery": True,
                            "failed_guards": ["¬IN(object, closed_receptacle)"],
                            "recovery_reason": f"닫힌 수용체 '{recp_id}' 내부 객체 접근을 위해 수용체 열기"
                        }
                        recovery_actions.append(recovery_action)
                        logger.info(f"    → 복구 액션 생성: OpenObject('{recp_id}')")
                        break  # 첫 번째 닫힌 수용체만 처리
        
        # ToggleObjectOn: ¬IN(object, closed_receptacle) 실패 → 부모 수용체 열기
        if action_type == "ToggleObjectOn" and "¬IN(object, closed_receptacle)" in failed_guards:
            # 부모 수용체 찾기
            if target_obj:
                parent_receptacles = target_obj.get("parentReceptacles", [])
                for recp_id in parent_receptacles:
                    # 수용체 노드 찾기
                    recp_node = None
                    if scene_graph:
                        object_nodes = scene_graph.get("nodes", {}).get("objects", [])
                        for obj_node in object_nodes:
                            if obj_node.get("nodeId") == recp_id:
                                recp_node = obj_node
                                break
                    
                    if recp_node and recp_node.get("openable", False) and not recp_node.get("isOpen", False):
                        recovery_action = {
                            "type": "OpenObject",
                            "args": {"o": recp_id},
                            "line": f"OpenObject('{recp_id}')",
                            "nodeId": recp_id,
                            "reason": f"부모 수용체 '{recp_id}' 열기 (객체 '{object_name}' 접근을 위해)",
                            "is_original": False,
                            "is_recovery": True,
                            "failed_guards": ["¬IN(object, closed_receptacle)"],
                            "recovery_reason": f"닫힌 수용체 '{recp_id}' 내부 객체 접근을 위해 수용체 열기"
                        }
                        recovery_actions.append(recovery_action)
                        logger.info(f"    → 복구 액션 생성: OpenObject('{recp_id}')")
                        break  # 첫 번째 닫힌 수용체만 처리
        
        # PutObject: OPENED(receptacle) 실패 → 수용체 열기
        elif action_type == "PutObject" and "OPENED(receptacle)" in failed_guards:
            # 수용체 찾기
            if receptacle_name and find_target_object:
                matched_receptacles = find_target_object(scene_graph, receptacle_name)
                if matched_receptacles:
                    recp_obj = matched_receptacles[0]
                    if recp_obj.get("openable", False) and not recp_obj.get("isOpen", False):
                        recp_type = recp_obj.get("objectType", "")
                        recp_id = recp_obj.get("nodeId", "")
                        # nodeId 전체를 사용하여 정확한 수용체 지정 (좌표 포함)
                        
                        recovery_action = {
                            "type": "OpenObject",
                            "args": {"o": recp_id},  # nodeId 전체 사용
                            "line": f"OpenObject('{recp_id}')",  # nodeId 전체 포함
                            "nodeId": recp_id,  # nodeId를 별도로 저장
                            "reason": f"수용체 '{recp_id}' 열기 (PutObject를 위해)",
                            "is_original": False,
                            "is_recovery": True,
                            "failed_guards": ["OPENED(receptacle)"],
                            "recovery_reason": f"PutObject를 위해 수용체 '{recp_id}' 열기"
                        }
                        recovery_actions.append(recovery_action)
                        logger.info(f"    → 복구 액션 생성: OpenObject('{recp_id}')")
    
    # EXISTS 가드 실패 시 특별 처리 (객체가 존재하지 않음)
    if object_not_exists:
        missing_object = object_name or receptacle_name or "알 수 없음"
        reason = f"객체 '{missing_object}'가 Scene Graph에 존재하지 않음"
        logger.error(f"  ✗ EXISTS 가드 실패: {reason}")
        return False, f"OBJECT_NOT_EXISTS: {reason}", failed_guards, []
    
    if all_passed:
        reason = f"모든 가드 통과 ({len(guards)}개)"
        logger.info(f"  ✓ 물리적 검증 통과: {reason}")
        
        # GoToObject 검증 통과 시 이동할 좌표를 action에 저장
        if action_type == "GoToObject" and target_position:
            action["target_position"] = target_position
        
        return True, f"PASS: {reason}", [], []
    else:
        reason = f"{len(failed_guards)}개 가드 실패: {', '.join(failed_guards)}"
        logger.warning(f"  ✗ 물리적 검증 실패: {reason}")
        return False, f"FAIL: {reason}", failed_guards, recovery_actions


def save_scene_graph_to_file(scene_graph: Dict[str, Any], scene_graph_path: str):
    """Scene Graph를 JSON 파일에 저장"""
    try:
        with open(scene_graph_path, "w", encoding="utf-8") as f:
            json.dump(scene_graph, f, indent=2, ensure_ascii=False)
        logger.debug(f"Scene Graph 저장 완료: {scene_graph_path}")
    except Exception as e:
        logger.error(f"Scene Graph 저장 실패: {e}")


def print_action_summary(action: Dict[str, Any], scene_graph: Dict[str, Any], object_name: Optional[str] = None, receptacle_name: Optional[str] = None):
    """액션 통과 후 상태 요약 출력"""
    action_type = action.get("type", "")
    logger.info(f"\n  📋 액션 실행 후 상태 요약:")
    
    # Agent 상태
    agent_node = scene_graph.get("nodes", {}).get("agent", {})
    agent_pos = agent_node.get("position", {})
    agent_rot = agent_node.get("rotation", {})
    is_holding = agent_node.get("isHolding", False)
    held_object_id = agent_node.get("heldObjectId", None)
    
    logger.info(f"    🤖 Agent:")
    logger.info(f"      위치: ({agent_pos.get('x', 0):.3f}, {agent_pos.get('y', 0):.3f}, {agent_pos.get('z', 0):.3f})")
    logger.info(f"      회전: ({agent_rot.get('x', 0):.1f}°, {agent_rot.get('y', 0):.1f}°, {agent_rot.get('z', 0):.1f}°)")
    logger.info(f"      손에 들고 있는 객체: {held_object_id if is_holding else 'None'}")
    
    # 목표 객체 상태
    if object_name and find_target_object:
        matched_objects = find_target_object(scene_graph, object_name)
        if matched_objects:
            target_obj = matched_objects[0]
            obj_id = target_obj.get("nodeId", "N/A")
            obj_type = target_obj.get("objectType", "N/A")
            obj_pos = target_obj.get("position", {})
            
            logger.info(f"    📦 목표 객체 ({object_name}):")
            logger.info(f"      nodeId: {obj_id}")
            logger.info(f"      타입: {obj_type}")
            logger.info(f"      위치: ({obj_pos.get('x', 0):.3f}, {obj_pos.get('y', 0):.3f}, {obj_pos.get('z', 0):.3f})")
            
            # 상태 정보
            states = []
            if target_obj.get('isOpen', False):
                states.append(f"isOpen=True (openness={target_obj.get('openness', 0.0):.2f})")
            if target_obj.get('isPickedUp', False):
                states.append("isPickedUp=True")
            if target_obj.get('isToggled', False):
                states.append("isToggled=True")
            if states:
                logger.info(f"      상태: {', '.join(states)}")
            
            # parentReceptacles 정보
            parent_receptacles = target_obj.get("parentReceptacles", [])
            if parent_receptacles:
                logger.info(f"      부모 수용체: {', '.join(parent_receptacles)}")
    
    # Receptacle 상태 (PutObject의 경우)
    if receptacle_name and find_target_object:
        matched_receptacles = find_target_object(scene_graph, receptacle_name)
        if matched_receptacles:
            recp_obj = matched_receptacles[0]
            recp_id = recp_obj.get("nodeId", "N/A")
            recp_type = recp_obj.get("objectType", "N/A")
            recp_pos = recp_obj.get("position", {})
            
            logger.info(f"    📦 수용체 ({receptacle_name}):")
            logger.info(f"      nodeId: {recp_id}")
            logger.info(f"      타입: {recp_type}")
            logger.info(f"      위치: ({recp_pos.get('x', 0):.3f}, {recp_pos.get('y', 0):.3f}, {recp_pos.get('z', 0):.3f})")
            
            # 상태 정보
            recp_states = []
            if recp_obj.get('openable', False):
                recp_states.append("openable=True")
                if recp_obj.get('isOpen', False):
                    recp_states.append(f"isOpen=True (openness={recp_obj.get('openness', 0.0):.2f})")
                else:
                    recp_states.append("isOpen=False")
            if recp_states:
                logger.info(f"      상태: {', '.join(recp_states)}")


def update_scene_graph_after_action(
    scene_graph: Dict[str, Any],
    action: Dict[str, Any],
    verification_passed: bool,
    scene_graph_path: Optional[str] = None,
    controller: Optional[Any] = None
) -> Dict[str, Any]:
    """
    액션 실행 후 Scene Graph 업데이트 및 JSON 파일 저장
    
    Args:
        scene_graph: 현재 Scene Graph
        action: 실행된 액션
        verification_passed: 검증 통과 여부
        scene_graph_path: Scene Graph JSON 파일 경로 (None이면 저장 안 함)
        
    Returns:
        업데이트된 Scene Graph
    """
    if not verification_passed:
        return scene_graph  # 검증 실패 시 업데이트 안 함
    
    action_type = action.get("type", "")
    args = action.get("args", {})
    object_name = args.get("o")
    receptacle_name = args.get("r")
    
    nodes = scene_graph.get("nodes", {})
    edges = scene_graph.get("edges", [])
    agent_node = nodes.get("agent", {})
    object_nodes = nodes.get("objects", [])
    
    if action_type == "GoToObject":
        # GoToObject 검증 통과 시 NavMesh에서 찾은 가장 가까운 이동 가능 위치로 Agent 좌표 업데이트
        if object_name:
            # find_target_object를 사용하여 정확한 객체 찾기
            target_obj = None
            if find_target_object:
                matched_objects = find_target_object(scene_graph, object_name)
                if matched_objects:
                    matched_obj = matched_objects[0]
                    matched_node_id = matched_obj.get("nodeId")
                    for obj_node in object_nodes:
                        if obj_node.get("nodeId") == matched_node_id:
                            target_obj = obj_node
                            break
            
            # find_target_object가 없거나 실패한 경우, 정확한 매칭 우선으로 직접 찾기
            if target_obj is None:
                object_name_lower = object_name.lower()
                exact_match = None
                partial_match = None
                
                for obj_node in object_nodes:
                    obj_type = obj_node.get("objectType", "")
                    obj_id = obj_node.get("nodeId", "")
                    obj_type_lower = obj_type.lower()
                    obj_id_lower = obj_id.lower()
                    
                    # "Knife"를 찾을 때 "ButterKnife"는 무조건 제외
                    if object_name_lower == "knife":
                        if "butter" in obj_type_lower or "butter" in obj_id_lower:
                            continue
                    
                    # 정확한 매칭 우선
                    if obj_type_lower == object_name_lower or obj_id_lower == object_name_lower:
                        exact_match = obj_node
                        break
                    # 정확한 매칭이 없으면 부분 매칭
                    elif object_name_lower in obj_type_lower or object_name_lower in obj_id_lower:
                        # 추가 확인: "knife"를 찾을 때 "butterknife"는 제외
                        if object_name_lower == "knife" and ("butter" in obj_type_lower or "butter" in obj_id_lower):
                            continue
                        if partial_match is None:
                            partial_match = obj_node
                
                # "Knife"를 찾을 때는 정확한 매칭만 사용
                target_obj = exact_match if (object_name_lower == "knife" and exact_match) else (exact_match if exact_match else partial_match)
            
            if target_obj:
                # action에 저장된 target_position 사용 (verify_action_with_scene_graph에서 계산된 좌표)
                target_position = action.get("target_position")
                if target_position:
                    # Agent 위치 업데이트 (검증 시 계산된 좌표 사용)
                    agent_node["position"] = {
                        "x": target_position.get("x", 0),
                        "y": target_position.get("y", 0),
                        "z": target_position.get("z", 0)
                    }
                    # 거리 계산 (객체까지의 거리)
                    obj_pos = target_obj.get("position", {})
                    if obj_pos:
                        distance = math.sqrt(
                            (target_position.get("x", 0) - obj_pos.get("x", 0))**2 +
                            (target_position.get("y", 0) - obj_pos.get("y", 0))**2 +
                            (target_position.get("z", 0) - obj_pos.get("z", 0))**2
                        )
                    else:
                        distance = 0.0
                    logger.info(f"  ✓ Agent 위치 업데이트: GoToObject('{object_name}') → ({target_position.get('x', 0):.3f}, {target_position.get('y', 0):.3f}, {target_position.get('z', 0):.3f}) (거리: {distance:.3f}m, 정면 위치)")
                else:
                    # target_position이 없으면 fallback으로 다시 계산
                    obj_pos = target_obj.get("position", {})
                    if obj_pos and controller is not None:
                        # 현재 Agent 위치 가져오기 (정면으로 마주보는 위치 선택을 위해)
                        current_agent_pos = agent_node.get("position", {})
                        
                        # NavMesh에서 목표 객체까지 거리만 고려한 가장 가까운 이동 가능 위치 찾기
                        closest_pos, distance, closest_by_distance = find_closest_reachable_position(
                            controller, obj_pos, current_agent_pos, return_closest_only=True
                        )
                        # 거리만 고려한 가장 가까운 위치 사용
                        if closest_by_distance:
                            closest_pos = closest_by_distance
                            # 거리 재계산
                            obj_x = obj_pos.get("x", 0)
                            obj_z = obj_pos.get("z", 0)
                            closest_x = closest_pos.get("x", 0)
                            closest_z = closest_pos.get("z", 0)
                            distance = math.sqrt((closest_x - obj_x)**2 + (closest_z - obj_z)**2)
                        
                        if closest_pos:
                            # Agent 위치 업데이트
                            agent_node["position"] = {
                                "x": closest_pos.get("x", 0),
                                "y": closest_pos.get("y", 0),
                                "z": closest_pos.get("z", 0)
                            }
                            logger.info(f"  ✓ Agent 위치 업데이트: GoToObject('{object_name}') → ({closest_pos.get('x', 0):.3f}, {closest_pos.get('y', 0):.3f}, {closest_pos.get('z', 0):.3f}) (거리: {distance:.3f}m, 가장 가까운 위치)")
                        else:
                            logger.warning(f"  ⚠️ GoToObject('{object_name}') 후 Agent 위치 업데이트 실패: 이동 가능한 위치를 찾을 수 없음")
                    elif obj_pos and controller is None:
                        logger.warning(f"  ⚠️ GoToObject('{object_name}') 후 Agent 위치 업데이트 실패: Controller가 없음")
    
    elif action_type == "PickupObject":
        # HOLDS 엣지 추가, IN 엣지 제거
        if object_name:
            # find_target_object를 사용하여 정확한 객체 찾기 (정확한 매칭 우선)
            target_obj_node = None
            if find_target_object:
                matched_objects = find_target_object(scene_graph, object_name)
                if matched_objects:
                    # 첫 번째 매칭된 객체의 nodeId로 object_nodes에서 찾기
                    matched_obj = matched_objects[0]
                    matched_node_id = matched_obj.get("nodeId")
                    for obj_node in object_nodes:
                        if obj_node.get("nodeId") == matched_node_id:
                            target_obj_node = obj_node
                            break
            
            # find_target_object가 없거나 실패한 경우, 정확한 매칭 우선으로 직접 찾기
            if target_obj_node is None:
                object_name_lower = object_name.lower()
                exact_matches = []
                partial_matches = []
                
                for obj_node in object_nodes:
                    obj_type = obj_node.get("objectType", "")
                    obj_id = obj_node.get("nodeId", "")
                    obj_type_lower = obj_type.lower()
                    
                    # "Knife"를 찾을 때 "ButterKnife"는 무조건 제외
                    if object_name_lower == "knife":
                        if "butter" in obj_type_lower or "butter" in obj_id.lower():
                            continue
                    
                    # 정확한 매칭 우선
                    if obj_type_lower == object_name_lower:
                        exact_matches.append(obj_node)
                    # 정확한 매칭이 없으면 부분 매칭
                    elif object_name_lower in obj_type_lower:
                        # 추가 확인: "knife"를 찾을 때 "butterknife"는 제외
                        if object_name_lower == "knife" and ("butter" in obj_type_lower or "butter" in obj_id.lower()):
                            continue
                        partial_matches.append(obj_node)
                
                # "Knife"를 찾을 때는 정확한 매칭만 사용
                if object_name_lower == "knife":
                    if exact_matches:
                        target_obj_node = exact_matches[0]
                else:
                    # 정확한 매칭이 있으면 그것만 사용, 없으면 부분 매칭 사용
                    if exact_matches:
                        target_obj_node = exact_matches[0]
                    elif partial_matches:
                        target_obj_node = partial_matches[0]
            
            # 찾은 객체로 Scene Graph 업데이트
            if target_obj_node:
                obj_id = target_obj_node.get("nodeId", "")
                obj_type = target_obj_node.get("objectType", "")
                
                # HOLDS 엣지 추가
                holds_edge = {
                    "edgeType": "HOLDS",
                    "source": "agent_0",
                    "target": obj_id,
                    "sourceType": "Agent",
                    "targetType": "Object",
                    "targetObjectType": obj_type
                }
                if holds_edge not in edges:
                    edges.append(holds_edge)
                
                # IN 엣지 제거
                edges = [e for e in edges if not (
                    e.get("edgeType") == "IN" and e.get("source") == obj_id
                )]
                
                # Agent 노드 업데이트
                agent_node["isHolding"] = True
                agent_node["heldObjectId"] = obj_id
                
                # Object 노드 업데이트
                target_obj_node["isPickedUp"] = True
                if "parentReceptacles" in target_obj_node:
                    target_obj_node["parentReceptacles"] = []
                
                logger.info(f"  → PickupObject: '{object_name}' (nodeId: {obj_id}, type: {obj_type}) 업데이트 완료")
            else:
                logger.warning(f"  → PickupObject: '{object_name}'를 Scene Graph에서 찾을 수 없음")
    
    elif action_type == "PutObject":
        # HOLDS 엣지 제거, IN 엣지 추가
        if object_name and receptacle_name:
            # find_target_object를 사용하여 정확한 객체 찾기 (정확한 매칭 우선)
            target_obj_node = None
            if find_target_object:
                matched_objects = find_target_object(scene_graph, object_name)
                if matched_objects:
                    # 첫 번째 매칭된 객체의 nodeId로 object_nodes에서 찾기
                    matched_obj = matched_objects[0]
                    matched_node_id = matched_obj.get("nodeId")
                    for obj_node in object_nodes:
                        if obj_node.get("nodeId") == matched_node_id:
                            target_obj_node = obj_node
                            break
            
            # find_target_object가 없거나 실패한 경우, 정확한 매칭 우선으로 직접 찾기
            if target_obj_node is None:
                object_name_lower = object_name.lower()
                exact_matches = []
                partial_matches = []
                
                for obj_node in object_nodes:
                    obj_type = obj_node.get("objectType", "")
                    obj_id = obj_node.get("nodeId", "")
                    obj_type_lower = obj_type.lower()
                    
                    # "Knife"를 찾을 때 "ButterKnife"는 무조건 제외
                    if object_name_lower == "knife":
                        if "butter" in obj_type_lower or "butter" in obj_id.lower():
                            continue
                    
                    # 정확한 매칭 우선
                    if obj_type_lower == object_name_lower:
                        exact_matches.append(obj_node)
                    # 정확한 매칭이 없으면 부분 매칭
                    elif object_name_lower in obj_type_lower:
                        # 추가 확인: "knife"를 찾을 때 "butterknife"는 제외
                        if object_name_lower == "knife" and ("butter" in obj_type_lower or "butter" in obj_id.lower()):
                            continue
                        partial_matches.append(obj_node)
                
                # "Knife"를 찾을 때는 정확한 매칭만 사용
                if object_name_lower == "knife":
                    if exact_matches:
                        target_obj_node = exact_matches[0]
                else:
                    # 정확한 매칭이 있으면 그것만 사용, 없으면 부분 매칭 사용
                    if exact_matches:
                        target_obj_node = exact_matches[0]
                    elif partial_matches:
                        target_obj_node = partial_matches[0]
            
            if target_obj_node:
                obj_id = target_obj_node.get("nodeId", "")
                obj_type = target_obj_node.get("objectType", "")
                
                # HOLDS 엣지 제거
                edges = [e for e in edges if not (
                    e.get("edgeType") == "HOLDS" and e.get("target") == obj_id
                )]
                
                # 수용체 찾기 (find_target_object 사용)
                recp_obj_node = None
                if find_target_object:
                    matched_receptacles = find_target_object(scene_graph, receptacle_name)
                    if matched_receptacles:
                        matched_recp = matched_receptacles[0]
                        matched_recp_node_id = matched_recp.get("nodeId")
                        for recp_node in object_nodes:
                            if recp_node.get("nodeId") == matched_recp_node_id:
                                recp_obj_node = recp_node
                                break
                
                # find_target_object가 없거나 실패한 경우 직접 찾기
                if recp_obj_node is None:
                    receptacle_name_lower = receptacle_name.lower()
                    for recp_node in object_nodes:
                        recp_type = recp_node.get("objectType", "")
                        if recp_type.lower() == receptacle_name_lower:
                            recp_obj_node = recp_node
                            break
                        elif receptacle_name_lower in recp_type.lower():
                            recp_obj_node = recp_node
                            break
                
                if recp_obj_node:
                    recp_id = recp_obj_node.get("nodeId", "")
                    recp_type = recp_obj_node.get("objectType", "")
                    
                    # IN 엣지 추가
                    in_edge = {
                        "edgeType": "IN",
                        "source": obj_id,
                        "target": recp_id,
                        "sourceType": "Object",
                        "targetType": "Object",
                        "sourceObjectType": obj_type,
                        "targetObjectType": recp_type
                    }
                    if in_edge not in edges:
                        edges.append(in_edge)
                    
                    # Object 노드 업데이트
                    target_obj_node["isPickedUp"] = False
                    if "parentReceptacles" not in target_obj_node:
                        target_obj_node["parentReceptacles"] = []
                    if recp_id not in target_obj_node["parentReceptacles"]:
                        target_obj_node["parentReceptacles"].append(recp_id)
                    
                    # Agent 노드 업데이트
                    agent_node["isHolding"] = False
                    agent_node["heldObjectId"] = None
                    
                    logger.info(f"  → PutObject: '{object_name}' (nodeId: {obj_id}) → '{receptacle_name}' (nodeId: {recp_id}) 업데이트 완료")
                else:
                    logger.warning(f"  → PutObject: 수용체 '{receptacle_name}'를 Scene Graph에서 찾을 수 없음")
            else:
                logger.warning(f"  → PutObject: 객체 '{object_name}'를 Scene Graph에서 찾을 수 없음")
    
    elif action_type == "OpenObject":
        # Object 노드의 isOpen 업데이트
        if object_name:
            # action에 nodeId가 있으면 정확한 nodeId로 찾기 (복구 액션에서 전달된 경우)
            target_node_id = action.get("nodeId")
            
            if target_node_id:
                # nodeId로 정확하게 찾기
                found = False
                for obj_node in object_nodes:
                    if obj_node.get("nodeId") == target_node_id:
                        obj_node["isOpen"] = True
                        obj_node["openness"] = 1.0
                        logger.info(f"  → OpenObject: nodeId '{target_node_id}'의 isOpen=True, openness=1.0으로 업데이트됨")
                        found = True
                        break
                if not found:
                    logger.warning(f"  → OpenObject: nodeId '{target_node_id}'를 object_nodes에서 찾을 수 없음")
            else:
                # object_name이 nodeId 형식인지 확인 (예: "Drawer|-01.56|+00.84|-00.20")
                if "|" in object_name and len(object_name.split("|")) >= 4:
                    # nodeId 형식이면 직접 사용
                    target_node_id = object_name
                    found = False
                    for obj_node in object_nodes:
                        if obj_node.get("nodeId") == target_node_id:
                            obj_node["isOpen"] = True
                            obj_node["openness"] = 1.0
                            logger.info(f"  → OpenObject: nodeId '{target_node_id}'의 isOpen=True, openness=1.0으로 업데이트됨")
                            found = True
                            break
                    if not found:
                        logger.warning(f"  → OpenObject: nodeId '{target_node_id}'를 object_nodes에서 찾을 수 없음")
                else:
                    # nodeId가 없으면 find_target_object를 사용하여 정확하게 receptacle 찾기
                    if find_target_object:
                        matched_objects = find_target_object(scene_graph, object_name)
                        if matched_objects:
                            # 첫 번째 매칭된 객체의 nodeId로 object_nodes에서 찾아서 업데이트
                            matched_obj = matched_objects[0]
                            matched_node_id = matched_obj.get("nodeId")
                            
                            # object_nodes에서 해당 nodeId를 가진 객체 찾기
                            found = False
                            for obj_node in object_nodes:
                                if obj_node.get("nodeId") == matched_node_id:
                                    obj_node["isOpen"] = True
                                    obj_node["openness"] = 1.0
                                    logger.info(f"  → OpenObject: '{object_name}' (nodeId: {matched_node_id})의 isOpen=True, openness=1.0으로 업데이트됨")
                                    found = True
                                    break
                            if not found:
                                logger.warning(f"  → OpenObject: '{object_name}' (nodeId: {matched_node_id})를 object_nodes에서 찾을 수 없음")
                        else:
                            logger.warning(f"  → OpenObject: '{object_name}'를 Scene Graph에서 찾을 수 없음")
                    else:
                        # find_target_object가 없는 경우 정확한 매칭 우선으로 직접 찾기
                        object_name_lower = object_name.lower()
                        exact_match = None
                        partial_match = None
                        
                        for obj_node in object_nodes:
                            obj_type = obj_node.get("objectType", "")
                            obj_id = obj_node.get("nodeId", "")
                            obj_type_lower = obj_type.lower()
                            obj_id_lower = obj_id.lower()
                            
                            # "Knife"를 찾을 때 "ButterKnife"는 무조건 제외
                            if object_name_lower == "knife":
                                if "butter" in obj_type_lower or "butter" in obj_id_lower:
                                    continue
                            
                            # 정확한 매칭 우선
                            if obj_type_lower == object_name_lower or obj_id_lower == object_name_lower:
                                exact_match = obj_node
                                break
                            # 정확한 매칭이 없으면 부분 매칭
                            elif object_name_lower in obj_type_lower or object_name_lower in obj_id_lower:
                                # 추가 확인: "knife"를 찾을 때 "butterknife"는 제외
                                if object_name_lower == "knife" and ("butter" in obj_type_lower or "butter" in obj_id_lower):
                                    continue
                                if partial_match is None:
                                    partial_match = obj_node
                        
                        # "Knife"를 찾을 때는 정확한 매칭만 사용
                        target_obj = exact_match if (object_name_lower == "knife" and exact_match) else (exact_match if exact_match else partial_match)
                        
                        if target_obj:
                            target_obj["isOpen"] = True
                            target_obj["openness"] = 1.0
                            logger.debug(f"  → OpenObject: '{object_name}' (nodeId: {target_obj.get('nodeId', 'N/A')})의 isOpen=True, openness=1.0으로 업데이트됨")
    
    elif action_type == "CloseObject":
        # Object 노드의 isOpen 업데이트
        if object_name:
            # find_target_object를 사용하여 정확하게 receptacle 찾기
            if find_target_object:
                matched_objects = find_target_object(scene_graph, object_name)
                if matched_objects:
                    # 첫 번째 매칭된 객체의 nodeId로 object_nodes에서 찾아서 업데이트
                    matched_obj = matched_objects[0]
                    matched_node_id = matched_obj.get("nodeId")
                    
                    # object_nodes에서 해당 nodeId를 가진 객체 찾기
                    for obj_node in object_nodes:
                        if obj_node.get("nodeId") == matched_node_id:
                            obj_node["isOpen"] = False
                            obj_node["openness"] = 0.0
                            logger.debug(f"  → CloseObject: '{object_name}' (nodeId: {matched_node_id})의 isOpen=False, openness=0.0으로 업데이트됨")
                            break
                    else:
                        logger.warning(f"  → CloseObject: '{object_name}' (nodeId: {matched_node_id})를 object_nodes에서 찾을 수 없음")
                else:
                    logger.warning(f"  → CloseObject: '{object_name}'를 Scene Graph에서 찾을 수 없음")
            else:
                # find_target_object가 없는 경우 정확한 매칭 우선으로 직접 찾기
                object_name_lower = object_name.lower()
                exact_match = None
                partial_match = None
                
                for obj_node in object_nodes:
                    obj_type = obj_node.get("objectType", "")
                    obj_id = obj_node.get("nodeId", "")
                    obj_type_lower = obj_type.lower()
                    obj_id_lower = obj_id.lower()
                    
                    # "Knife"를 찾을 때 "ButterKnife"는 무조건 제외
                    if object_name_lower == "knife":
                        if "butter" in obj_type_lower or "butter" in obj_id_lower:
                            continue
                    
                    # 정확한 매칭 우선
                    if obj_type_lower == object_name_lower or obj_id_lower == object_name_lower:
                        exact_match = obj_node
                        break
                    # 정확한 매칭이 없으면 부분 매칭
                    elif object_name_lower in obj_type_lower or object_name_lower in obj_id_lower:
                        # 추가 확인: "knife"를 찾을 때 "butterknife"는 제외
                        if object_name_lower == "knife" and ("butter" in obj_type_lower or "butter" in obj_id_lower):
                            continue
                        if partial_match is None:
                            partial_match = obj_node
                
                # "Knife"를 찾을 때는 정확한 매칭만 사용
                target_obj = exact_match if (object_name_lower == "knife" and exact_match) else (exact_match if exact_match else partial_match)
                
                if target_obj:
                    target_obj["isOpen"] = False
                    target_obj["openness"] = 0.0
                    logger.debug(f"  → CloseObject: '{object_name}' (nodeId: {target_obj.get('nodeId', 'N/A')})의 isOpen=False, openness=0.0으로 업데이트됨")
    
    elif action_type == "ToggleObjectOn":
        # Object 노드의 isToggled 업데이트
        if object_name:
            # find_target_object를 사용하여 정확한 객체 찾기
            target_obj_node = None
            if find_target_object:
                matched_objects = find_target_object(scene_graph, object_name)
                if matched_objects:
                    matched_obj = matched_objects[0]
                    matched_node_id = matched_obj.get("nodeId")
                    for obj_node in object_nodes:
                        if obj_node.get("nodeId") == matched_node_id:
                            target_obj_node = obj_node
                            break
            
            # find_target_object가 없거나 실패한 경우, 정확한 매칭 우선으로 직접 찾기
            if target_obj_node is None:
                object_name_lower = object_name.lower()
                exact_match = None
                partial_match = None
                
                for obj_node in object_nodes:
                    obj_type = obj_node.get("objectType", "")
                    obj_id = obj_node.get("nodeId", "")
                    obj_type_lower = obj_type.lower()
                    obj_id_lower = obj_id.lower()
                    
                    # "Knife"를 찾을 때 "ButterKnife"는 무조건 제외
                    if object_name_lower == "knife":
                        if "butter" in obj_type_lower or "butter" in obj_id_lower:
                            continue
                    
                    # 정확한 매칭 우선
                    if obj_type_lower == object_name_lower or obj_id_lower == object_name_lower:
                        exact_match = obj_node
                        break
                    # 정확한 매칭이 없으면 부분 매칭
                    elif object_name_lower in obj_type_lower or object_name_lower in obj_id_lower:
                        # 추가 확인: "knife"를 찾을 때 "butterknife"는 제외
                        if object_name_lower == "knife" and ("butter" in obj_type_lower or "butter" in obj_id_lower):
                            continue
                        if partial_match is None:
                            partial_match = obj_node
                
                # "Knife"를 찾을 때는 정확한 매칭만 사용
                target_obj_node = exact_match if (object_name_lower == "knife" and exact_match) else (exact_match if exact_match else partial_match)
            
            if target_obj_node:
                target_obj_node["isToggled"] = True
                logger.info(f"  → ToggleObjectOn: '{object_name}' (nodeId: {target_obj_node.get('nodeId', 'N/A')})의 isToggled=True로 업데이트됨")
            else:
                logger.warning(f"  → ToggleObjectOn: '{object_name}'를 Scene Graph에서 찾을 수 없음")
    
    elif action_type == "ToggleObjectOff":
        # Object 노드의 isToggled 업데이트
        if object_name:
            # find_target_object를 사용하여 정확한 객체 찾기
            target_obj_node = None
            if find_target_object:
                matched_objects = find_target_object(scene_graph, object_name)
                if matched_objects:
                    matched_obj = matched_objects[0]
                    matched_node_id = matched_obj.get("nodeId")
                    for obj_node in object_nodes:
                        if obj_node.get("nodeId") == matched_node_id:
                            target_obj_node = obj_node
                            break
            
            # find_target_object가 없거나 실패한 경우, 정확한 매칭 우선으로 직접 찾기
            if target_obj_node is None:
                object_name_lower = object_name.lower()
                exact_match = None
                partial_match = None
                
                for obj_node in object_nodes:
                    obj_type = obj_node.get("objectType", "")
                    obj_id = obj_node.get("nodeId", "")
                    obj_type_lower = obj_type.lower()
                    obj_id_lower = obj_id.lower()
                    
                    # "Knife"를 찾을 때 "ButterKnife"는 무조건 제외
                    if object_name_lower == "knife":
                        if "butter" in obj_type_lower or "butter" in obj_id_lower:
                            continue
                    
                    # 정확한 매칭 우선
                    if obj_type_lower == object_name_lower or obj_id_lower == object_name_lower:
                        exact_match = obj_node
                        break
                    # 정확한 매칭이 없으면 부분 매칭
                    elif object_name_lower in obj_type_lower or object_name_lower in obj_id_lower:
                        # 추가 확인: "knife"를 찾을 때 "butterknife"는 제외
                        if object_name_lower == "knife" and ("butter" in obj_type_lower or "butter" in obj_id_lower):
                            continue
                        if partial_match is None:
                            partial_match = obj_node
                
                # "Knife"를 찾을 때는 정확한 매칭만 사용
                target_obj_node = exact_match if (object_name_lower == "knife" and exact_match) else (exact_match if exact_match else partial_match)
            
            if target_obj_node:
                target_obj_node["isToggled"] = False
                logger.info(f"  → ToggleObjectOff: '{object_name}' (nodeId: {target_obj_node.get('nodeId', 'N/A')})의 isToggled=False로 업데이트됨")
            else:
                logger.warning(f"  → ToggleObjectOff: '{object_name}'를 Scene Graph에서 찾을 수 없음")
    
    elif action_type == "SliceObject":
        # Object 노드의 isSliced 업데이트
        if object_name:
            # find_target_object를 사용하여 정확한 객체 찾기
            target_obj_node = None
            if find_target_object:
                matched_objects = find_target_object(scene_graph, object_name)
                if matched_objects:
                    matched_obj = matched_objects[0]
                    matched_node_id = matched_obj.get("nodeId")
                    for obj_node in object_nodes:
                        if obj_node.get("nodeId") == matched_node_id:
                            target_obj_node = obj_node
                            break
            
            # find_target_object가 없거나 실패한 경우, 정확한 매칭 우선으로 직접 찾기
            if target_obj_node is None:
                object_name_lower = object_name.lower()
                exact_match = None
                partial_match = None
                
                for obj_node in object_nodes:
                    obj_type = obj_node.get("objectType", "")
                    obj_id = obj_node.get("nodeId", "")
                    obj_type_lower = obj_type.lower()
                    obj_id_lower = obj_id.lower()
                    
                    # "Knife"를 찾을 때 "ButterKnife"는 무조건 제외
                    if object_name_lower == "knife":
                        if "butter" in obj_type_lower or "butter" in obj_id_lower:
                            continue
                    
                    # 정확한 매칭 우선
                    if obj_type_lower == object_name_lower or obj_id_lower == object_name_lower:
                        exact_match = obj_node
                        break
                    # 정확한 매칭이 없으면 부분 매칭
                    elif object_name_lower in obj_type_lower or object_name_lower in obj_id_lower:
                        # 추가 확인: "knife"를 찾을 때 "butterknife"는 제외
                        if object_name_lower == "knife" and ("butter" in obj_type_lower or "butter" in obj_id_lower):
                            continue
                        if partial_match is None:
                            partial_match = obj_node
                
                # "Knife"를 찾을 때는 정확한 매칭만 사용
                target_obj_node = exact_match if (object_name_lower == "knife" and exact_match) else (exact_match if exact_match else partial_match)
            
            if target_obj_node:
                target_obj_node["isSliced"] = True
                logger.info(f"  → SliceObject: '{object_name}' (nodeId: {target_obj_node.get('nodeId', 'N/A')})의 isSliced=True로 업데이트됨")
            else:
                logger.warning(f"  → SliceObject: '{object_name}'를 Scene Graph에서 찾을 수 없음")
    
    elif action_type == "BreakObject":
        # Object 노드의 isBroken 업데이트
        if object_name:
            # find_target_object를 사용하여 정확한 객체 찾기
            target_obj_node = None
            if find_target_object:
                matched_objects = find_target_object(scene_graph, object_name)
                if matched_objects:
                    matched_obj = matched_objects[0]
                    matched_node_id = matched_obj.get("nodeId")
                    for obj_node in object_nodes:
                        if obj_node.get("nodeId") == matched_node_id:
                            target_obj_node = obj_node
                            break
            
            # find_target_object가 없거나 실패한 경우, 정확한 매칭 우선으로 직접 찾기
            if target_obj_node is None:
                object_name_lower = object_name.lower()
                exact_match = None
                partial_match = None
                
                for obj_node in object_nodes:
                    obj_type = obj_node.get("objectType", "")
                    obj_id = obj_node.get("nodeId", "")
                    obj_type_lower = obj_type.lower()
                    obj_id_lower = obj_id.lower()
                    
                    # "Knife"를 찾을 때 "ButterKnife"는 무조건 제외
                    if object_name_lower == "knife":
                        if "butter" in obj_type_lower or "butter" in obj_id_lower:
                            continue
                    
                    # 정확한 매칭 우선
                    if obj_type_lower == object_name_lower or obj_id_lower == object_name_lower:
                        exact_match = obj_node
                        break
                    # 정확한 매칭이 없으면 부분 매칭
                    elif object_name_lower in obj_type_lower or object_name_lower in obj_id_lower:
                        # 추가 확인: "knife"를 찾을 때 "butterknife"는 제외
                        if object_name_lower == "knife" and ("butter" in obj_type_lower or "butter" in obj_id_lower):
                            continue
                        if partial_match is None:
                            partial_match = obj_node
                
                # "Knife"를 찾을 때는 정확한 매칭만 사용
                target_obj_node = exact_match if (object_name_lower == "knife" and exact_match) else (exact_match if exact_match else partial_match)
            
            if target_obj_node:
                target_obj_node["isBroken"] = True
                logger.info(f"  → BreakObject: '{object_name}' (nodeId: {target_obj_node.get('nodeId', 'N/A')})의 isBroken=True로 업데이트됨")
            else:
                logger.warning(f"  → BreakObject: '{object_name}'를 Scene Graph에서 찾을 수 없음")
    
    # 업데이트된 Scene Graph 반환
    scene_graph["nodes"]["agent"] = agent_node
    scene_graph["nodes"]["objects"] = object_nodes
    scene_graph["edges"] = edges
    
    # JSON 파일에 저장
    if scene_graph_path:
        save_scene_graph_to_file(scene_graph, scene_graph_path)
        logger.info(f"  💾 Scene Graph JSON 파일 업데이트 완료: {scene_graph_path}")
    
    return scene_graph


def verify_task_completion_with_llm(
    client: OpenAI,
    model: str,
    task: str,
    final_plan: str
) -> Tuple[bool, List[str], str]:
    """
    LLM을 사용하여 자연어 목표와 최종 plan을 비교하여 모든 필요한 task가 수행되었는지 검증
    
    Args:
        client: LLM 클라이언트
        model: LLM 모델 이름
        task: 자연어 목표 (예: "put apple in fridge")
        final_plan: 최종 생성된 plan 코드
        
    Returns:
        (모든 task 완료 여부, 수행되지 않은 task 리스트, 검증 결과 설명)
    """
    prompt = f"""다음은 자연어로 주어진 목표와 최종적으로 생성된 plan입니다.

자연어 목표:
{task}

최종 생성된 Plan:
{final_plan}

위 정보를 바탕으로 다음을 수행해주세요:
1. 자연어 목표에서 요구하는 모든 task를 추출하세요.
2. 최종 plan에서 실제로 수행되는 task를 추출하세요.
3. 자연어 목표에서 요구하지만 plan에서 수행되지 않은 task를 찾아서 리스트로 정리하세요.

응답 형식 (JSON):
{{
    "allTasksCompleted": true/false,
    "missingTasks": ["task1", "task2", ...],
    "reasoning": "왜 이 task들이 누락되었는지 설명, 어떤 액션이 어디 사이에 추가적으로 필요한지 설명"
}}

만약 모든 task가 완료되었다면:
{{
    "allTasksCompleted": true,
    "missingTasks": [],
    "reasoning": "모든 필요한 task가 plan에 포함되어 있습니다."
}}

JSON 형식으로만 응답하세요:"""

    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that verifies if a plan completes all required tasks from a natural language goal."},
            {"role": "user", "content": prompt}
        ]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # JSON 추출
        import json
        # JSON 부분만 추출 (```json ... ``` 또는 {...} 형식)
        if "```json" in result_text:
            start_idx = result_text.find("```json") + 7
            end_idx = result_text.find("```", start_idx)
            json_text = result_text[start_idx:end_idx].strip()
        elif "```" in result_text:
            start_idx = result_text.find("```") + 3
            end_idx = result_text.find("```", start_idx)
            json_text = result_text[start_idx:end_idx].strip()
        else:
            # JSON 객체 찾기
            start_idx = result_text.find("{")
            end_idx = result_text.rfind("}") + 1
            json_text = result_text[start_idx:end_idx]
        
        result = json.loads(json_text)
        
        all_completed = result.get("allTasksCompleted", False)
        missing_tasks = result.get("missingTasks", [])
        reasoning = result.get("reasoning", "")
        
        return all_completed, missing_tasks, reasoning
        
    except Exception as e:
        logger.warning(f"LLM task 완료 검증 실패: {e}")
        return False, [], f"검증 실패: {e}"


def generate_failure_comment_with_llm(
    client: OpenAI,
    model: str,
    action: Dict[str, Any],
    failed_guards: List[str],
    failure_reason: str
) -> str:
    """
    LLM을 사용하여 물리적 검증 실패 이유를 분석하고 주석 생성
    
    Args:
        client: LLM 클라이언트
        model: LLM 모델 이름
        action: 실패한 액션
        failed_guards: 실패한 가드 리스트
        failure_reason: 실패 이유
        
    Returns:
        생성된 주석 문자열
    """
    action_type = action.get("type", "")
    args = action.get("args", {})
    object_name = args.get("o", "")
    receptacle_name = args.get("r", "")
    action_line = action.get("line", "")
    
    # LLM 프롬프트 구성 (예시만 포함, 사용자가 수정 가능)
    prompt = f"""다음은 물리적 검증 실패한 액션에 대한 정보입니다.

액션: {action_line}
액션 타입: {action_type}
타겟 객체: {object_name}
수용체: {receptacle_name if receptacle_name else "N/A"}
실패한 가드: {', '.join(failed_guards)}
실패 이유: {failure_reason}

위 정보를 바탕으로 왜 이 액션이 물리적 검증을 통과하지 못했는지 간단히 설명하는 주석을 생성해주세요.
주석은 Python 코드 주석 형식(#)으로 작성하고, 한 줄 또는 여러 줄로 작성할 수 있습니다.

예시:
# 물리적 검증 실패: 객체 'Apple'이 Scene Graph에 존재하지 않음
# 물리적 검증 실패: 객체 'Fridge'까지의 가장 가까운 이동 가능 위치가 1.3m를 초과함 (거리: 2.5m)
# 물리적 검증 실패: Agent가 이미 다른 객체를 들고 있음 (heldObjectId: 'object')
# 물리적 검증 실패: 객체 'Egg'가 닫힌 수용체 'Fridge' 안에 있어 접근 불가
# 물리적 검증 실패: 객체 'Mug'가 pickupable이 아님
# 물리적 검증 실패: Agent 손과 객체 'Bread'까지의 거리가 1.39m를 초과함 (거리: 2.1m)
# 물리적 검증 실패: 수용체 'Cabinet'가 열려있지 않음 (isOpen: False)
# 물리적 검증 실패: 객체 'CounterTop'가 receptacle 타입이 아님

"""

    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates Python code comments explaining why physical verification failed."},
            {"role": "user", "content": prompt}
        ]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=200
        )
        
        comment = response.choices[0].message.content.strip()
        
        # LLM 응답에서 불필요한 설명 텍스트 제거
        # "Here is the Python code comment..." 같은 부분 제거
        lines = comment.split("\n")
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # 불필요한 설명 텍스트 건너뛰기
            if any(skip_phrase in line.lower() for skip_phrase in [
                "here is", "this comment", "explaining why", "python code comment"
            ]):
                continue
            # 주석이 #로 시작하지 않으면 추가
            if line and not line.startswith("#"):
                line = "# " + line
            if line:
                cleaned_lines.append(line)
        
        comment = "\n".join(cleaned_lines) if cleaned_lines else f"# 물리적 검증 실패: {failure_reason}"
        
        # 주석이 비어있으면 기본 주석 반환
        if not comment or comment.strip() == "#":
            comment = f"# 물리적 검증 실패: {failure_reason}"
        
        return comment
    except Exception as e:
        logger.warning(f"LLM 주석 생성 실패: {e}")
        # 기본 주석 반환
        return f"# 물리적 검증 실패: {failure_reason}"


def generate_final_plan_with_physical_verification(
    task: str,
    initial_program: str,
    scene_graph: Dict[str, Any],
    controller: Optional[Any] = None,
    client: Optional[OpenAI] = None,
    model: str = "llama3",
    scene_graph_path: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    논리적 검증된 plan을 물리적 검증하여 최종 plan 생성
    
    Args:
        task: 작업 설명
        initial_program: 논리적 검증 완료된 초기 프로그램
        scene_graph: Scene Graph 딕셔너리
        controller: AI2-THOR Controller (NavMesh 검증용, 선택사항)
        
    Returns:
        (최종 plan 코드, 검증 결과 딕셔너리)
    """
    logger.info("=" * 80)
    logger.info(f"물리적 검증 시작: '{task}'")
    logger.info("=" * 80)
    
    # Step 1: 프로그램 파싱
    logger.info("\n[Step 1] 프로그램 파싱 중...")
    plan_actions = parse_program_to_actions(initial_program)
    logger.info(f"✓ {len(plan_actions)}개의 액션 파싱 완료")
    
    # Step 2: 각 액션에 대해 물리적 검증 수행
    logger.info("\n[Step 2] 물리적 검증: 각 액션 검증 중...")
    verified_actions = []
    failed_actions = []
    
    i = 0
    verification_count = 0  # 검증 횟수 추적
    max_verifications = 20  # 최대 검증 횟수
    
    while i < len(plan_actions):
        # 최대 검증 횟수 확인
        if verification_count >= max_verifications:
            logger.warning(f"  ⚠️  최대 검증 횟수({max_verifications}회)에 도달했습니다. 검증을 중단합니다.")
            # 남은 액션들을 실패한 액션으로 추가
            for remaining_action in plan_actions[i:]:
                failed_actions.append({
                    "action": remaining_action,
                    "reason": f"최대 검증 횟수({max_verifications}회) 초과로 검증 중단",
                    "failed_guards": [],
                    "recovery_actions": []
                })
            break
        
        verification_count += 1
        action = plan_actions[i]
        logger.info(f"\n[액션 {i+1}/{len(plan_actions)}] {action.get('line', '')}")
        
        # Scene Graph 기반 물리적 검증
        passed, reason, failed_guards, recovery_actions = verify_action_with_scene_graph(
            action, scene_graph, controller
        )
        
        # EXISTS 가드 실패 시 (객체가 존재하지 않음) 검증 종료
        if reason.startswith("OBJECT_NOT_EXISTS:"):
            error_message = reason.replace("OBJECT_NOT_EXISTS: ", "")
            logger.error(f"\n{'='*80}")
            logger.error(f"❌ 검증 중단: 존재하지 않는 객체를 사용하여 계획을 생성할 수 없음")
            logger.error(f"   {error_message}")
            logger.error(f"{'='*80}")
            print(f"\n{'='*80}")
            print(f"❌ 검증 중단: 존재하지 않는 객체를 사용하여 계획을 생성할 수 없음")
            print(f"   {error_message}")
            print(f"{'='*80}")
            # 검증 종료 - 빈 plan 반환
            return "", {
                "total_actions": len(plan_actions),
                "passed_actions": len(verified_actions),
                "failed_actions": 1,
                "failed_actions_list": [{
                    "action": action,
                    "reason": error_message,
                    "failed_guards": failed_guards,
                    "recovery_actions": []
                }],
                "updated_scene_graph": scene_graph,
                "error": "OBJECT_NOT_EXISTS"
            }
        
        # REACHABLE 가드 실패 시 처리
        # Proximity와 REACHABLE이 모두 실패했을 때는 Proximity 복구 후 REACHABLE 재검사
        # REACHABLE만 실패했을 때는 검증 종료
        has_proximity_failure = any("Proximity" in guard for guard in failed_guards)
        has_reachable_failure = any("REACHABLE" in guard for guard in failed_guards)
        
        if has_reachable_failure and not has_proximity_failure:
            # REACHABLE만 실패한 경우: 검증 종료
            action_type = action.get("type", "")
            action_args = action.get("args", {})
            object_name = action_args.get("o") or action_args.get("r") or "알 수 없음"
            error_message = f"액션 '{action_type}'의 대상 객체 '{object_name}'가 agent의 손이 닿지 않는 거리에 있어 계획을 생성할 수 없음"
            logger.error(f"\n{'='*80}")
            logger.error(f"❌ 검증 중단: REACHABLE 가드 위반")
            logger.error(f"   {error_message}")
            logger.error(f"   실패한 가드: {', '.join([g for g in failed_guards if 'REACHABLE' in g])}")
            logger.error(f"{'='*80}")
            print(f"\n{'='*80}")
            print(f"❌ 검증 중단: REACHABLE 가드 위반")
            print(f"   {error_message}")
            print(f"   실패한 가드: {', '.join([g for g in failed_guards if 'REACHABLE' in g])}")
            print(f"{'='*80}")
            # 검증 종료 - 빈 plan 반환
            return "", {
                "total_actions": len(plan_actions),
                "passed_actions": len(verified_actions),
                "failed_actions": 1,
                "failed_actions_list": [{
                    "action": action,
                    "reason": error_message,
                    "failed_guards": [g for g in failed_guards if 'REACHABLE' in g],
                    "recovery_actions": []
                }],
                "updated_scene_graph": scene_graph,
                "error": "REACHABLE_VIOLATION"
            }
        
        if passed:
            verified_actions.append(action)
            # Scene Graph 업데이트 및 JSON 파일 저장
            scene_graph = update_scene_graph_after_action(
                scene_graph, action, verification_passed=True, scene_graph_path=scene_graph_path, controller=controller
            )
            logger.info(f"  ✓ 검증 통과: {reason}")
            
            # 액션 실행 후 상태 요약 출력
            action_type = action.get("type", "")
            action_args = action.get("args", {})
            object_name = action_args.get("o")
            receptacle_name = action_args.get("r")
            print_action_summary(action, scene_graph, object_name, receptacle_name)
            
            # PickupObject가 통과한 경우, 부모 수용체가 열려있으면 CloseObject 추가
            if action_type == "PickupObject" and object_name:
                # 목표 객체 찾기
                target_obj = None
                if scene_graph:
                    object_nodes = scene_graph.get("nodes", {}).get("objects", [])
                    for obj_node in object_nodes:
                        obj_type = obj_node.get("objectType", "")
                        obj_id = obj_node.get("nodeId", "")
                        if object_name.lower() in obj_type.lower() or object_name.lower() in obj_id.lower():
                            target_obj = obj_node
                            break
                
                # 부모 수용체 확인
                if target_obj:
                    parent_receptacles = target_obj.get("parentReceptacles", [])
                    for recp_id in parent_receptacles:
                        # 수용체 노드 찾기
                        recp_node = None
                        if scene_graph:
                            object_nodes = scene_graph.get("nodes", {}).get("objects", [])
                            for obj_node in object_nodes:
                                if obj_node.get("nodeId") == recp_id:
                                    recp_node = obj_node
                                    break
                        
                        # 수용체가 openable이고 열려있으면 CloseObject 추가
                        if recp_node and recp_node.get("openable", False) and recp_node.get("isOpen", False):
                            close_action = {
                                "type": "CloseObject",
                                "args": {"o": recp_id},  # nodeId 전체 사용
                                "line": f"CloseObject('{recp_id}')",  # nodeId 전체 포함
                                "nodeId": recp_id,  # nodeId를 별도로 저장
                                "reason": f"부모 수용체 '{recp_id}' 닫기 (PickupObject 후)",
                                "is_original": False,
                                "is_recovery": True,
                                "failed_guards": [],
                                "recovery_reason": f"PickupObject 후 부모 수용체 '{recp_id}' 자동 닫기"
                            }
                            # 다음 액션 위치에 CloseObject 삽입
                            plan_actions.insert(i + 1, close_action)
                            logger.info(f"  → CloseObject 추가: CloseObject('{recp_id}')")
                            break  # 첫 번째 열린 수용체만 처리
            
            # 다음 액션 검증을 위해 업데이트된 JSON 파일 다시 로드
            if scene_graph_path:
                scene_graph = load_scene_graph(scene_graph_path)
                logger.debug(f"  🔄 업데이트된 Scene Graph 다시 로드 완료")
            
            i += 1  # 다음 액션으로 진행
        else:
            # PickupObject이고 ¬HOLDS(agent, *) 가드가 실패한 경우 (이미 손에 물체가 있음)
            # 해당 액션을 건너뛰고 다음 액션으로 진행
            action_type = action.get("type", "")
            if action_type == "PickupObject" and "¬HOLDS(agent, *)" in failed_guards:
                logger.info(f"  ⏭️  PickupObject 건너뛰기: 이미 손에 물체가 있음")
                logger.info(f"     → 다음 액션으로 진행")
                i += 1  # 다음 액션으로 진행
                continue
            
            # OpenObject이고 ¬OPENED(object) 가드가 실패한 경우 (이미 열려있음)
            # 해당 액션을 건너뛰고 다음 액션으로 진행
            if action_type == "OpenObject" and "¬OPENED(object)" in failed_guards:
                logger.info(f"  ⏭️  OpenObject 건너뛰기: 객체 '{object_name}'가 이미 열려있음")
                logger.info(f"     → 다음 액션으로 진행")
                i += 1  # 다음 액션으로 진행
                continue
            
            # OpenObject이고 openable(object) 가드가 실패한 경우 (openable하지 않은 객체)
            # 해당 액션을 삭제하고 다음 액션으로 진행
            if action_type == "OpenObject" and "openable(object)" in failed_guards:
                logger.warning(f"  ⚠️  OpenObject 삭제: 객체 '{object_name}'가 openable하지 않음")
                logger.info(f"     → 액션 삭제 후 다음 액션으로 진행")
                plan_actions.pop(i)  # 현재 액션 삭제
                # i는 그대로 유지 (다음 액션이 현재 위치로 이동)
                continue
            
            # CloseObject이고 openable(object) 가드가 실패한 경우 (openable하지 않은 객체)
            # 해당 액션을 삭제하고 다음 액션으로 진행
            if action_type == "CloseObject" and "openable(object)" in failed_guards:
                logger.warning(f"  ⚠️  CloseObject 삭제: 객체 '{object_name}'가 openable하지 않음")
                logger.info(f"     → 액션 삭제 후 다음 액션으로 진행")
                plan_actions.pop(i)  # 현재 액션 삭제
                # i는 그대로 유지 (다음 액션이 현재 위치로 이동)
                continue
            
            # 복구 액션이 있으면 원래 액션 이전에 삽입하고 다시 검증
            if recovery_actions:
                # Proximity와 REACHABLE이 모두 실패했을 때: Proximity 복구 후 REACHABLE 재검사
                has_proximity_recovery = any(ra.get("type") == "GoToObject" and "Proximity" in ra.get("failed_guards", []) for ra in recovery_actions)
                has_reachable_failure = any("REACHABLE" in guard for guard in failed_guards)
                
                if has_proximity_recovery and has_reachable_failure:
                    logger.info(f"  → Proximity와 REACHABLE이 모두 실패: Proximity 복구 후 REACHABLE 재검사")
                    
                    # Proximity 복구 액션만 먼저 실행
                    proximity_recovery = None
                    other_recoveries = []
                    for ra in recovery_actions:
                        if ra.get("type") == "GoToObject" and "Proximity" in ra.get("failed_guards", []):
                            proximity_recovery = ra
                        else:
                            other_recoveries.append(ra)
                    
                    if proximity_recovery:
                        # Proximity 복구 액션을 plan에 삽입하고 실행
                        plan_actions.insert(i, proximity_recovery)
                        logger.info(f"    → Proximity 복구 액션 삽입: {proximity_recovery.get('line', '')}")
                        
                        # Proximity 복구 액션 실행 (검증 통과로 처리)
                        verified_actions.append(proximity_recovery)
                        scene_graph = update_scene_graph_after_action(
                            scene_graph, proximity_recovery, verification_passed=True, scene_graph_path=scene_graph_path, controller=controller
                        )
                        logger.info(f"  ✓ Proximity 복구 액션 실행 완료")
                        
                        # Scene Graph 업데이트 후 다시 로드
                        if scene_graph_path:
                            scene_graph = load_scene_graph(scene_graph_path)
                            logger.debug(f"  🔄 업데이트된 Scene Graph 다시 로드 완료")
                        
                        # Agent 위치 업데이트 (복구 액션 실행 후)
                        agent_position = None
                        if scene_graph:
                            agent_node = scene_graph.get("nodes", {}).get("agent", {})
                            if agent_node:
                                agent_position = agent_node.get("position", {})
                        
                        # scene_context 재생성 (업데이트된 scene_graph 기반)
                        scene_context = get_relevant_scene_context(
                            scene_graph, action_type, object_name, receptacle_name
                        )
                        
                        # REACHABLE 가드만 다시 한 번 검사
                        logger.info(f"  → REACHABLE 가드 재검사 중...")
                        reachable_guards = [g for g in failed_guards if "REACHABLE" in g]
                        reachable_passed = True
                        reachable_failure_reason = ""
                        
                        for reachable_guard in reachable_guards:
                            passed, reason = verify_guard_with_scene_graph(
                                reachable_guard, scene_context, action_type, object_name, receptacle_name,
                                controller, scene_graph, agent_position
                            )
                            if not passed:
                                reachable_passed = False
                                reachable_failure_reason = reason
                                logger.warning(f"    ✗ REACHABLE 가드 재검사 실패: {reason}")
                                break
                            else:
                                logger.info(f"    ✓ REACHABLE 가드 재검사 통과: {reason}")
                        
                        if reachable_passed:
                            # REACHABLE 재검사 통과: 나머지 가드들도 다시 검사
                            logger.info(f"  → REACHABLE 재검사 통과, 나머지 가드들 재검사 중...")
                            remaining_failed_guards = [g for g in failed_guards if "REACHABLE" not in g and "Proximity" not in g]
                            
                            all_passed = True
                            final_reason = "모든 가드 통과"
                            
                            for guard in remaining_failed_guards:
                                passed, reason = verify_guard_with_scene_graph(
                                    guard, scene_context, action_type, object_name, receptacle_name,
                                    controller, scene_graph, agent_position
                                )
                                if not passed:
                                    all_passed = False
                                    final_reason = reason
                                    break
                            
                            if all_passed:
                                # 모든 가드 통과: 원래 액션 통과 처리
                                verified_actions.append(action)
                                scene_graph = update_scene_graph_after_action(
                                    scene_graph, action, verification_passed=True, scene_graph_path=scene_graph_path, controller=controller
                                )
                                logger.info(f"  ✓ 모든 가드 통과: {final_reason}")
                                
                                # 나머지 복구 액션들도 처리
                                if other_recoveries:
                                    for other_ra in other_recoveries:
                                        plan_actions.insert(i + 1, other_ra)
                                        logger.info(f"    → 기타 복구 액션 삽입: {other_ra.get('line', '')}")
                                
                                # Scene Graph 업데이트 후 다시 로드
                                if scene_graph_path:
                                    scene_graph = load_scene_graph(scene_graph_path)
                                    logger.debug(f"  🔄 업데이트된 Scene Graph 다시 로드 완료")
                                
                                i += 1  # 다음 액션으로 진행
                                continue
                            else:
                                # 나머지 가드 실패: 기존 로직으로 처리
                                logger.warning(f"  ✗ 나머지 가드 실패: {final_reason}")
                                failed_guards = remaining_failed_guards + [g for g in failed_guards if "REACHABLE" in g or "Proximity" in g]
                        else:
                            # REACHABLE 재검사 실패: 기존 로직으로 처리
                            logger.warning(f"  ✗ REACHABLE 재검사 실패: {reachable_failure_reason}")
                            # Proximity 복구 액션은 이미 실행했으므로 제거하지 않음
                            # 나머지 복구 액션들과 함께 처리
                            if other_recoveries:
                                recovery_actions = other_recoveries
                            else:
                                recovery_actions = []
                
                # 일반적인 복구 액션 처리 (Proximity+REACHABLE 케이스가 아닌 경우)
                if not (has_proximity_recovery and has_reachable_failure):
                    logger.info(f"  → 복구 액션 {len(recovery_actions)}개를 원래 액션 이전에 삽입")
                    
                    # 복구 액션 중 OpenObject가 있는지 확인하고, 있다면 원래 액션 이후에 CloseObject 추가
                    open_object_recoveries = []
                    for recovery_action in recovery_actions:
                        if recovery_action.get("type") == "OpenObject":
                            open_object_recoveries.append(recovery_action)
                    
                    # 복구 액션들을 plan_actions에 현재 위치에 삽입 (역순으로 삽입하여 순서 유지)
                    for j, recovery_action in enumerate(reversed(recovery_actions)):
                        plan_actions.insert(i, recovery_action)
                        logger.info(f"    → 복구 액션 삽입: {recovery_action.get('line', '')}")
                    
                    # OpenObject 복구 액션이 있으면 원래 액션 이후에 CloseObject 추가
                    if open_object_recoveries:
                        # 원래 액션의 위치는 i + len(recovery_actions) (복구 액션 삽입 후)
                        # 원래 액션 이후에 CloseObject 추가 (역순으로 삽입하여 순서 유지)
                        for open_object_recovery in reversed(open_object_recoveries):
                            open_obj_id = open_object_recovery.get("nodeId") or open_object_recovery.get("args", {}).get("o")
                            if open_obj_id:
                                close_action = {
                                    "type": "CloseObject",
                                    "args": {"o": open_obj_id},
                                    "line": f"CloseObject('{open_obj_id}')",
                                    "nodeId": open_obj_id,
                                    "reason": f"복구 액션으로 열린 수용체 '{open_obj_id}' 닫기",
                                    "is_original": False,
                                    "is_recovery": True,
                                    "failed_guards": [],
                                    "recovery_reason": f"복구 액션 OpenObject('{open_obj_id}') 이후 자동 닫기"
                                }
                                # 원래 액션 이후 위치에 CloseObject 삽입
                                plan_actions.insert(i + len(recovery_actions) + 1, close_action)
                                logger.info(f"  → CloseObject 추가 (복구 액션 OpenObject 이후): CloseObject('{open_obj_id}')")
                    
                    # 복구 액션부터 다시 검증하도록 인덱스 유지 (i는 그대로, continue로 루프 재시작)
                    logger.info(f"  → 복구 액션부터 다시 검증 시작")
                    continue  # while 루프의 시작으로 돌아가서 복구 액션부터 검증
            
            # 복구 액션이 없는 경우 실패로 처리하고 검증 중단
            failed_actions.append({
                "action": action,
                "reason": reason,
                "failed_guards": failed_guards,
                "recovery_actions": recovery_actions
            })
            logger.warning(f"  ✗ 검증 실패: {reason}")
            logger.warning(f"  ⚠️  복구 액션이 없으므로 검증을 중단합니다.")
            # 남은 액션들을 실패한 액션으로 추가
            for remaining_action in plan_actions[i+1:]:
                failed_actions.append({
                    "action": remaining_action,
                    "reason": "이전 액션 검증 실패로 인한 중단",
                    "failed_guards": [],
                    "recovery_actions": []
                })
            break  # 검증 중단
    
    logger.info(f"\n✓ 물리적 검증 완료: {len(verified_actions)}/{len(plan_actions)} 액션 통과")
    
    # Step 3: 실패한 액션들에 대해 LLM으로 주석 생성
    logger.info("\n[Step 3] 실패한 액션들에 대한 주석 생성 중...")
    failed_actions_with_comments = []
    if client and failed_actions:
        for failed_action_info in failed_actions:
            action = failed_action_info["action"]
            reason = failed_action_info["reason"]
            failed_guards = failed_action_info.get("failed_guards", [])
            
            comment = generate_failure_comment_with_llm(
                client=client,
                model=model,
                action=action,
                failed_guards=failed_guards,
                failure_reason=reason
            )
            
            failed_actions_with_comments.append({
                "action": action,
                "comment": comment,
                "reason": reason,
                "failed_guards": failed_guards
            })
            logger.info(f"  ✓ 주석 생성 완료: {action.get('line', '')}")
    
    # Step 4: 최종 Plan 생성 (통과한 액션 + 실패한 액션(주석 포함))
    logger.info("\n[Step 4] 최종 Plan 생성 중...")
    action_lines = []
    step_counter = 1
    
    # 통과한 액션들 추가
    for action in verified_actions:
        line = action.get("line", "")
        if line:
            action_lines.append(f"\t# Step {step_counter}")
            
            # 액션 출처 주석 추가
            is_original = action.get("is_original", True)
            is_recovery = action.get("is_recovery", False)
            
            if is_recovery:
                failed_guards = action.get("failed_guards", [])
                recovery_reason = action.get("recovery_reason", action.get("reason", ""))
                action_lines.append(f"\t# [시스템 생성] 복구 액션")
                if failed_guards:
                    guards_str = ", ".join(failed_guards)
                    action_lines.append(f"\t# 위반한 가드: {guards_str}")
                if recovery_reason:
                    action_lines.append(f"\t# 이유: {recovery_reason}")
            elif is_original:
                action_lines.append(f"\t# [LLM 생성 - 논리적 검증] 원본 액션")
            
            # GoToObject 검증 통과 시 이동할 좌표 주석 추가
            action_type = action.get("type", "")
            if action_type == "GoToObject" and action.get("target_position"):
                target_pos = action.get("target_position")
                action_lines.append(f"\t# 이동할 좌표: ({target_pos.get('x', 0):.3f}, {target_pos.get('y', 0):.3f}, {target_pos.get('z', 0):.3f})")
            
            action_lines.append(f"\t{line}")
            step_counter += 1
    
    # 실패한 액션들 추가 (주석 포함 및 주석처리)
    for failed_info in failed_actions_with_comments:
        action = failed_info["action"]
        comment = failed_info["comment"]
        line = action.get("line", "")
        
        if line:
            action_lines.append(f"\t# Step {step_counter}")
            action_lines.append(f"\t# [LLM 생성 - 논리적 검증] 원본 액션 - 검증 실패")
            action_lines.append(f"\t# [LLM 주석] {comment}")
            # 실패한 액션을 주석처리
            action_lines.append(f"\t# {line}")
            step_counter += 1
    
    actions_code = "\n".join(action_lines)
    
    # 함수 이름 생성
    task_name = task.lower().replace(" ", "_").replace("'", "").replace('"', "")
    task_name = "".join(c if c.isalnum() or c == "_" else "_" for c in task_name)
    if not task_name:
        task_name = "execute_task"
    
    final_plan = f"""def {task_name}():
\t# Task: {task}
\t# Generated plan with logical and physical verification
\t# 
\t# [LLM 생성 - 논리적 검증] = LLM이 논리적 검증 단계에서 생성한 원본 액션
\t# [LLM 생성 - 누락 Task 보완] = LLM이 물리적 검증 후 누락된 task를 위해 추가로 생성한 액션
\t# [시스템 생성] = 시스템이 물리적 검증 단계에서 자동으로 생성한 복구 액션
\t# [LLM 주석] = LLM이 실패한 액션에 대해 생성한 설명 주석
{actions_code}
"""
    
    logger.info("✓ 최종 Plan 생성 완료")
    
    return final_plan, {
        "total_actions": len(plan_actions),
        "passed_actions": len(verified_actions),
        "failed_actions": len(failed_actions),
        "failed_actions_list": failed_actions,
        "updated_scene_graph": scene_graph
    }


def main():
    """
    메인 함수: 명령줄 인자를 파싱하고, 작업 목록을 로드하며, 
    LLM을 사용하여 프로그램을 생성하고 저장합니다.
    """
    # 명령줄 인자 파서 생성
    parser = argparse.ArgumentParser(
        description="ProgPrompt planner for AI2-THOR FloorPlan1 (llama3@Ollama)."  # 프로그램 설명
    )
    
    # 모델 관련 인자
    parser.add_argument("--model", type=str, default="llama3", help="Ollama 모델 이름.")
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434/v1",  # 기본 Ollama 엔드포인트
        help="Ollama OpenAI 호환 엔드포인트 URL.",
    )
    
    # 작업 관련 인자
    parser.add_argument(
        "--task-file",
        type=str,
        default=None,
        help="작업 목록이 담긴 JSON/JSONL 파일 경로 (선택사항).",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",  # 0개 이상의 인자 허용
        default=[],
        help="명령줄에서 직접 제공하는 인라인 작업 목록.",
    )
    
    # 프롬프트 관련 인자
    parser.add_argument(
        "--max-examples",
        type=int,
        default=3,
        help="프롬프트에 포함할 예제 프로그램 개수.",
    )
    
    # LLM 생성 파라미터
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,  # 결정적 생성 (일관된 결과)
        help="LLM 샘플링 온도 (0.0 = 결정적, 높을수록 다양함).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=700,
        help="각 완성(completion)에 대한 최대 토큰 수.",
    )
    
    # 기타 인자
    parser.add_argument("--seed", type=int, default=0, help="작업 셔플링을 위한 랜덤 시드.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",  # 기본 출력 디렉토리
        help="생성된 계획을 저장할 디렉토리.",
    )
    parser.add_argument(
        "--info-file",
        type=str,
        default="data/all_plans_env0/info.txt",
        help="액션과 객체 정보가 담긴 info.txt 파일 경로.",
    )
    parser.add_argument(
        "--use-ai2thor-objects",
        action="store_true",  # 플래그 인자 (있으면 True)
        help="info.txt 파일 대신 AI2THOR 환경에서 동적으로 객체와 액션을 로드합니다.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="FloorPlan1",
        help="AI2THOR 씬 이름 (--use-ai2thor-objects가 활성화된 경우 사용).",
    )
    parser.add_argument(
        "--export-info-txt",
        type=str,
        default=None,
        help="AI2THOR 객체와 액션을 지정된 경로의 info.txt 파일로 내보냅니다.",
    )
    parser.add_argument(
        "--scene-graph",
        type=str,
        default=None,
        help="Scene Graph JSON 파일 경로 (물리적 검증용). 지정하지 않으면 scene 번호 입력 시 자동 생성.",
    )
    parser.add_argument(
        "--scene-number",
        type=int,
        default=None,
        help="FloorPlan 번호 (예: 1, 201, 301, 401). --scene-graph가 지정되지 않으면 이 번호로 자동 경로 생성.",
    )

    # 명령줄 인자 파싱
    args = parser.parse_args()
    
    # Scene 번호 입력 받기 (--scene-number이 없으면 사용자 입력)
    if args.scene_number is None:
        try:
            scene_number = int(input("FloorPlan 번호를 입력하세요 (예: 1, 201, 301, 401): "))
        except (ValueError, KeyboardInterrupt):
            print("❌ 잘못된 입력입니다. 숫자를 입력해주세요.")
            sys.exit(1)
    else:
        scene_number = args.scene_number
    
    # Scene Graph 경로 결정
    if args.scene_graph:
        # 명령줄에서 직접 지정된 경우
        scene_graph_path = Path(args.scene_graph)
    else:
        # Scene 번호에 따라 자동 생성
        scene_graph_path = Path(f"scripts/scene_graph_structured_FloorPlan{scene_number}.json")
        print(f"🔍 FloorPlan {scene_number}의 Scene Graph 사용: {scene_graph_path}")
    
    # 랜덤 시드 설정 (재현 가능한 결과를 위해)
    random.seed(args.seed)

    # 작업 목록 로드 (파일 또는 인라인)
    tasks = load_task_list(args.task_file, args.tasks)
    if not tasks:
        raise ValueError("작업이 제공되지 않았습니다. --tasks 또는 --task-file을 사용하세요.")

    # AI2THOR 환경 또는 info.txt 파일에서 액션과 객체 로드
    if args.use_ai2thor_objects:
        print(f"📦 AI2THOR 환경에서 액션과 객체 로드 중 (씬: {args.scene})")
        
        # 객체를 가져오기 위해 AI2THOR 실행자 초기화
        executor = ManipulaThorExecutor(scene=args.scene, headless=True)  # 헤드리스 모드로 실행
        executor.initialize()  # 환경 초기화
        
        # AI2THOR에서 사용 가능한 객체 목록 가져오기
        ai2thor_objects = executor.get_available_objects()
        # 객체 타입만 추출하여 정렬된 리스트로 변환
        objects = sorted([obj["objectType"] for obj in ai2thor_objects if obj["objectType"]])
        
        # AI2THOR에서 사용 가능한 액션 목록 가져오기
        actions = executor.get_available_actions()
        
        print(f"   AI2THOR에서 {len(actions)}개의 액션과 {len(objects)}개의 객체를 찾았습니다.")
        
        # 요청된 경우 info.txt 형식으로 내보내기
        if args.export_info_txt:
            executor.export_to_info_txt_format(args.export_info_txt)
        
        # 실행자는 나중에 실행을 위해 새로 생성할 것이므로 여기서 닫기
        executor.close()
    else:
        # info.txt 파일에서 액션과 객체 파싱
        info_path = Path(args.info_file)
        if info_path.exists():
            print(f"📦 {info_path}에서 액션과 객체 로드 중")
            # 파일 파싱하여 액션과 객체 추출
            actions, objects = parse_info_txt(str(info_path))
            
            # 액션이 없으면 기본 액션 사용
            if not actions:
                print("⚠️  info.txt에서 액션을 찾을 수 없습니다. 기본 액션을 사용합니다.")
                actions = AI2THOR_ACTIONS
            
            # 객체가 없으면 기본 객체 사용
            if not objects:
                print("⚠️  info.txt에서 객체를 찾을 수 없습니다. 기본 객체를 사용합니다.")
                objects = DEFAULT_FLOORPLAN1_OBJECTS
            else:
                # 객체 목록 정렬
                objects = sorted(objects)
            print(f"   {len(actions)}개의 액션과 {len(objects)}개의 객체를 찾았습니다.")
        else:
            # 파일이 없으면 기본값 사용
            print(f"⚠️  {info_path}에서 info 파일을 찾을 수 없습니다. 기본 액션과 객체를 사용합니다.")
            actions = AI2THOR_ACTIONS
            objects = sorted(DEFAULT_FLOORPLAN1_OBJECTS)

    # 예제 프로그램 딕셔너리 복사 (원본 보존)
    examples = dict(list(DEFAULT_EXAMPLES.items()))
    
    # 시스템 프롬프트 구성 (객체, 액션, 예제 포함)
    prompt = build_prompt(objects, actions, examples, max_examples=args.max_examples)

    # 출력 디렉토리 생성 (없으면 생성)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 타임스탬프 생성 (파일명에 사용)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # JSON 출력 파일 경로 생성
    output_path = Path(args.output_dir) / f"ai2thor_progprompt_{timestamp}.json"

    # Scene Graph 로드 (물리적 검증용)
    # scene_graph_path는 이미 위에서 결정됨
    if not scene_graph_path.is_absolute():
        scene_graph_path = Path(__file__).parent.parent / scene_graph_path
    
    # Scene Graph 파일 존재 확인
    if not scene_graph_path.exists():
        print(f"❌ 오류: Scene Graph 파일을 찾을 수 없습니다: {scene_graph_path.absolute()}")
        print(f"   다음 경로를 확인했습니다:")
        print(f"     - {scene_graph_path.absolute()}")
        sys.exit(1)
    
    # 업데이트된 Scene Graph 파일 경로 생성 (원본과 같은 디렉토리에)
    updated_scene_graph_path = scene_graph_path.parent / "updated_scene_graph.json"
    
    # 처음에만 원본 Scene Graph를 updated_scene_graph.json으로 복사
    shutil.copy2(str(scene_graph_path), str(updated_scene_graph_path))
    logger.info(f"원본 Scene Graph를 {updated_scene_graph_path}로 복사 완료 (초기 복사)")
    
    # 이후부터는 updated_scene_graph.json만 사용 (원본은 읽기 전용으로 유지)
    scene_graph = load_scene_graph(str(updated_scene_graph_path))
    logger.info(f"업데이트된 Scene Graph 로드 완료: {updated_scene_graph_path}")
    
    # AI2-THOR Controller 초기화 (NavMesh 검증용, 선택사항)
    controller = None
    try:
        from ai2thor.controller import Controller
        logger.info("📦 AI2-THOR Controller 초기화 중...")
        # Scene 번호에 따라 scene 이름 생성
        scene_name = f"FloorPlan{scene_number}_physics"
        logger.info(f"   Scene: {scene_name}")
        controller = Controller(
            agentMode="arm",
            scene=scene_name,
            gridSize=0.25,
            snapToGrid=False,
            rotateStepDegrees=90,
            visibilityDistance=1.5,
            renderInstanceSegmentation=False,
            renderDepthImage=False,
            renderSemanticSegmentation=False,
            width=300,
            height=300,
            fieldOfView=90
        )
        logger.info("✓ AI2-THOR Controller 초기화 완료")
    except ImportError:
        logger.warning("⚠️  ai2thor 라이브러리가 설치되지 않았습니다. NavMesh 검증이 제한됩니다.")
    except Exception as e:
        logger.warning(f"⚠️  AI2-THOR Controller 초기화 실패: {e}. NavMesh 검증이 제한됩니다.")
    
    # 각 작업에 대한 프로그램 생성 및 물리적 검증
    plan_dict: Dict[str, str] = {}  # 작업명: 프로그램 코드 딕셔너리
    physical_verification_results: Dict[str, Dict[str, Any]] = {}  # 물리적 검증 결과
    
    for task in tasks:
        print(f"🧠 작업에 대한 프로그램 생성 중: {task}")
        
        # 각 작업마다 새로운 클라이언트 생성 (LLM 상태 초기화)
        client = OpenAI(base_url=args.ollama_url, api_key="ollama")
        
        # 각 task 시작 전에 원본 Scene Graph를 updated_scene_graph.json으로 복사 (원본으로 리셋)
        shutil.copy2(str(scene_graph_path), str(updated_scene_graph_path))
        logger.info(f"원본 Scene Graph를 {updated_scene_graph_path}로 복사 완료 (task 시작 전 리셋)")
        
        # 업데이트된 Scene Graph 로드
        scene_graph = load_scene_graph(str(updated_scene_graph_path))
        logger.info(f"업데이트된 Scene Graph 로드 완료: {updated_scene_graph_path}")
        
        # Step 1: 논리적 검증 - LLM을 사용하여 프로그램 생성
        initial_program = generate_program(
            client=client,  # API 클라이언트 (각 작업마다 새로 생성)
            model=args.model,  # 사용할 모델
            base_prompt=prompt,  # 시스템 프롬프트
            task=task,  # 작업 설명
            temperature=args.temperature,  # 샘플링 온도
            max_tokens=args.max_tokens,  # 최대 토큰 수
        )
        
        print("논리적 검증 완료된 프로그램:")
        print(initial_program)
        print("-" * 80)
        
        # Step 2: 물리적 검증 - Scene Graph 기반 검증
        
        final_program, verification_result = generate_final_plan_with_physical_verification(
            task=task,
            initial_program=initial_program,
            scene_graph=scene_graph.copy(),  # 복사본 사용 (각 task마다 독립적으로 검증)
            controller=controller,
            client=client,  # LLM 클라이언트 전달
            model=args.model,  # LLM 모델 전달
            scene_graph_path=str(updated_scene_graph_path)  # 업데이트된 Scene Graph JSON 파일 경로 전달
        )
        
        # 생성된 최종 프로그램을 딕셔너리에 저장
        plan_dict[task] = final_program
        physical_verification_results[task] = verification_result
        
        print("물리적 검증 완료된 최종 프로그램:")
        print(final_program)
        print("-" * 80)
        
        # 모든 액션이 통과했는지 확인
        failed_actions_count = verification_result.get("failed_actions", 0)
        all_actions_passed = failed_actions_count == 0
        
        # Step 3: Task 완료 검증 - 모든 액션이 통과하지 않았을 때만 LLM으로 누락된 task 찾기
        if all_actions_passed:
            print(f"\n✅ 모든 액션이 통과했습니다. LLM으로 누락된 task를 찾지 않습니다.")
            all_completed = True
            missing_tasks = []
            reasoning = "모든 액션이 물리적 검증을 통과했습니다."
            missing_task_plans = []
        else:
            print(f"\n🔍 Task 완료 검증 중: '{task}' (일부 액션이 실패했으므로 누락된 task 확인)")
            all_completed, missing_tasks, reasoning = verify_task_completion_with_llm(
                client=client,
                model=args.model,
                task=task,
                final_plan=final_program
            )
            
            if all_completed:
                print(f"✅ 모든 필요한 task가 완료되었습니다.")
                missing_task_plans = []
            else:
                print(f"⚠️  일부 task가 누락되었습니다:")
                for i, missing_task in enumerate(missing_tasks, 1):
                    print(f"  {i}. {missing_task}")
                print(f"\n📝 검증 결과: {reasoning}")
                
                # Step 4: 누락된 task들에 대한 plan 생성 (논리적 + 물리적 검증)
                print(f"\n🔧 누락된 task들에 대한 plan 생성 중...")
                missing_task_plans = []
                
                for missing_task in missing_tasks:
                    print(f"\n  📋 누락된 task 처리 중: '{missing_task}'")
                    
                    # 논리적 검증 - LLM을 사용하여 프로그램 생성
                    missing_initial_program = generate_program(
                        client=client,
                        model=args.model,
                        base_prompt=prompt,
                        task=missing_task,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                    
                    print(f"    논리적 검증 완료된 프로그램:")
                    print(f"    {missing_initial_program[:200]}...")  # 처음 200자만 출력
                    
                    # 물리적 검증 - Scene Graph 기반 검증
                    # 누락된 task 처리 시에도 updated_scene_graph.json 사용
                    missing_final_program, missing_verification_result = generate_final_plan_with_physical_verification(
                        task=missing_task,
                        initial_program=missing_initial_program,
                        scene_graph=scene_graph.copy(),  # 복사본 사용
                        controller=controller,
                        client=client,
                        model=args.model,
                        scene_graph_path=str(updated_scene_graph_path)  # 업데이트된 Scene Graph JSON 파일 경로 전달
                    )
                    
                    missing_task_plans.append({
                        "task": missing_task,
                        "plan": missing_final_program,
                        "verification_result": missing_verification_result
                    })
                    
                    print(f"    ✓ '{missing_task}'에 대한 plan 생성 완료")
            
            # 누락된 task들의 plan을 기존 plan에 추가
            if missing_task_plans:
                print(f"\n  📝 누락된 task들의 plan을 기존 plan에 추가 중...")
                
                # 기존 plan에 누락된 task들의 plan 추가
                additional_plan_lines = []
                for missing_plan_info in missing_task_plans:
                    missing_task_name = missing_plan_info["task"]
                    missing_plan = missing_plan_info["plan"]
                    
                    # plan에서 함수 본문만 추출 (def ... 제외)
                    plan_lines = missing_plan.split("\n")
                    function_body = []
                    in_function = False
                    for line in plan_lines:
                        if line.strip().startswith("def "):
                            in_function = True
                            continue
                        if in_function:
                            function_body.append(line)
                    
                    additional_plan_lines.append(f"\t# [LLM 생성 - 누락 Task 보완] Additional task: {missing_task_name}")
                    # 누락된 task plan의 모든 라인에서 마커 변경
                    for line in function_body:
                        # 기존 [LLM 생성 - 논리적 검증] 마커를 [LLM 생성 - 누락 Task 보완]으로 변경
                        if "[LLM 생성 - 논리적 검증]" in line:
                            line = line.replace("[LLM 생성 - 논리적 검증]", "[LLM 생성 - 누락 Task 보완]")
                        additional_plan_lines.append(line)
                
                # 기존 plan에 추가
                if additional_plan_lines:
                    # 기존 plan의 마지막 줄 찾기
                    final_plan_lines = final_program.split("\n")
                    # 마지막 } 전에 추가
                    new_final_plan = final_program
                    if new_final_plan.rstrip().endswith("}"):
                        new_final_plan = new_final_plan.rstrip()[:-1]  # 마지막 } 제거
                        new_final_plan += "\n".join(additional_plan_lines) + "\n}"
                    else:
                        new_final_plan += "\n" + "\n".join(additional_plan_lines)
                    
                    final_program = new_final_plan
                    plan_dict[task] = final_program  # 업데이트된 plan 저장
                    
                    print(f"    ✓ 누락된 task들의 plan 추가 완료")
        
        # 검증 결과를 physical_verification_results에 추가
        physical_verification_results[task]["task_completion"] = {
            "all_completed": all_completed,
            "missing_tasks": missing_tasks,
            "reasoning": reasoning,
            "missing_task_plans": missing_task_plans if not all_completed else []
        }
        print("-" * 80)

    # JSON 파일로 계획 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(plan_dict, f, indent=2, ensure_ascii=False)  # 들여쓰기 2칸, 한글 지원
    print(f"✅ 계획이 {output_path}에 저장되었습니다.")

    # 작업 이름과 타임스탬프가 포함된 텍스트 파일로도 저장
    # 첫 번째 작업 이름으로 안전한 파일명 생성
    if plan_dict:
        first_task = list(plan_dict.keys())[0]  # 첫 번째 작업 이름 가져오기
        
        # 파일명에 사용할 수 있도록 특수문자 제거 및 공백을 언더스코어로 변환
        safe_task_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in first_task)
        safe_task_name = safe_task_name.replace(' ', '_')[:50]  # 길이 제한 및 공백을 언더스코어로 변환
        txt_filename = f"physical_guard_set3_result_{safe_task_name}_{timestamp}.txt"
    else:
        # 작업이 없으면 타임스탬프만 사용
        txt_filename = f"physical_guard_set3_result_{timestamp}.txt"
    
    # 텍스트 출력 파일 경로 생성
    txt_output_path = Path(args.output_dir) / txt_filename
    
    # 텍스트 파일로 계획 저장 (읽기 쉬운 형식)
    with open(txt_output_path, "w", encoding="utf-8") as f:
        for task, program in plan_dict.items():
            f.write(f"Task: {task}\n")  # 작업 이름
            f.write("=" * 80 + "\n")  # 구분선
            f.write(program)  # 프로그램 코드
            f.write("\n\n")
            
            # 물리적 검증 결과 추가
            if task in physical_verification_results:
                result = physical_verification_results[task]
                f.write("Physical Verification Summary:\n")
                f.write(f"  Total Actions: {result.get('total_actions', 0)}\n")
                f.write(f"  Passed Actions: {result.get('passed_actions', 0)}\n")
                f.write(f"  Failed Actions: {result.get('failed_actions', 0)}\n")
                if result.get('failed_actions_list'):
                    f.write("\nFailed Actions:\n")
                    for fa in result['failed_actions_list']:
                        f.write(f"  - {fa['action'].get('line', '')}: {fa['reason']}\n")
                
                # Task 완료 검증 결과 추가
                task_completion = result.get('task_completion', {})
                if task_completion:
                    f.write("\nTask Completion Verification:\n")
                    if task_completion.get('all_completed', False):
                        f.write("  ✅ All tasks completed: Yes\n")
                    else:
                        f.write("  ⚠️  All tasks completed: No\n")
                        missing_tasks = task_completion.get('missing_tasks', [])
                        if missing_tasks:
                            f.write(f"  Missing Tasks ({len(missing_tasks)}):\n")
                            for i, missing_task in enumerate(missing_tasks, 1):
                                f.write(f"    {i}. {missing_task}\n")
                    reasoning = task_completion.get('reasoning', '')
                    if reasoning:
                        f.write(f"  Reasoning: {reasoning}\n")
                
                f.write("\n" + "=" * 80 + "\n\n")  # 작업 간 구분
    print(f"✅ 계획이 {txt_output_path}에도 저장되었습니다.")
    
    # Controller 종료
    if controller:
        try:
            controller.stop()
            logger.info("✓ AI2-THOR Controller 종료 완료")
        except Exception as e:
            logger.warning(f"⚠️  Controller 종료 실패: {e}")

    # Baseline(ProgPrompt).py 스크립트 실행
    script_dir = Path(__file__).parent
    baseline_progprompt_path = script_dir / "Baseline(ProgPrompt).py"
    if baseline_progprompt_path.exists():
        # Baseline(ProgPrompt).py 실행 명령어 구성
        cmd = [ 
            "python", str(baseline_progprompt_path),
            "--output-dir", str(Path(args.output_dir).resolve())
        ]
        
        # --tasks 인자로 직접 전달 (이미 파싱된 tasks 변수 사용)
        if tasks:
            cmd.append("--tasks")
            cmd.extend(tasks)
        # --tasks 인자로 직접 전달 (args.tasks가 있으면 - fallback)
        elif args.tasks:
            cmd.append("--tasks")
            cmd.extend(args.tasks)
        # --task-file이 있으면 전달
        elif args.task_file:
            cmd.extend(["--task-file", str(Path(args.task_file).resolve())])
        # 둘 다 없으면 JSON 파일 사용 (fallback)
        else:
            output_path_abs = Path(output_path).resolve()
            cmd.extend(["--task-file", str(output_path_abs)])
        
        # 추가 인자들 전달 (모델, ollama-url 등)
        if hasattr(args, 'model') and args.model:
            cmd.extend(["--model", args.model])
        if hasattr(args, 'ollama_url') and args.ollama_url:
            cmd.extend(["--ollama-url", args.ollama_url])
        if hasattr(args, 'temperature') and args.temperature is not None:
            cmd.extend(["--temperature", str(args.temperature)])
        if hasattr(args, 'max_tokens') and args.max_tokens:
            cmd.extend(["--max-tokens", str(args.max_tokens)])
        if hasattr(args, 'max_examples') and args.max_examples:
            cmd.extend(["--max-examples", str(args.max_examples)])
        if hasattr(args, 'info_file') and args.info_file:
            cmd.extend(["--info-file", str(Path(args.info_file).resolve())])
        
        subprocess.run(cmd, cwd=str(script_dir))
        print(f"✅ Baseline(ProgPrompt).py 스크립트 실행 완료")
        
        # Baseline 실행 후 평가 스크립트 실행
        evaluation_path = script_dir / "evaluation.py"
        if evaluation_path.exists():
            print(f"\n📊 Baseline과 Physical Guard 결과 비교 평가 중...")
            
            # Baseline JSON 파일 찾기 (가장 최근 파일)
            baseline_json_files = sorted(
                Path(args.output_dir).glob("ai2thor_progprompt_*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            if baseline_json_files:
                baseline_json = baseline_json_files[0]  # 가장 최근 파일
                physical_guard_txt = txt_output_path
                
                # 평가 결과 저장 경로
                eval_output_path = Path(args.output_dir) / f"evaluation_comparison_{timestamp}.txt"
                
                eval_cmd = [
                    "python", str(evaluation_path),
                    "--baseline-json", str(baseline_json.resolve()),
                    "--physical-guard-txt", str(physical_guard_txt.resolve()),
                    "--output", str(eval_output_path.resolve())
                ]
                
                # Scene Graph 파일 경로 추가 (존재하는 경우)
                baseline_sg_path = Path(__file__).parent / "baseline_updated_scene_graph.json"
                pg_sg_path = Path(__file__).parent / "updated_scene_graph.json"
                
                if baseline_sg_path.exists() and pg_sg_path.exists():
                    eval_cmd.extend([
                        "--baseline-scene-graph", str(baseline_sg_path.resolve()),
                        "--physical-guard-scene-graph", str(pg_sg_path.resolve())
                    ])
                
                subprocess.run(eval_cmd, cwd=str(script_dir))
                print(f"✅ 평가 결과가 {eval_output_path}에 저장되었습니다.")
            else:
                print(f"⚠️  Baseline JSON 파일을 찾을 수 없습니다.")
        else:
            print(f"⚠️  evaluation.py를 찾을 수 없습니다: {evaluation_path}")
    else:
        print(f"⚠️  Baseline(ProgPrompt).py를 찾을 수 없습니다: {baseline_progprompt_path}")



# 스크립트가 직접 실행될 때만 main 함수 호출
if __name__ == "__main__":
    main()