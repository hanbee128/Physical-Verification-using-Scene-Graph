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
import textwrap  # 텍스트 들여쓰기 및 포맷팅을 위한 모듈
from datetime import datetime  # 타임스탬프 생성을 위한 날짜/시간 모듈
from pathlib import Path  # 파일 경로 처리를 위한 모듈
from typing import Dict, List, Optional, Tuple, Any  # 타입 힌팅을 위한 모듈

# 로컬 모듈 임포트
from ai2thor_connector_ithor import AI2ThorExecutor  # AI2-THOR 환경 실행을 위한 커넥터

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
                        "SliceObject": "Slice sliceable food while holding a Knife",
                        "ToggleObjectOn": "Turn on toggleable object (Faucet, Microwave, Toaster, Dishwasher, CoffeeMachine, StoveBurner)",
                        "ToggleObjectOff": "Turn off toggleable object",
                        "CleanObject": "Wash/clean an object near Sink/Faucet",
                        "BreakObject": "Break breakable object",
                        "DropHandObject": "Drop whatever the agent is holding (required before picking up another object)",
                        "ThrowObject": "Throw an object",
                        "PushObject": "Push an object",
                        "PullObject": "Pull an object",
                        "UseUpObject": "Use up an object",
                        "FillObjectWithLiquid": "Fill an object with a liquid",
                        "EmptyLiquidFromObject": "Empty an object",
                        "DirtyObject": "Make an object dirty",
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
    "SliceObject <obj>             # Slice sliceable food while holding a Knife",
    "ToggleObjectOn <obj>          # Turn on toggleable object (Faucet, Microwave, Toaster, Dishwasher, CoffeeMachine, StoveBurner)",
    "ToggleObjectOff <obj>         # Turn off toggleable object",
    "CleanObject <obj>             # Wash/clean an object near Sink/Faucet",
    "BreakObject <obj>             # Break breakable object",
    "DropHandObject        # Drop whatever the agent is holding (required before picking up another object)",
    "ThrowObject              # Throw an object",
    "PushObject <obj> <obj>        # Push an object",
    "PullObject <obj> <obj>        # Pull an object",
    "UseUpObject <obj>             # Use up an object",
    "FillObjectWithLiquid <obj> <liquid>     # Fill an object with a liquid",
    "EmptyLiquidFromObject <obj>             # Empty an object",
    "DirtyObject <obj>             # Make an object dirty",
]

# FloorPlan1 씬에서 사용 가능한 기본 객체 목록 (info.txt에서 추출)
DEFAULT_FLOORPLAN1_OBJECTS = [
    "AlarmClock", "Apple", "AppleSliced", "ArmChair", "BaseballBat", "BasketBall", "Bathtub", "BathtubBasin",
    "Bed", "Blinds", "Book", "Boots", "Bottle", "Bowl", "Box", "Bread", "BreadSliced", "ButterKnife",
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
        \tPutObject('Mug', 'Sink')
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

    # 파일에서 작업 목록 읽기
    with open(task_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # 앞뒤 공백 제거
            if not line:  # 빈 줄은 건너뛰기
                continue
            try:
                # JSON 형식으로 파싱 시도
                data = json.loads(line)
            except json.JSONDecodeError:
                # JSON이 아니면 일반 텍스트로 처리하여 작업 목록에 추가
                tasks.append(line)
                continue
            
            # 파싱된 데이터 타입에 따라 처리
            if isinstance(data, dict):
                # 딕셔너리인 경우 키들을 작업으로 사용
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
# Scene Graph 업데이트 함수들 (Baseline용)
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
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("def "):
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
        elif action == "Cook":
            action_type = "CookObject"
        
        if len(params) == 1:
            plan.append({
                "type": action_type,
                "args": {"o": params[0]},
                "line": line
            })
        elif len(params) == 2:
            plan.append({
                "type": action_type,
                "args": {"o": params[0], "r": params[1]},
                "line": line
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


def save_scene_graph_to_file(scene_graph: Dict[str, Any], scene_graph_path: str):
    """Scene Graph를 JSON 파일에 저장"""
    try:
        with open(scene_graph_path, "w", encoding="utf-8") as f:
            json.dump(scene_graph, f, indent=2, ensure_ascii=False)
        logger.debug(f"Scene Graph 저장 완료: {scene_graph_path}")
    except Exception as e:
        logger.error(f"Scene Graph 저장 실패: {e}")


def update_scene_graph_after_action(
    scene_graph: Dict[str, Any],
    action: Dict[str, Any],
    scene_graph_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    액션 실행 후 Scene Graph 업데이트 및 JSON 파일 저장 (Baseline용 - 검증 없이 실행)
    
    Args:
        scene_graph: 현재 Scene Graph
        action: 실행된 액션
        scene_graph_path: Scene Graph JSON 파일 경로 (None이면 저장 안 함)
        
    Returns:
        업데이트된 Scene Graph
    """
    action_type = action.get("type", "")
    args = action.get("args", {})
    object_name = args.get("o")
    receptacle_name = args.get("r")
    
    nodes = scene_graph.get("nodes", {})
    edges = scene_graph.get("edges", [])
    agent_node = nodes.get("agent", {})
    object_nodes = nodes.get("objects", [])
    
    if action_type == "GoToObject":
        # GoToObject: Agent 위치를 객체 근처로 업데이트 (간단한 추정)
        if object_name:
            target_obj = None
            for obj_node in object_nodes:
                obj_type = obj_node.get("objectType", "")
                if object_name.lower() in obj_type.lower():
                    target_obj = obj_node
                    break
            
            if target_obj:
                obj_pos = target_obj.get("position", {})
                if obj_pos:
                    # 객체 위치 근처로 Agent 위치 업데이트 (간단한 추정)
                    agent_node["position"] = {
                        "x": obj_pos.get("x", 0) - 0.5,  # 객체 앞 0.5m
                        "y": obj_pos.get("y", 0),
                        "z": obj_pos.get("z", 0) - 0.5
                    }
                    logger.info(f"  ✓ Agent 위치 업데이트: GoToObject('{object_name}')")
    
    elif action_type == "PickupObject":
        # HOLDS 엣지 추가, IN 엣지 제거
        if object_name:
            for obj_node in object_nodes:
                obj_type = obj_node.get("objectType", "")
                if object_name.lower() in obj_type.lower():
                    obj_id = obj_node.get("nodeId", "")
                    
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
                    obj_node["isPickedUp"] = True
                    if "parentReceptacles" in obj_node:
                        obj_node["parentReceptacles"] = []
                    break
    
    elif action_type == "PutObject":
        # HOLDS 엣지 제거, IN 엣지 추가
        if object_name and receptacle_name:
            for obj_node in object_nodes:
                obj_type = obj_node.get("objectType", "")
                if object_name.lower() in obj_type.lower():
                    obj_id = obj_node.get("nodeId", "")
                    
                    # HOLDS 엣지 제거
                    edges = [e for e in edges if not (
                        e.get("edgeType") == "HOLDS" and e.get("target") == obj_id
                    )]
                    
                    # 수용체 찾기
                    for recp_node in object_nodes:
                        recp_type = recp_node.get("objectType", "")
                        if receptacle_name.lower() in recp_type.lower():
                            recp_id = recp_node.get("nodeId", "")
                            
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
                            obj_node["isPickedUp"] = False
                            if "parentReceptacles" not in obj_node:
                                obj_node["parentReceptacles"] = []
                            if recp_id not in obj_node["parentReceptacles"]:
                                obj_node["parentReceptacles"].append(recp_id)
                            break
                    
                    # Agent 노드 업데이트
                    agent_node["isHolding"] = False
                    agent_node["heldObjectId"] = None
                    break
    
    elif action_type == "OpenObject":
        # Object 노드의 isOpen 업데이트
        if object_name:
            for obj_node in object_nodes:
                obj_type = obj_node.get("objectType", "")
                obj_id = obj_node.get("nodeId", "")
                if object_name.lower() in obj_type.lower() or object_name.lower() in obj_id.lower():
                    obj_node["isOpen"] = True
                    obj_node["openness"] = 1.0
                    logger.debug(f"  → OpenObject: '{object_name}'의 isOpen=True, openness=1.0으로 업데이트됨")
                    break
    
    elif action_type == "CloseObject":
        # Object 노드의 isOpen 업데이트
        if object_name:
            for obj_node in object_nodes:
                obj_type = obj_node.get("objectType", "")
                obj_id = obj_node.get("nodeId", "")
                if object_name.lower() in obj_type.lower() or object_name.lower() in obj_id.lower():
                    obj_node["isOpen"] = False
                    obj_node["openness"] = 0.0
                    logger.debug(f"  → CloseObject: '{object_name}'의 isOpen=False, openness=0.0으로 업데이트됨")
                    break
    
    elif action_type == "ToggleObjectOn":
        # Object 노드의 isToggled 업데이트
        if object_name:
            for obj_node in object_nodes:
                obj_type = obj_node.get("objectType", "")
                obj_id = obj_node.get("nodeId", "")
                if object_name.lower() in obj_type.lower() or object_name.lower() in obj_id.lower():
                    obj_node["isToggled"] = True
                    logger.debug(f"  → ToggleObjectOn: '{object_name}'의 isToggled=True로 업데이트됨")
                    break
    
    elif action_type == "ToggleObjectOff":
        # Object 노드의 isToggled 업데이트
        if object_name:
            for obj_node in object_nodes:
                obj_type = obj_node.get("objectType", "")
                obj_id = obj_node.get("nodeId", "")
                if object_name.lower() in obj_type.lower() or object_name.lower() in obj_id.lower():
                    obj_node["isToggled"] = False
                    logger.debug(f"  → ToggleObjectOff: '{object_name}'의 isToggled=False로 업데이트됨")
                    break
    
    elif action_type == "SliceObject":
        # Object 노드의 isSliced 업데이트
        if object_name:
            for obj_node in object_nodes:
                obj_type = obj_node.get("objectType", "")
                obj_id = obj_node.get("nodeId", "")
                if object_name.lower() in obj_type.lower() or object_name.lower() in obj_id.lower():
                    obj_node["isSliced"] = True
                    logger.debug(f"  → SliceObject: '{object_name}'의 isSliced=True로 업데이트됨")
                    break
    
    elif action_type == "BreakObject":
        # Object 노드의 isBroken 업데이트
        if object_name:
            for obj_node in object_nodes:
                obj_type = obj_node.get("objectType", "")
                obj_id = obj_node.get("nodeId", "")
                if object_name.lower() in obj_type.lower() or object_name.lower() in obj_id.lower():
                    obj_node["isBroken"] = True
                    logger.debug(f"  → BreakObject: '{object_name}'의 isBroken=True로 업데이트됨")
                    break
    
    # 업데이트된 Scene Graph 반환
    scene_graph["nodes"]["agent"] = agent_node
    scene_graph["nodes"]["objects"] = object_nodes
    scene_graph["edges"] = edges
    
    # JSON 파일에 저장
    if scene_graph_path:
        save_scene_graph_to_file(scene_graph, scene_graph_path)
        logger.info(f"  💾 Scene Graph JSON 파일 업데이트 완료: {scene_graph_path}")
    
    return scene_graph


def update_scene_graph_from_plan(
    scene_graph: Dict[str, Any],
    program_code: str,
    scene_graph_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Plan 코드를 파싱하여 각 액션을 실행하고 Scene Graph 업데이트
    
    Args:
        scene_graph: 현재 Scene Graph
        program_code: 프로그램 코드
        scene_graph_path: Scene Graph JSON 파일 경로
        
    Returns:
        업데이트된 Scene Graph
    """
    # Plan 파싱
    actions = parse_program_to_actions(program_code)
    logger.info(f"  📋 {len(actions)}개의 액션 파싱 완료")
    
    # 각 액션 실행 및 Scene Graph 업데이트
    for i, action in enumerate(actions):
        action_type = action.get("type", "")
        action_line = action.get("line", "")
        logger.info(f"  [{i+1}/{len(actions)}] {action_line}")
        
        # Scene Graph 업데이트 (검증 없이 실행)
        scene_graph = update_scene_graph_after_action(
            scene_graph, action, scene_graph_path
        )
    
    return scene_graph


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
        executor = AI2ThorExecutor(scene=args.scene, headless=True)  # 헤드리스 모드로 실행
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
    output_path = Path(args.output_dir) / f"baseline_result_{timestamp}.json"
    
    # Scene Graph 경로 결정
    scene_graph_path = None
    baseline_updated_scene_graph_path = None
    
    if args.scene_number is not None or args.scene_graph is not None:
        if args.scene_graph:
            # 명령줄에서 직접 지정된 경우
            scene_graph_path = Path(args.scene_graph)
        else:
            # Scene 번호에 따라 자동 생성
            scene_number = args.scene_number
            scene_graph_path = Path(f"scripts/scene_graph_structured_FloorPlan{scene_number}.json")
            print(f"🔍 FloorPlan {scene_number}의 Scene Graph 사용: {scene_graph_path}")
        
        # 상대 경로인 경우 절대 경로로 변환
        if not scene_graph_path.is_absolute():
            scene_graph_path = Path(__file__).parent.parent / scene_graph_path
        
        # Scene Graph 파일 존재 확인
        if not scene_graph_path.exists():
            print(f"⚠️  Scene Graph 파일을 찾을 수 없습니다: {scene_graph_path.absolute()}")
            print(f"   Scene Graph 업데이트를 건너뜁니다.")
            scene_graph_path = None
        else:
            # Baseline용 업데이트된 Scene Graph 파일 경로 생성
            baseline_updated_scene_graph_path = scene_graph_path.parent / "baseline_updated_scene_graph.json"
            print(f"📋 Baseline용 Scene Graph 업데이트 파일: {baseline_updated_scene_graph_path}")

    # 각 작업에 대한 프로그램 생성
    plan_dict: Dict[str, str] = {}  # 작업명: 프로그램 코드 딕셔너리
    for task in tasks:
        print(f"🧠 작업에 대한 프로그램 생성 중: {task}")
        
        # 각 작업마다 새로운 클라이언트 생성 (LLM 상태 초기화)
        client = OpenAI(base_url=args.ollama_url, api_key="ollama")
        
        # LLM을 사용하여 프로그램 생성
        program = generate_program(
            client=client,  # API 클라이언트 (각 작업마다 새로 생성)
            model=args.model,  # 사용할 모델
            base_prompt=prompt,  # 시스템 프롬프트
            task=task,  # 작업 설명
            temperature=args.temperature,  # 샘플링 온도
            max_tokens=args.max_tokens,  # 최대 토큰 수
        )
        
        # 생성된 프로그램을 딕셔너리에 저장
        plan_dict[task] = program
        
        # 생성된 프로그램 출력 (콘솔 확인용)
        print(program)
        print("-" * 80)  # 구분선
        
        # Scene Graph 업데이트 (Baseline용)
        if scene_graph_path and baseline_updated_scene_graph_path:
            # 각 task 시작 전에 원본 Scene Graph를 baseline_updated_scene_graph.json으로 복사 (원본으로 리셋)
            shutil.copy2(str(scene_graph_path), str(baseline_updated_scene_graph_path))
            logger.info(f"원본 Scene Graph를 {baseline_updated_scene_graph_path}로 복사 완료 (task 시작 전 리셋)")
            
            # 업데이트된 Scene Graph 로드
            scene_graph = load_scene_graph(str(baseline_updated_scene_graph_path))
            logger.info(f"업데이트된 Scene Graph 로드 완료: {baseline_updated_scene_graph_path}")
            
            # Plan을 실행하여 Scene Graph 업데이트
            logger.info(f"📋 Plan 실행 및 Scene Graph 업데이트 중: '{task}'")
            scene_graph = update_scene_graph_from_plan(
                scene_graph, program, str(baseline_updated_scene_graph_path)
            )
            logger.info(f"✓ Scene Graph 업데이트 완료: '{task}'")

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
        txt_filename = f"run_eval_result_{safe_task_name}_{timestamp}.txt"
    else:
        # 작업이 없으면 타임스탬프만 사용
        txt_filename = f"run_eval_result_{timestamp}.txt"
    
    # 텍스트 출력 파일 경로 생성
    txt_output_path = Path(args.output_dir) / txt_filename
    
    # 텍스트 파일로 계획 저장 (읽기 쉬운 형식)
    with open(txt_output_path, "w", encoding="utf-8") as f:
        for task, program in plan_dict.items():
            f.write(f"Task: {task}\n")  # 작업 이름
            f.write("=" * 80 + "\n")  # 구분선
            f.write(program)  # 프로그램 코드
            f.write("\n\n" + "=" * 80 + "\n\n")  # 작업 간 구분
    print(f"✅ 계획이 {txt_output_path}에도 저장되었습니다.")


# 스크립트가 직접 실행될 때만 main 함수 호출
if __name__ == "__main__":
    main()