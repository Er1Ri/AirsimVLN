# -*- coding: utf-8 -*-
"""
High-level controller that combines:
1) Visual grounding with RAM + GroundingDINO via ``AirSimWrapper``.
2) Task planning & code synthesis via DeepSeek's Chat Completions API.
3) Execution of generated AirSim control code inside a constrained sandbox.

Prerequisites
-------------
* Set DEEPSEEK_API_KEY in your environment.
* Download pretrained weights referenced by ``AirSimWrapper`` (RAM + GroundingDINO).
* Start an AirSim multirotor simulation.

Usage
-----
python ds_airsim.py --goal "起飞后寻找门并降落在门前"
"""
import argparse
import os
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from airsim_wrapper_ob import AirSimWrapper


# ------------------------------- LLM Client --------------------------------- #
class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        api_base: str = "https://api.deepseek.com/v1",
        timeout: int = 60,
    ) -> None:
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY 环境变量未设置。")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )
        self.api_base = api_base.rstrip("/")

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        """
        Call DeepSeek Chat Completions.
        """
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        response = self.session.post(
            f"{self.api_base}/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


# ------------------------------- Controller --------------------------------- #
SYSTEM_PROMPT = """
你是无人机多模态控制助手。你可以根据当前观测结果和目标任务，生成可直接执行的 Python 控制代码。
必须严格使用已经封装好的 AirSim 控制接口：aw.takeoff(), aw.land(), aw.get_drone_position(),
aw.fly_to(point), aw.fly_path(points), aw.set_yaw(yaw), aw.get_yaw(), aw.get_position(object_name),
aw.reset(), aw.get_distance(), aw.look_at(yaw_degree), aw.turn(angle), aw.move(distance),
aw.turn_left(), aw.turn_right(), aw.forward(), aw.get_image(), aw.get_objects(img),
aw.ob_objects(obj_name_list), aw.ob_objects_llm(obj_name_list)。

注意：
1. 所有位置点格式为 [x, y, z] 或 [x, y, z, yaw]，单位米/度。
2. 高度 z > 0 表示向上（内部会处理 AirSim 坐标系）。
3. 生成代码必须放在 ```python ... ``` 代码块内，且不包含多余文本。
4. 如果需要识别目标，请调用 aw.get_image() 获取图像，再用 aw.get_objects(img) 获取类别列表，
   然后用 aw.ob_objects(obj_name_list) 获得带角度的框选信息。
"""


class DeepSeekAirSim:
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        api_base: str = "https://api.deepseek.com/v1",
        output_dir: str = "outputs",
    ) -> None:
        self.aw = AirSimWrapper()
        self.llm = DeepSeekClient(api_key=api_key, model=model, api_base=api_base)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chat_history: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # ---------------------------- Utility methods ---------------------------- #
    @staticmethod
    def extract_python_code(content: str) -> Optional[str]:
        code_block_regex = re.compile(r"```python\\s*(.*?)```", re.DOTALL)
        match = code_block_regex.search(content)
        if match:
            return match.group(1).strip()
        return None

    def build_observation(self) -> Tuple[List[str], List[Tuple[str, float, float]], str]:
        """
        Capture image, run RAM + GroundingDINO via wrapper, save annotated frame.
        Returns: (detected_names, [(name, distance, yaw)], saved_path)
        """
        img = self.aw.get_image()
        detected_names = self.aw.get_objects(img)
        phrases, obj_infos, annotated_frame = self.aw.ob_objects(detected_names)

        # Save annotated visualization for debugging.
        save_path = self.output_dir / "latest_detection.jpg"
        import cv2  # local import to avoid hard dependency if unused

        cv2.imwrite(str(save_path), annotated_frame)

        ob_for_llm = []
        for name, camera_distance, _, yaw_degree, _, _ in obj_infos:
            ob_for_llm.append((name, float(camera_distance), float(yaw_degree)))
        return detected_names, ob_for_llm, str(save_path)

    def plan(self, goal: str, observation: List[Tuple[str, float, float]]) -> str:
        obs_text = "; ".join([f"{name} 距离 {dist:.2f}m 偏航 {yaw:.1f}°" for name, dist, yaw in observation])
        user_prompt = textwrap.dedent(
            f"""
            当前任务: {goal}
            观测摘要: {obs_text or '暂无可识别目标'}
            请给出下一步动作的可执行 Python 代码，务必使用 aw.* 接口，不要调用未定义函数。
            """
        ).strip()
        self.chat_history.append({"role": "user", "content": user_prompt})
        response = self.llm.chat(self.chat_history)
        self.chat_history.append({"role": "assistant", "content": response})
        return response

    def execute_generated_code(self, code: str) -> None:
        safe_globals = {"aw": self.aw, "__builtins__": {"range": range, "len": len, "min": min, "max": max}}
        exec(code, safe_globals, {})

    # ------------------------------- Pipeline -------------------------------- #
    def run_once(self, goal: str) -> None:
        detected_names, observation, annotated_path = self.build_observation()
        print(f"[INFO] Detected objects: {detected_names}")
        print(f"[INFO] Observation for LLM: {observation}")
        print(f"[INFO] Annotated frame saved to: {annotated_path}")

        llm_response = self.plan(goal, observation)
        print("[LLM RESPONSE]\n", llm_response)

        code = self.extract_python_code(llm_response)
        if not code:
            raise RuntimeError("未在大模型回复中找到可执行的 Python 代码块。")

        print("[EXEC] Running generated code...")
        self.execute_generated_code(code)
        print("[EXEC] Completed.")


# ----------------------------------- CLI ----------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek + RAM + GroundingDINO AirSim controller")
    parser.add_argument("--goal", type=str, required=True, help="高层任务描述，如：'起飞后寻找门并降落在门前'")
    parser.add_argument("--api-base", type=str, default="https://api.deepseek.com/v1", help="DeepSeek API base url")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="DeepSeek model name")
    parser.add_argument("--output-dir", type=str, default="outputs", help="检测结果可视化的输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    controller = DeepSeekAirSim(api_key=api_key, model=args.model, api_base=args.api_base, output_dir=args.output_dir)
    controller.run_once(goal=args.goal)


if __name__ == "__main__":
    main()
