# -*- coding: utf-8 -*-
# @Time    : 2023/11/18  23:12
# @Author  : mariswang@rflysim
# @File    : ernie_airsim.py
# @Software: PyCharm
# @Describe: RAM+GroundingDINO 感知 + DeepSeek 规划/代码生成 + AirSim 仿真执行
# -*- encoding:utf-8 -*-

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import re
import cv2

# 按你的 wrapper 文件规范导入（不修改 wrapper，只使用它）
from airsim_wrapper_ob import AirSimWrapper

# DeepSeek OpenAI 兼容接口（官方文档）
# base_url: https://api.deepseek.com  (或 https://api.deepseek.com/v1)
# models: deepseek-chat / deepseek-reasoner
# 参考：DeepSeek API Docs :contentReference[oaicite:3]{index=3}

DEEPSEEK_API_KEY = "sk-81d66b953f984c17a36ff91ab8c1e2a1"  # 改成你自己的
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MY_MODEL = "deepseek-chat"

try:
    from openai import OpenAI  # openai>=1.0.0
    _USE_NEW_OPENAI_SDK = True
except Exception:
    import openai  # openai<1.0.0
    _USE_NEW_OPENAI_SDK = False


class ErnieAirSim:
    """
    保留原类名，避免你项目里其它 import 需要改。
    功能：用 AirSimWrapper 做感知，把观测喂给 DeepSeek，让模型生成 aw.* 控制代码并执行。
    """

    def __init__(self, system_prompts="system_prompts/airsim_basic_cn.txt",
                 prompt="prompts/airsim_basic_cn.txt",
                 example_msg=None):
        # 系统提示/知识提示（保持你原本结构；若你不需要也可以留空文件）
        self.sysprompt = open(system_prompts, "r", encoding="utf-8").read()
        self.knowledge_prompt = open(prompt, "r", encoding="utf-8").read()

        self.chat_history = []
        if example_msg:
            self.chat_history.extend(example_msg)

        # DeepSeek Client
        if _USE_NEW_OPENAI_SDK:
            self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        else:
            openai.api_key = DEEPSEEK_API_KEY
            openai.api_base = DEEPSEEK_BASE_URL
            self.client = openai

        # AirSim wrapper（包含 RAM + GroundingDINO + AirSim 控制封装）
        self.aw = AirSimWrapper()

        # 先把“知识库 prompt”塞进对话历史（可选但你原逻辑如此）
        if self.knowledge_prompt.strip():
            self.ask(self.knowledge_prompt)

    def ask(self, user_prompt: str) -> str:
        """
        多轮对话：把 system + history 一起发给 DeepSeek
        """
        self.chat_history.append({"role": "user", "content": user_prompt})
        messages = [{"role": "system", "content": self.sysprompt}] + self.chat_history

        if _USE_NEW_OPENAI_SDK:
            completion = self.client.chat.completions.create(
                model=MY_MODEL,
                messages=messages,
                # deepseek-chat 下 temperature/top_p 可写但可能被忽略；写了不影响兼容
                temperature=0,
                top_p=1,
                stream=False,
            )
            assistant_text = completion.choices[0].message.content
        else:
            completion = self.client.ChatCompletion.create(
                model=MY_MODEL,
                messages=messages,
                temperature=0,
                top_p=1,
                stream=False,
            )
            assistant_text = completion["choices"][0]["message"]["content"]

        self.chat_history.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    def extract_python_code(self, content: str):
        """
        从模型回复中抽取 ```python ... ``` 代码块
        """
        # 优先匹配 ```python
        m = re.search(r"```python\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()

        # 退化：匹配任意 ``` ... ```
        m = re.search(r"```(.*?)```", content, re.DOTALL)
        if m:
            code = m.group(1)
            if code.lstrip().startswith("python"):
                code = code.lstrip()[6:]
            return code.strip()

        return None

    def perceive(self, save_box_img_path="boxed.jpg"):
        """
        感知：取图 -> RAM 标签 -> GroundingDINO 框定
        返回：
          - obj_name_list: RAM 输出的目标名称列表
          - ob_for_llm: [(obj_name, distance, angle_deg), ...] 给 LLM 推理用
        """
        img = self.aw.get_image()

        # 1) RAM：识别图中可能存在的目标类别名
        obj_name_list = self.aw.get_objects(img)

        # 2) GroundingDINO：根据目标名做框定，并输出给 LLM 的观测结果
        phrases, final_obj_list, annotated_frame = self.aw.ob_objects(obj_name_list)

        # 保存框定图（满足“识别和框定”）
        if annotated_frame is not None:
            cv2.imwrite(save_box_img_path, annotated_frame)

        # wrapper 已经提供 llm 友好的简化观测
        ob_for_llm = self.aw.ob_objects_llm(obj_name_list)
        return obj_name_list, ob_for_llm

    def build_task_prompt(self, instruction: str, ob_for_llm):
        """
        把观测 + 任务指令组织成给 DeepSeek 的 prompt，并约束其按 wrapper 规范输出代码。
        """
        drone_pose = self.aw.get_drone_position()

        # 关键约束：只允许生成 aw.* 调用；必须放在 python 代码块里
        prompt = f"""
当前无人机位姿（x,y,z,yaw_degree）: {drone_pose}

当前视觉观测（目标, 距离, 偏航角偏差degree）:
{ob_for_llm}

任务指令：
{instruction}

要求：
1) 只输出一段可执行的 Python 代码，并放在 ```python 代码块``` 中。
2) 代码中只能使用变量 aw（AirSimWrapper实例）以及其方法：
   aw.takeoff(), aw.land(), aw.get_drone_position(), aw.fly_to(), aw.fly_path(),
   aw.set_yaw(), aw.get_yaw(), aw.turn(), aw.move(), aw.turn_left(), aw.turn_right(),
   aw.forward(), aw.get_distance(), aw.look_at(), aw.get_image(), aw.get_objects(),
   aw.ob_objects_llm()
3) 不要 import，不要定义新的类，不要访问外部文件。
4) 代码尽量短，完成任务即可。
"""
        return prompt.strip()

    def run(self, instruction: str, save_box_img_path="boxed.jpg"):
        """
        一次闭环：感知 -> 规划(生成代码) -> 执行
        """
        _, ob_for_llm = self.perceive(save_box_img_path=save_box_img_path)
        task_prompt = self.build_task_prompt(instruction, ob_for_llm)

        response = self.ask(task_prompt)
        python_code = self.extract_python_code(response)

        if not python_code:
            raise RuntimeError("模型未返回代码块，无法执行。")

        # 只把 aw 暴露给 exec，符合 wrapper 规范使用
        exec_globals = {"aw": self.aw}
        exec(python_code, exec_globals, {})

        return python_code, response


if __name__ == "__main__":
    # 简单演示：输入中文任务指令
    ernie_airsim = ErnieAirSim()
    cmd = input("请输入无人机任务指令：").strip()
    code, resp = ernie_airsim.run(cmd, save_box_img_path="boxed.jpg")
    print("\n====== 生成的执行代码 ======\n", code)
    print("\n====== 模型原始回复 ======\n", resp)
