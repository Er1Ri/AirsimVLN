import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import time
import torch
import numpy as np
from PIL import Image
from transformers import pipeline
import cosysairsim as airsim


class AirSimVisionAI:
    def __init__(self):
        # 1. 初始化深度估计模型 (针对 4GB 显存优化)
        print("正在加载深度估计模型...")
        checkpoint = "vinvino02/glpn-nyu"
        self.depth_estimator = pipeline(
            "depth-estimation",
            model=checkpoint,
            device=0,  # 使用 GPU
            model_kwargs={"torch_dtype": torch.float16}  # 半精度减少显存占用
        )

        # 2. 连接到 AirSim (Blocks 环境)
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        print("已成功连接到 AirSim")

    def get_and_analyze(self):
        try:
            # 3. 从 AirSim 获取彩色图像
            # 注意：此处使用 '0' 号摄像头，Scene 场景图
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])
            response = responses[0]

            # 将字节流转换为 PIL Image
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)
            raw_image = Image.fromarray(img_rgb)

            # 针对 4GB 显存：如果图像过大，强制缩小分辨率
            #if raw_image.width > 640:
            #    raw_image = raw_image.resize((640, 480))

            raw_image.save("airsim_current.png")
            print("AirSim 图像已捕获并保存")

            # 4. 执行深度估计
            print("正在进行深度分析...")
            with torch.no_grad():  # 禁用梯度，节省内存
                result = self.depth_estimator(raw_image)

            # 5. 保存并显示结果
            depth_map = result["depth"]
            depth_map.save("airsim_depth_result.png")

            # 获取预测的原始深度数组进行逻辑判断
            predicted_depth = result["predicted_depth"].numpy()
            avg_dist = predicted_depth.mean()
            print(f"分析完成！当前前方平均距离指标: {avg_dist:.2f}")

        except Exception as e:
            print(f"运行出错: {e}")
        finally:
            # 释放显存
            torch.cuda.empty_cache()


if __name__ == "__main__":
    ai_vision = AirSimVisionAI()

    # 模拟一个简单的循环
    for i in range(3):
        print(f"\n--- 执行第 {i + 1} 次检查 ---")
        ai_vision.get_and_analyze()
        time.sleep(2)  # 给系统留出喘息时间

    # 释放控制权
    ai_vision.client.enableApiControl(False)