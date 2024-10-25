import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from conformer import Conformer
import os
import datasets.transforms as T

model = Conformer(patch_size=16, channel_ratio=2, num_med_block=4, embed_dim=384, img_size=512, num_frames=7,
                  attention_type='divided_space_time',
                  depth=21, num_heads=4, mlp_ratio=2, qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.0,
                  drop_path_rate=0.1)
print(model)
device = torch.device("cuda")
model.to(device)

checkpoint = torch.load("/home/cbl/Ceusformer/512size_7frames_21depth.pth", map_location='cpu')
missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=True)
unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
if len(missing_keys) > 0:
    print('Missing Keys: {}'.format(missing_keys))
elif len(unexpected_keys) > 0:
    print('Unexpected Keys: {}'.format(unexpected_keys))
else:
    print("Model Loaded ~")

model.eval()
root = "/home/cbl/Ceusformer/ceus_data/rawframes"

def img_resize(img):
    img = img.resize((1100, 582))
    img = img.crop((550, 0, 1100, 582))  # (0, 0, 550, 582)
    img = img.resize((512, 512))
    return img

# 图像预处理
def preprocess_image(image_path):
    image_path = root + "/" + image_path
    path_name_id = int(image_path[-6:-4])
    path_filetyp = image_path[-4:]
    path_dirname = os.path.dirname(image_path)

    supp_frms_id = [path_name_id - 7, path_name_id - 5, path_name_id - 3, path_name_id + 3, path_name_id + 5,
                    path_name_id + 7]  # pre and after
    supp_frms = [
        img_resize(Image.open(os.path.join(f'{path_dirname}/{i:02d}{path_filetyp}')).convert('RGB'))
        for i in supp_frms_id
    ]

    imgs = []
    image = Image.open(image_path).convert("RGB")
    image = image.resize((1100, 582))
    img1 = image.crop((550, 0, 1100, 582))  # (0, 0, 550, 582)
    img1 = img1.resize((512, 512))
    imgs.append(img1)

    img2 = image.crop((0, 0, 550, 582))  # (0, 0, 550, 582)
    img2 = img2.resize((512, 512))
    imgs.append(img2)

    l = int(len(supp_frms) / 2)
    imgs = supp_frms[0:l] + imgs[0:1] + supp_frms[l:] + imgs[-1:]

    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    imgs, target = preprocess(imgs)  # 添加批次维度
    return imgs

# Grad-CAM 实现
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.feature_maps = None

        # 注册钩子
        self.hook = self.target_layer.register_forward_hook(self.save_feature_maps)
        self.hook_backward = self.target_layer.register_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate_cam(self):
        # 计算权重
        weights = F.adaptive_avg_pool2d(self.gradients, 1)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)

        # 应用 ReLU 激活
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().detach().numpy()

        # 归一化到 [0, 1]
        cam -= cam.min()
        cam /= cam.max()
        return cam

# 主函数
def main(image_path, target_label):
    input_image = preprocess_image(image_path).unsqueeze(0).to(device)

    # 选择目标层（倒数第二个卷积层）
    target_layer = model.conv_trans_21.fusion_block  # 根据你的模型选择合适的层
    grad_cam = GradCAM(model, target_layer)

    # 前向传播
    output_dict = model(input_image)
    output = output_dict['label']  # 从输出字典获取 label

    # 反向传播以计算梯度
    model.zero_grad()
    loss = output[0, target_label]
    loss.backward()

    # 生成 Grad-CAM
    cam = grad_cam.generate_cam()

    # 将 CAM 叠加到原始图像上
    cam = cv2.resize(cam, (512, 512))
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 读取原始图像并进行处理
    original_image = Image.open(root + "/" + image_path)
    original_image = original_image.resize((1100, 582))
    original_image = original_image.crop((550, 0, 1100, 582))
    # original_image = original_image.crop((0, 0, 550, 582))
    original_image = original_image.resize((512, 512))
    original_image = cv2.cvtColor(np.asarray(original_image),cv2.COLOR_RGB2BGR)
    original_image = cv2.resize(original_image, (512, 512))

    # 将热图叠加到原始图像
    superimposed_img = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)

    # 显示结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title("Superimposed Image")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    image_path = "benign/14/202112101143320016BREAST/10.jpg"  # 替换为你的图像路径
    # image_path = "malignant/77/202203110915470010BREAST/10.jpg"
    target_label = 0  # 选择要可视化的标签（根据模型输出调整）
    main(image_path, target_label)
