import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import torch


def visualize_3d_feat_map(feature_map: torch.Tensor, 
                            channels_per_row: int = 4,
                            cmap: str = 'viridis',
                            projection: str = 'slice',
                            tb: bool = False,
                            ):
    """
    可视化3D特征图的函数，支持正交多平面重建
    
    Args:
        feature_map (torch.Tensor): 5D张量 (batch, channels, depth, height, width)
        channels_per_row (int): 每行显示的通道数，默认4
        cmap (str): 颜色映射方案，默认'viridis'
        projection (str): 投影方式，可选 ['mpr'(三视图)|'max'|'mean'|'slice']
    
    示例用法：
        feature_3d = torch.randn(1, 16, 64, 64, 64)  # 模拟3D特征图
        visualize_3d_feature_map(feature_3d, channels_per_row=4, projection='mpr')
    """
    assert projection in ['mpr', 'max', 'mean', 'slice'], "Invalid projection type"
    batch, num_channels, D, H, W = feature_map.shape
    
    # 获取第一个样本的特征图
    feat = feature_map[0].to(torch.float32).detach().cpu()
    
    # 计算显示参数
    num_rows = math.ceil(num_channels / channels_per_row)
    fig_width = channels_per_row * 6  # 每通道宽度增加
    fig_height = num_rows * 6         # 保持宽高比
    
    if projection == 'mpr':
        fig_width *= 3  # 三视图需要横向三倍空间
        col_mul = 3     # 每通道占3列
    else:
        col_mul = 1
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    for ch_idx in range(num_channels):
        vol = feat[ch_idx]  # 获取当前通道的3D特征体
        
        # 计算子图位置
        row = ch_idx // channels_per_row
        col = ch_idx % channels_per_row
        
        if projection == 'mpr':
            # 多平面重建显示三视图
            slices = [
                vol[D//2, :, :],  # 轴向 (Axial)
                vol[:, H//2, :],  # 矢状 (Sagittal)
                vol[:, :, W//2]   # 冠状 (Coronal)
            ]
            titles = ['Axial', 'Sagittal', 'Coronal']
            
            for i, (slice_img, title) in enumerate(zip(slices, titles)):
                ax = plt.subplot(num_rows, channels_per_row*col_mul, 
                                row*channels_per_row*col_mul + col*col_mul + i + 1)
                ax.imshow(slice_img, cmap=cmap, origin='lower')
                ax.set_title(f"Ch{ch_idx} {title}", fontsize=8)
                ax.axis('off')
                
        elif projection == 'max':
            # 最大强度投影
            ax = plt.subplot(num_rows, channels_per_row, ch_idx+1)
            ax.imshow(vol.max(dim=0)[0], cmap=cmap)
            ax.set_title(f"Ch{ch_idx} MIP", fontsize=8)
            ax.axis('off')
            
        elif projection == 'mean':
            # 平均投影
            ax = plt.subplot(num_rows, channels_per_row, ch_idx+1)
            ax.imshow(vol.mean(dim=0), cmap=cmap)
            ax.set_title(f"Ch{ch_idx} Avg", fontsize=8)
            ax.axis('off')
            
        elif projection == 'slice':
            # 显示中间切片
            ax = plt.subplot(num_rows, channels_per_row, ch_idx+1)
            ax.imshow(vol[D//2], cmap=cmap)
            ax.set_title(f"Ch{ch_idx} Mid", fontsize=8)
            ax.axis('off')
    
    plt.tight_layout()
    if tb:
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1]+(3,))
        plt.close(fig)
        return torch.from_numpy(img.transpose(2, 0, 1))
    else:
        plt.savefig('3d_feats.png', bbox_inches='tight')
        plt.close(fig)
    