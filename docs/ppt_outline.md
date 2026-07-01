# PPT Outline: 三维重建与高斯绘制

## 1. 任务与输入
- 输入：`数据3-场景.mp4`，从 32.07 秒、30 FPS 视频中抽取多视角帧。
- 目标：无标定条件下恢复相机、点云，并训练可交互渲染的 3D Gaussian Splatting 模型。
- 展示内容：VGGT 初始化、BA 优化、3DGS 训练与实时渲染、改进实验、未来方向。

## 2. 系统流程
- 视频抽帧：质量感知采样或均匀采样，输出 `outputs/scene/images`。
- VGGT：预测相机参数、深度/点图，并导出 COLMAP 格式稀疏模型。
- 自实现 BA：固定内参，优化相机外参与 3D 点坐标，最小化鲁棒重投影误差。
- 3DGS：使用 VGGT-only 和 BA-refined 两套 COLMAP 结果分别训练高斯点云。
- 评估：对比重投影误差、渲染质量、训练耗时、实时 FPS。

## 3. VGGT 初始化
- 使用多视角图像直接预测几何，不需要输入相机标定参数。
- 输出相机外参、内参、初始点云/深度，并转换为 COLMAP `cameras/images/points3D`。
- 展示建议：放 4-6 张输入帧、VGGT 初始点云截图、相机轨迹截图。

## 4. Bundle Adjustment 实现
- 观测来自 COLMAP `images.txt` 中的 2D 点与 `points3D.txt` track。
- 参数化：每个相机优化一个 SE(3) 增量，每个 3D 点优化 XYZ。
- 损失：Huber 重投影误差，过滤 track 太短和无效深度观测。
- 展示建议：BA 前后平均重投影误差表格和相机轨迹变化。

## 5. 3D Gaussian Splatting
- 输入：VGGT-only 或 BA-refined COLMAP 稀疏模型与图像。
- 优化：高斯位置、颜色/SH、不透明度、尺度、旋转，通过可微 rasterization 训练。
- 输出：两个高斯模型 `gaussians_vggt` 和 `gaussians_ba`。
- 展示建议：实时 viewer 截图、旋转视角视频、同一视角的两分支渲染对比。

## 6. BA 对高斯绘制效果分析
- 对比指标：重投影 RMSE、PSNR/SSIM 或 MSE、训练收敛速度、viewer FPS。
- 预期结论：BA 若降低重投影误差，通常能减少高斯漂浮、边缘重影和相机轨迹抖动。
- 需要如实报告：如果 VGGT 初始结果已经很稳定，BA 提升可能较小，甚至因错误 tracks 引入局部退化。

## 7. VGGT 改进实验
- 改进方法：质量感知帧选择，优先保留清晰、曝光正常且时间分布均衡的帧。
- 对照组：均匀采样 48 帧。
- 实验组：从更多候选帧中按清晰度/曝光分数选择 48 帧。
- 报告：VGGT+BA 重投影误差、3DGS 渲染质量、总运行时间。

## 8. 实验结果页模板
| Method | Frames | BA RMSE ↓ | Render PSNR ↑ | Train Time ↓ | FPS ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform + VGGT + 3DGS | 48 | TBD | TBD | TBD | TBD |
| Uniform + VGGT + BA + 3DGS | 48 | TBD | TBD | TBD | TBD |
| Quality + VGGT + BA + 3DGS | 48 | TBD | TBD | TBD | TBD |

## 9. 答辩演示顺序
1. 展示输入视频抽帧结果。
2. 展示 VGGT 初始点云/相机。
3. 运行或展示 BA 日志：RMSE 从初始值下降到最终值。
4. 打开 3DGS viewer，交互旋转场景。
5. 展示 BA 前后与采样改进实验的表格。

## 10. 未来研究方向
- 使用更可靠的跨视角匹配或 VGGT confidence 过滤 BA 观测。
- 联合优化 VGGT 深度、BA 相机与 Gaussian 参数，减少分阶段误差传递。
- 引入动态物体/曝光变化处理，提高视频场景的鲁棒性。
- 使用更快的 splatting trainer 或蒸馏方法提升训练和渲染速度。
