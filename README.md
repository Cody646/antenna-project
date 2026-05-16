# antenna-project
本项目旨在复现发表于 IEEE Transactions on Antennas and Propagation (2023) 的研究成果。

## 📖 论文信息
- **标题**: Single-Mode Dual-Band Patch Antenna Based on the Lorentz Model of Dispersive Metamaterials
- **作者**: Jiexi Yin, Zhi Ning Chen, Hanyang Wang, Qun Lou, and Ben Lai
- **期刊**: IEEE Transactions on Antennas and Propagation, Vol. 71, No. 11, Nov 2023
- **核心创新**: 利用洛伦兹色散材料在不同频率下的折射率差异，实现在单一电气尺寸下（TM10 模）同时工作于 2.6 GHz 和 4.1 GHz。

## 👥 团队成员与分工 (Co-Authors)
本复现项目由以下三位小组成员共同完成：

## 🛠 仿真与实验环境
- **电磁仿真**:  HFSS
- **数值计算**: MATLAB 
- **超材料单元**: 分裂环谐振器 (SRR)

## 📂 项目结构
```text
├── Theory/           # 洛伦兹模型推导及数学计算 (MATLAB)
├── Simulation/       # CST 仿真模型文件 (.cst)
├── Results/          # 提取的 S参数、增益、方向图数据
└── README.md         # 项目说明文档
