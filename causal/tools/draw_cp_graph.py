import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 新数据（来自 causal/results/n3_all_observations_qwen.json 与 n3_all_observations_qwencausal.json 的 statistics.mean/std）
metrics = ['Valid Rate', 'Novelty Rate', 'Recovery Rate']

# qwen（baseline）: mean
qwen_means = [
    0.10611111111111111,  # valid_rate.mean
    0.6827777777777778,   # novelty_rate.mean
    0.03166666666666666   # recovery_rate.mean
]

# qwencausal（增强版）: mean
qwencausal_means = [
    0.4933333333333333,   # valid_rate.mean
    0.9366666666666668,   # novelty_rate.mean
    0.45499999999999996   # recovery_rate.mean
]

# 标准差（std）
std_qwen = [
    0.25467784055377846,  # valid_rate.std
    0.32777259882914234,  # novelty_rate.std
    0.07519727142409907   # recovery_rate.std
]

std_qwencausal = [
    0.40275900339385945,  # valid_rate.std
    0.14884742374510737,  # novelty_rate.std
    0.39607215017992375   # recovery_rate.std
]

# 创建子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 柱状图
x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, qwen_means, width, label='qwen',
                color='#1f77b4', alpha=0.8, yerr=std_qwen, capsize=5, error_kw={'elinewidth': 1, 'markeredgewidth': 1})
bars2 = ax1.bar(x + width/2, qwencausal_means, width, label='qwencausal',
                color='#ff7f0e', alpha=0.8, yerr=std_qwencausal, capsize=5, error_kw={'elinewidth': 1, 'markeredgewidth': 1})

ax1.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance Comparison\n(Bar Chart)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.2)  # 调整y轴范围以适应新数据

# 在柱子上添加数值
for bar, value in zip(bars1, qwen_means):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

for bar, value in zip(bars2, qwencausal_means):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 雷达图
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # 闭合雷达图

qwen_radar = qwen_means + [qwen_means[0]]
qwencausal_radar = qwencausal_means + [qwencausal_means[0]]
angles_plot = angles

ax2 = plt.subplot(122, polar=True)
ax2.plot(angles_plot, qwen_radar, 'o-', linewidth=2, label='qwen',
         color='#1f77b4', markersize=6)
ax2.fill(angles_plot, qwen_radar, alpha=0.3, color='#1f77b4')

ax2.plot(angles_plot, qwencausal_radar, 'o-', linewidth=2, label='qwencausal',
         color='#ff7f0e', markersize=6)
ax2.fill(angles_plot, qwencausal_radar, alpha=0.3, color='#ff7f0e')

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(metrics, fontsize=10)
ax2.set_ylim(0, 1.0)
ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax2.grid(True)
ax2.set_title('Model Performance Comparison\n(Radar Chart)', fontsize=14, fontweight='bold', pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# 调整布局
plt.tight_layout()

# 添加总体标题
fig.suptitle('Qwen vs QwenCausal Performance Comparison', fontsize=16, fontweight='bold', y=1.02)

plt.show()

# 打印关键观察结果
print("\n关键观察结果:")
print(f"1. Valid Rate: qwen2.5-tony 显著更好 ({qwencausal_means[0]:.3f} vs {qwen_means[0]:.3f})")
print(f"2. Novelty Rate: qwencausal 显著更好 ({qwencausal_means[1]:.3f} vs {qwen_means[1]:.3f})")
print(f"3. Recovery Rate: qwencausal 显著更好 ({qwencausal_means[2]:.3f} vs {qwen_means[2]:.3f})")
print("\n性能提升倍数:")
print(f"Valid Rate: {qwencausal_means[0]/qwen_means[0]:.1f}x")
print(f"Novelty Rate: {qwencausal_means[1]/qwen_means[1]:.1f}x")
print(f"Recovery Rate: {qwencausal_means[2]/qwen_means[2]:.1f}x")