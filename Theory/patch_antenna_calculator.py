# -*- coding: utf-8 -*-
"""
论文: Single-Mode Dual-Band Patch Antenna Based on the Lorentz Model
      of Dispersive Metamaterials (IEEE TAP, 2023)

该脚本实现了论文中基于 Lorentz 色散超材料的单模双频贴片天线尺寸计算。

参考文献:
    [1] J. Yin et al., IEEE Trans. Antennas Propag., vol. 71, no. 11, Nov. 2023.
    [2] J. B. Pendry et al., IEEE Trans. Microw. Theory Techn., vol. 47, no. 11, 1999.

作者: 基于论文 [1] 实现
"""

import sys
import io

# 设置控制台输出编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np

# ============================================================
# 常量
# ============================================================
C0 = 3e8  # 自由空间光速 (m/s)


# ============================================================
# 1. Lorentz 模型 (公式 1-3)
# ============================================================
def lorentz_permittivity(f, fp, f0, gamma0):
    """
    计算 Lorentz 色散模型的复相对介电常数 (公式 1)

    参数:
        f:      频率 (Hz)
        fp:     等离子体频率 (Hz), fp = ωp / (2π)
        f0:     谐振频率 (Hz),     f0 = ω0 / (2π)
        gamma0: 阻尼系数 (s^-1)

    返回:
        epsilon_r: 复相对介电常数
    """
    omega = 2 * np.pi * f
    omega_p = 2 * np.pi * fp
    omega_0 = 2 * np.pi * f0

    epsilon_r = 1 + omega_p**2 / (omega_0**2 - omega**2 - 1j * omega * gamma0)
    return epsilon_r


def lorentz_permeability(f, fp, f0, gamma0):
    """
    计算 Lorentz 色散模型的复相对磁导率 (公式 11)

    参数与 lorentz_permittivity 相同
    """
    return lorentz_permittivity(f, fp, f0, gamma0)


def refractive_index(epsilon_r, mu_r=1.0):
    """计算折射率 n = sqrt(epsilon_r * mu_r)"""
    return np.sqrt(epsilon_r * mu_r)


# ============================================================
# 2. 贴片天线尺寸计算 (公式 10)
# ============================================================
def patch_length(f, epsilon_r, mu_r=1.0):
    """
    计算矩形贴片天线 TM10 模式的长度 (公式 10)

        L ≈ c0 / (2 * f * sqrt(epsilon_r * mu_r))

    参数:
        f:         工作频率 (Hz)
        epsilon_r: 相对介电常数 (实数部分)
        mu_r:      相对磁导率 (实数部分, 默认 1.0)

    返回:
        L: 贴片长度 (m)
    """
    L = C0 / (2 * f * np.sqrt(epsilon_r * mu_r))
    return L


def patch_width(f, epsilon_r):
    """
    计算矩形贴片天线的宽度 (常用经验公式)

        W = c0 / (2 * f) * sqrt(2 / (epsilon_r + 1))

    参数:
        f:         工作频率 (Hz)
        epsilon_r: 相对介电常数

    返回:
        W: 贴片宽度 (m)
    """
    W = C0 / (2 * f) * np.sqrt(2 / (epsilon_r + 1))
    return W


# ============================================================
# 3. 单模双频条件 (公式 8-9)
# ============================================================
def check_dual_band_condition(f1, f2, n1, n2):
    """
    检查是否满足单模双频条件 (公式 9):
        f1 * n(f1) = f2 * n(f2)

    返回:
        ratio: f1*n1 / (f2*n2), 越接近 1 越满足条件
    """
    ratio = (f1 * n1) / (f2 * n2)
    return ratio


def find_f2_for_f1(f1, fp, f0, gamma0, f2_range=(3e9, 6e9), steps=100000):
    """
    给定 f1, 在 f2_range 范围内搜索满足 f1*n(f1) = f2*n(f2) 的 f2

    参数:
        f1:        低频工作频率 (Hz)
        fp, f0, gamma0: Lorentz 模型参数
        f2_range:  搜索范围 (Hz)
        steps:     搜索步数

    返回:
        f2: 满足条件的高频频率 (Hz), 未找到返回 None
    """
    epsilon_r_f1 = lorentz_permittivity(f1, fp, f0, gamma0)
    n1 = np.real(refractive_index(epsilon_r_f1))
    target = f1 * n1

    f2_candidates = np.linspace(f2_range[0], f2_range[1], steps)
    best_f2 = None
    best_diff = np.inf

    for f2 in f2_candidates:
        if f2 <= f0:  # f2 应在谐振频率之上
            continue
        epsilon_r_f2 = lorentz_permittivity(f2, fp, f0, gamma0)
        n2 = np.real(refractive_index(epsilon_r_f2))
        diff = abs(f2 * n2 - target)

        if diff < best_diff:
            best_diff = diff
            best_f2 = f2

    if best_f2 is not None and best_diff / target < 0.05:
        return best_f2
    else:
        return None


# ============================================================
# 4. 损耗计算 (公式 4-5)
# ============================================================
def loss_tangent(epsilon_r):
    """计算损耗角正切 (公式 4): tan(delta) = Im(epsilon_r) / Re(epsilon_r)"""
    return np.imag(epsilon_r) / np.real(epsilon_r)


# ============================================================
# 5. 主程序: 复现论文中的天线设计
# ============================================================
def main():
    print("=" * 70)
    print("论文复现: 基于 Lorentz 色散超材料的单模双频贴片天线")
    print("J. Yin et al., IEEE TAP, vol. 71, no. 11, 2023")
    print("=" * 70)

    # ----------------------------------------------------------
    # 5.1 论文中的设计参数
    # ----------------------------------------------------------
    print("\n" + "-" * 70)
    print("【1】设计参数")
    print("-" * 70)

    # 目标工作频率 (论文 Fig.14: 2.6 GHz 和 4.1 GHz)
    f1_target = 2.6e9  # 低频 (Hz)
    f2_target = 4.1e9  # 高频 (Hz)

    # SRR 超材料参数 (论文 Section III-C)
    # SRR 谐振频率约 3.1 GHz
    f0_srr = 3.1e9   # SRR 谐振频率 (Hz)
    fp_srr = 5.0e9   # 等效等离子体频率 (Hz, 根据论文 Fig.11 估算)
    gamma0_srr = 0.2 # 阻尼系数 (s^-1, 论文中使用的值)

    # 基板参数 (论文 Section IV)
    # F4BM220 材料
    epsilon_r_substrate = 2.2  # 基板相对介电常数

    print(f"  目标低频 f1 = {f1_target/1e9:.2f} GHz")
    print(f"  目标高频 f2 = {f2_target/1e9:.2f} GHz")
    print(f"  SRR 谐振频率 f0 = {f0_srr/1e9:.2f} GHz")
    print(f"  等效等离子体频率 fp = {fp_srr/1e9:.2f} GHz")
    print(f"  阻尼系数 gamma0 = {gamma0_srr}")
    print(f"  基板介电常数 epsilon_r = {epsilon_r_substrate}")

    # ----------------------------------------------------------
    # 5.2 计算 Lorentz 色散曲线
    # ----------------------------------------------------------
    print("\n" + "-" * 70)
    print("【2】Lorentz 色散超材料的有效参数")
    print("-" * 70)

    # 计算两个频率下的有效磁导率 (SRR 提供色散磁导率)
    mu_r_f1 = lorentz_permeability(f1_target, fp_srr, f0_srr, gamma0_srr)
    mu_r_f2 = lorentz_permeability(f2_target, fp_srr, f0_srr, gamma0_srr)

    # 有效介电常数 = 基板介电常数 (SRR 主要影响磁导率)
    eps_eff_f1 = epsilon_r_substrate
    eps_eff_f2 = epsilon_r_substrate

    # 有效折射率
    n1 = np.real(refractive_index(eps_eff_f1, mu_r_f1))
    n2 = np.real(refractive_index(eps_eff_f2, mu_r_f2))

    print(f"\n  在 f1 = {f1_target/1e9:.2f} GHz:")
    print(f"    有效磁导率 Re(mu_r) = {np.real(mu_r_f1):.4f}")
    print(f"    有效磁导率 Im(mu_r) = {np.imag(mu_r_f1):.4f}")
    print(f"    有效折射率 n = {n1:.4f}")

    print(f"\n  在 f2 = {f2_target/1e9:.2f} GHz:")
    print(f"    有效磁导率 Re(mu_r) = {np.real(mu_r_f2):.4f}")
    print(f"    有效磁导率 Im(mu_r) = {np.imag(mu_r_f2):.4f}")
    print(f"    有效折射率 n = {n2:.4f}")

    # ----------------------------------------------------------
    # 5.3 验证单模双频条件 (公式 9)
    # ----------------------------------------------------------
    print("\n" + "-" * 70)
    print("【3】验证单模双频条件 f1 * n(f1) = f2 * n(f2)")
    print("-" * 70)

    ratio = check_dual_band_condition(f1_target, f2_target, n1, n2)
    print(f"\n  f1 * n(f1) = {f1_target/1e9:.2f}G x {n1:.4f} = {f1_target * n1:.2e}")
    print(f"  f2 * n(f2) = {f2_target/1e9:.2f}G x {n2:.4f} = {f2_target * n2:.2e}")
    print(f"  比值 (f1*n1)/(f2*n2) = {ratio:.4f}")

    # 计算两个频率下的波长 (在介质中)
    lambda_f1 = C0 / (f1_target * n1)
    lambda_f2 = C0 / (f2_target * n2)
    print(f"\n  在介质中的波长:")
    print(f"    lambda(f1) = {lambda_f1*1e3:.2f} mm")
    print(f"    lambda(f2) = {lambda_f2*1e3:.2f} mm")
    print(f"    波长差异: {abs(lambda_f1 - lambda_f2)*1e3:.2f} mm "
          f"({abs(lambda_f1 - lambda_f2)/lambda_f1*100:.2f}%)")

    # ----------------------------------------------------------
    # 5.4 计算贴片天线尺寸 (公式 10)
    # ----------------------------------------------------------
    print("\n" + "-" * 70)
    print("【4】贴片天线尺寸计算 (公式 10)")
    print("-" * 70)

    # 使用 f1 计算贴片长度
    L_from_f1 = patch_length(f1_target, eps_eff_f1, np.real(mu_r_f1))

    # 使用 f2 计算贴片长度
    L_from_f2 = patch_length(f2_target, eps_eff_f2, np.real(mu_r_f2))

    # 贴片宽度 (使用 f1 计算)
    W_patch = patch_width(f1_target, eps_eff_f1)

    print(f"\n  基于 f1 = {f1_target/1e9:.2f} GHz 计算:")
    print(f"    L = c0/(2*f1*sqrt(eps_r*mu_r))")
    print(f"    L = {L_from_f1*1e3:.2f} mm  (eps_r={eps_eff_f1}, mu_r={np.real(mu_r_f1):.4f})")

    print(f"\n  基于 f2 = {f2_target/1e9:.2f} GHz 计算:")
    print(f"    L = c0/(2*f2*sqrt(eps_r*mu_r))")
    print(f"    L = {L_from_f2*1e3:.2f} mm  (eps_r={eps_eff_f2}, mu_r={np.real(mu_r_f2):.4f})")

    print(f"\n  贴片宽度 W = {W_patch*1e3:.2f} mm")

    # 论文中的实际尺寸 (Fig.13)
    L_paper = 37.5  # mm
    W_paper = 25.5  # mm
    print(f"\n  论文中优化后的实际尺寸 (Fig.13):")
    print(f"    贴片长度 L_paper = {L_paper:.1f} mm")
    print(f"    贴片宽度 W_paper = {W_paper:.1f} mm")

    # ----------------------------------------------------------
    # 5.5 损耗分析
    # ----------------------------------------------------------
    print("\n" + "-" * 70)
    print("【5】材料损耗分析")
    print("-" * 70)

    tan_delta_f1 = loss_tangent(mu_r_f1)
    tan_delta_f2 = loss_tangent(mu_r_f2)

    print(f"\n  在 f1 = {f1_target/1e9:.2f} GHz:")
    print(f"    损耗角正切 tan(delta) = {tan_delta_f1:.6f}")
    print(f"    辐射效率估算 ≈ {1/(1+tan_delta_f1)*100:.1f}% "
          f"(论文: 98% @ 2.5 GHz)")

    print(f"\n  在 f2 = {f2_target/1e9:.2f} GHz:")
    print(f"    损耗角正切 tan(delta) = {tan_delta_f2:.6f}")
    print(f"    辐射效率估算 ≈ {1/(1+tan_delta_f2)*100:.1f}% "
          f"(论文: 95% @ 4.0 GHz)")

    # ----------------------------------------------------------
    # 5.6 搜索满足条件的 f2
    # ----------------------------------------------------------
    print("\n" + "-" * 70)
    print("【6】搜索满足 f1*n(f1)=f2*n(f2) 的 f2")
    print("-" * 70)

    # 尝试不同的 f1 值
    test_f1s = [2.0e9, 2.2e9, 2.4e9, 2.6e9, 2.8e9, 3.0e9]
    print(f"\n  {'f1 (GHz)':<12} {'f2 (GHz)':<12} {'f2/f1':<10} {'条件比值':<12}")
    print(f"  {'-'*11} {'-'*11} {'-'*9} {'-'*11}")

    for f1_test in test_f1s:
        f2_found = find_f2_for_f1(f1_test, fp_srr, f0_srr, gamma0_srr)
        if f2_found:
            eps1 = lorentz_permittivity(f1_test, fp_srr, f0_srr, gamma0_srr)
            eps2 = lorentz_permittivity(f2_found, fp_srr, f0_srr, gamma0_srr)
            n1_test = np.real(refractive_index(eps1))
            n2_test = np.real(refractive_index(eps2))
            r = check_dual_band_condition(f1_test, f2_found, n1_test, n2_test)
            print(f"  {f1_test/1e9:<12.2f} {f2_found/1e9:<12.2f} "
                  f"{f2_found/f1_test:<10.2f} {r:<12.4f}")
        else:
            print(f"  {f1_test/1e9:<12.2f} {'未找到':<12} {'-':<10} {'-':<12}")

    # ----------------------------------------------------------
    # 5.7 总结
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("设计总结")
    print("=" * 70)
    print(f"""
  工作频率:        f1 = {f1_target/1e9:.1f} GHz, f2 = {f2_target/1e9:.1f} GHz
  SRR 谐振频率:    f0 = {f0_srr/1e9:.1f} GHz (介于 f1 和 f2 之间)
  基板材料:        F4BM220, epsilon_r = {epsilon_r_substrate}

  贴片尺寸 (计算值):
    长度 L1 (基于 f1) = {L_from_f1*1e3:.1f} mm
    长度 L2 (基于 f2) = {L_from_f2*1e3:.1f} mm
    宽度 W           = {W_patch*1e3:.1f} mm

  贴片尺寸 (论文优化值):
    长度 L = {L_paper:.1f} mm
    宽度 W = {W_paper:.1f} mm

  设计原理:
    利用 SRR 超材料在谐振频率附近的 Lorentz 色散特性,
    使低频 (f1) 具有高折射率, 高频 (f2) 具有低折射率,
    满足 f1*n(f1) = f2*n(f2), 从而同一贴片可在两个频率
    以相同的 TM10 模式工作, 实现单模双频。
""")


if __name__ == "__main__":
    main()
