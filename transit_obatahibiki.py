import numpy as np
import matplotlib.pyplot as plt

# --- 各種定数の定義（SI単位） ---
R_sun = 6.957e8           # 太陽半径 [m]
M_sun = 1.989e30          # 太陽質量 [kg]
R_jup = 7.1492e7          # 木星半径 [m]
M_jup = 1.898e27          # 木星質量 [kg]
R_earth = 6.371e6         # 地球半径 [m]
M_earth = 5.972e24        # 地球質量 [kg]
au = 1.496e11             # 1 au [m]
G = 6.67430e-11           # 万有引力定数 [m^3/kg/s^2]
day = 86400               # 1日 [s]

# --- 軌道半径と公転周期 ---
a = 3 * au                                  # 軌道半径 [m]
P = 2 * np.pi * np.sqrt(a**3 / (G * M_sun)) # 公転周期 [s]
P_day = P / day                             # 公転周期 [day]

# --- 観測時間範囲（恒星1個分通過する前後） ---
t_half = P * R_sun / (np.pi * a) / day     # 時間範囲（日）
t = np.linspace(-t_half, t_half, 2000)     # 横軸：日単位時間

# --- 距離d(t)の計算（直線近似：軌道速度一定） ---
v = 2 * np.pi * a / P                      # 軌道速度 [m/s]
d = np.abs(v * t * day)                   # 恒星中心から惑星中心までの距離 [m]

# --- 重なり面積の厳密計算関数 ---
def calc_overlap_area(R_p, d):
    R_s = R_sun
    S = np.zeros_like(d)
    no_overlap = d >= R_s + R_p
    S[no_overlap] = 0
    full_overlap = d <= np.abs(R_s - R_p)
    S[full_overlap] = np.pi * min(R_p, R_s)**2
    partial = ~(no_overlap | full_overlap)
    d_p = d[partial]
    r1, r2 = R_p, R_s
    part1 = r1**2 * np.arccos((d_p**2 + r1**2 - r2**2) / (2 * d_p * r1))
    part2 = r2**2 * np.arccos((d_p**2 + r2**2 - r1**2) / (2 * d_p * r2))
    part3 = 0.5 * np.sqrt(
        (-d_p + r1 + r2) *
        (d_p + r1 - r2) *
        (d_p - r1 + r2) *
        (d_p + r1 + r2)
    )
    S[partial] = part1 + part2 - part3
    return S

# --- 光度曲線を計算 ---
def compute_flux(R_p):
    S = calc_overlap_area(R_p, d)
    return 1 - S / (np.pi * R_sun**2)

flux_a = compute_flux(R_jup)
flux_b = compute_flux(R_earth)

# --- プロット ---
fig, ax1 = plt.subplots(figsize=(10,5))

# 左軸に木星サイズの光度
ax1.plot(t, flux_a, color='tab:blue', label='(a) R=R_J, M=13M_J')
#fontsizeのみ自分で設定
ax1.set_ylabel('Relative Flux (a)', color='tab:blue', fontsize=14)
ax1.tick_params(axis='y', labelcolor='tab:blue')
##軸の設定のし直し
ax1.set_ylim(1 - 1.2*(R_jup/R_sun)**2, 1 + 0.2*(R_jup/R_sun)**2)

# 右軸に地球サイズの光度
ax2 = ax1.twinx()
ax2.plot(t, flux_b, color='tab:green', label='(b) R=R_⊕, M=0.6M_⊙')
#fontsizeのみ自分で設定
ax2.set_ylabel('Relative Flux (b)', color='tab:green', fontsize=14)
ax2.tick_params(axis='y', labelcolor='tab:green')
##軸の設定のし直し
ax2.set_ylim(1 - 1.2*(R_earth/R_sun)**2, 1 + 0.2*(R_earth/R_sun)**2)
##自動でオフセットがついていたため、消去
ax2.get_yaxis().get_major_formatter().set_useOffset(False)

ax1.set_xlabel('Time [day]')
ax1.set_title('Transit Light Curves', fontsize=16)
ax1.grid(True)

fig.tight_layout()
plt.show()