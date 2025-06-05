import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def solve_kepler(M, e, tol=1e-10, max_iter=100):
    E = M if e < 0.8 else np.pi
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        E_new = E - f / f_prime
        if abs(E_new - E) < tol:
            return E_new
        E = E_new
    return E

def true_anomaly(t, P, e, T0):
    M = 2 * np.pi / P * (t - T0)
    M = np.mod(M, 2*np.pi)
    E = np.array([solve_kepler(Mi, e) for Mi in M])
    # 真近点離角θ
    cos_theta = (np.cos(E) - e) / (1 - e * np.cos(E))
    sin_theta = (np.sqrt(1 - e**2) * np.sin(E)) / (1 - e * np.cos(E))
    theta = np.arctan2(sin_theta, cos_theta)
    return theta

def radial_velocity_model(t, P, Kp, e, omega, Vsys, T0):
    theta = true_anomaly(t, P, e, T0)
    # 視線速度モデル
    return Vsys + Kp * (np.cos(theta + omega) + e * np.cos(omega))

def residuals(params, t, v_obs):
    P, Kp, e, omega, Vsys, T0 = params
    v_model = radial_velocity_model(t, P, Kp, e, omega, Vsys, T0)
    return v_obs - v_model

data = np.array([
    [14.75943013143777449, 10.29276526418392912],
    [51.85632248445747194, -3.249059287763180759],
    [63.60438878212471536, -1.700108753335274336],
    [86.34033088432124714, -6.955230249810869125],
    [104.6478998086038672, -5.988878466163186864],
    [279.9123087428177428, 0.4491811842698901769],
    [302.7032161930821985, 5.343830234167499249],
    [309.2680035174004161, 6.473086308887705087],
    [319.4386642217655208, 7.339880010333093452],
    [336.8799344446402415, 12.37654249070353174],
    [380.9492748775522841, 7.930107959072777213],
    [386.0932914196202432, 4.327858815767007705],
    [397.7647235877346930, 1.437625811081091776],
    [400.6338578669470394, 1.084282895994354012],
    [414.6725295985705770, -0.9271728835113239864],
    [440.0172645322999756, -3.695967346343064897],
    [467.1423455690924129, -6.373664267627315638],
    [471.5027025386589230, -6.368450716381578225],
    [522.0882374518662346, -6.976346898853369893],
    [568.0544281933908906, -6.050366999996878548],
    [569.7862986891125274, -6.267611406635054117],
    [577.9592778003451485, -2.262851651405441089],
    [583.3857518782082252, -4.471246878319825591],
    [607.8124872499947742, -3.020041449722365989],
    [635.1088682201780102, -1.646300434931433854],
    [650.9942905709182241, 2.140575039752232289],
    [675.6855459536425315, 3.668134396731464797],
    [689.6083094461962446, 8.258289731291169389],
    [703.4738151657512617, 11.59576134374126077],
    [714.3913898299176708, 16.05189275361949797]
])
t = data[:, 0]
v = data[:, 1]

##Pの初期値を変更
init_params = [
    400.0,       # P [day]
    10.0,        # Kp [cm/s]
    0.1,         # e
    0.0,         # omega [rad]
    np.mean(v),  # Vsys [cm/s]
    t[0]         # T0 [day]
]

result = least_squares(
    residuals, init_params, args=(t, v),
    bounds=([1, 0, 0, -2*np.pi, -np.inf, t[0]-100],
            [1000, 100, 0.99, 2*np.pi, np.inf, t[-1]+100])
)

P_fit, Kp_fit, e_fit, omega_fit, Vsys_fit, T0_fit = result.x
residuals_fit = residuals(result.x, t, v)
sigma_fit = np.std(residuals_fit)

print(f"周期 P = {P_fit:.3f} 日")
print(f"半振幅 Kp = {Kp_fit:.3f} cm/s")
print(f"離心率 e = {e_fit:.3f}")
print(f"近点引数 ω = {omega_fit:.3f} rad")
print(f"重心速度 V_sys = {Vsys_fit:.3f} cm/s")
print(f"近点通過時刻 T0 = {T0_fit:.3f} 日")
print(f"ノイズ標準偏差 σ = {sigma_fit:.3f} cm/s")

t_fit = np.linspace(t.min(), t.max(), 1000)
v_fit = radial_velocity_model(t_fit, *result.x)

plt.figure(figsize=(8,5))
plt.scatter(t, v, label="raw data")
plt.plot(t_fit, v_fit, 'r-', label="fitting model")
plt.xlabel("time [day]")
plt.ylabel("radial velosity [cm/s]")
plt.legend()
plt.show()
