import pybullet as p
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation    # Для анимации, если хочется попробовать - надо раскоментить
import pandas as pd


dt = 1/240
maxTime = 5.0
logTime = np.arange(0.0, maxTime, dt)
sz = len(logTime)

logX = np.zeros(sz)
logY = np.zeros(sz)
logZ = np.zeros(sz)
logVx = np.zeros(sz)
logVy = np.zeros(sz)
logVz = np.zeros(sz)
logJointAngles = np.zeros((sz, 3))
logJointVelocities = np.zeros((sz, 3))
logError = np.zeros(sz)


targetPos = [0.5, 0.3, 0.8]
kp = 2.0
damping = 0.05

physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -10)

script_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(script_dir, "first_task.urdf")
robotId = p.loadURDF(urdf_path, useFixedBase=True)

jointIndices = []
jointNames = []
for i in range(p.getNumJoints(robotId)):
    jointInfo = p.getJointInfo(robotId, i)
    jointType = jointInfo[2]
    if jointType != p.JOINT_FIXED:
        jointIndices.append(i)
        jointNames.append(jointInfo[1].decode('utf-8'))

eef_link_index = -1
for i in range(p.getNumJoints(robotId)):
    jointInfo = p.getJointInfo(robotId, i)
    child_link_name = jointInfo[12].decode('utf-8')
    if child_link_name == "link_eef3":
        eef_link_index = i
        break

if eef_link_index == -1:
    eef_link_index = p.getNumJoints(robotId) - 1

p.setJointMotorControlArray(
    bodyIndex=robotId,
    jointIndices=jointIndices,
    targetPositions=[0.0, 0.0, 0.0],
    controlMode=p.POSITION_CONTROL,
    forces=[50, 50, 50]
)

for _ in range(200):
    p.stepSimulation()

for idx in range(sz):
    jointStates = p.getJointStates(robotId, jointIndices)
    jointAngles = [state[0] for state in jointStates]
    jointVels = [state[1] for state in jointStates]
    
    logJointAngles[idx] = jointAngles
    logJointVelocities[idx] = jointVels
    
    linkState = p.getLinkState(robotId, eef_link_index, computeLinkVelocity=1)
    currentPos = linkState[0]
    # print(currentPos, '!!!!')
    currentVel = linkState[6]
    logX[idx] = currentPos[0]
    logY[idx] = currentPos[1]
    logZ[idx] = currentPos[2]

    logVx[idx] = currentVel[0]
    logVy[idx] = currentVel[1]
    logVz[idx] = currentVel[2]
    error = np.array(targetPos) - np.array(currentPos)
    logError[idx] = np.linalg.norm(error)
    
    if idx == 0:
        startPos = list(currentPos)
        print(f"Начальная позиция - {startPos}")
    
    # Рассчет Якобиана
    zeroVec = [0, 0, 0]
    jac_t, jac_r = p.calculateJacobian(
        robotId,
        eef_link_index,
        [0, 0, 0],
        jointAngles,
        zeroVec,
        zeroVec
    )
    
    jac_t = np.array(jac_t)[:3, :]
    
    singular_values = np.linalg.svd(jac_t, compute_uv=False)
    min_sv = np.min(singular_values)
    
    adaptive_damping = damping
    if min_sv < 0.01:
        adaptive_damping = 0.1
        if idx % 50 == 0:
            print(f"Близко к сингулярности на шаге {idx}")
    
    # Псевдообратный Якобиан
    jac_pinv = np.linalg.pinv(jac_t.T @ jac_t + adaptive_damping * np.eye(3)) @ jac_t.T
    q_dot = jac_pinv @ (kp * error)
    
    max_velocity = 1.5
    velocity_norm = np.linalg.norm(q_dot)
    if velocity_norm > max_velocity:
        q_dot = q_dot / velocity_norm * max_velocity
    
    p.setJointMotorControlArray(
        bodyIndex=robotId,
        jointIndices=jointIndices,
        targetVelocities=q_dot.tolist(),
        controlMode=p.VELOCITY_CONTROL,
        forces=[30, 30, 30]
    )
    
    p.stepSimulation()
    
    if idx == sz - 1:
        endPos = list(currentPos)
        print(f"Конечная позиция: {endPos}")
        print(f"Финальная ошибка: {logError[-1]}")
    
    time.sleep(dt/2)

p.disconnect()

fig = plt.figure(figsize=(20, 12))

ax1 = plt.subplot(3, 4, 1)
ax1.plot(logTime, logX, 'b-', linewidth=2, label='X')
ax1.axhline(y=targetPos[0], color='r', linestyle='--', linewidth=1.5, label='Цель X')
ax1.set_xlabel('Время')
ax1.set_ylabel('X')
ax1.set_title('По оси X')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(3, 4, 2)
ax2.plot(logTime, logY, 'g-', linewidth=2, label='Y')
ax2.axhline(y=targetPos[1], color='r', linestyle='--', linewidth=1.5, label='Цель Y')
ax2.set_xlabel('Время')
ax2.set_ylabel('Y')
ax2.set_title('По оси Y')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(3, 4, 3)
ax3.plot(logTime, logZ, 'm-', linewidth=2, label='Z')
ax3.axhline(y=targetPos[2], color='r', linestyle='--', linewidth=1.5, label='Цель Z')
ax3.set_xlabel('Время')
ax3.set_ylabel('Z')
ax3.set_title('По оси Z')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Скорости
ax4 = plt.subplot(3, 4, 4)
ax4.plot(logTime, logVx, 'b-', linewidth=1.5, alpha=0.7, label='Vx')
ax4.plot(logTime, logVy, 'g-', linewidth=1.5, alpha=0.7, label='Vy')
ax4.plot(logTime, logVz, 'm-', linewidth=1.5, alpha=0.7, label='Vz')
ax4.set_xlabel('Время')
ax4.set_ylabel('Скорость')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, linestyle='-', linewidth=0.5)

# Углы
ax5 = plt.subplot(3, 4, 5)
colors = ['r', 'g', 'b']
for i in range(3):
    ax5.plot(logTime, logJointAngles[:, i], color=colors[i], linewidth=2, 
             label=f'Сустав {i+1} ({jointNames[i]})')
ax5.set_xlabel('Время')
ax5.set_ylabel('Угол')
ax5.set_title('Углы джоинтов')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Скорости джоинтов
ax6 = plt.subplot(3, 4, 6)
for i in range(3):
    ax6.plot(logTime, logJointVelocities[:, i], color=colors[i], linewidth=1.5, 
             alpha=0.7, label=f'Скорость {i+1}')
ax6.set_xlabel('Время')
ax6.set_ylabel('Скорость')
ax6.set_title('Скорости звеньев')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.axhline(y=0, linestyle='-', linewidth=0.5)

# Фазовый портрет скорости
ax8 = plt.subplot(3, 4, 8)
speed = np.sqrt(logVx**2 + logVy**2 + logVz**2)
ax8.plot(logTime, speed, 'purple', linewidth=2)
ax8.set_xlabel('Время')
ax8.set_ylabel('Скорость')
ax8.set_title('Общая скорость')
ax8.grid(True, alpha=0.3)

# 3D траектория
ax9 = plt.subplot(3, 4, (9, 12), projection='3d')
scatter = ax9.scatter(logX, logY, logZ, c=logTime, cmap='viridis', 
                     s=10, alpha=0.6, label='Траектория')
ax9.scatter([startPos[0]], [startPos[1]], [startPos[2]], 
           c='green', marker='o', s=150, label='Начало', 
           edgecolors='black', linewidth=2, zorder=5)
ax9.scatter([targetPos[0]], [targetPos[1]], [targetPos[2]], 
           c='red', marker='X', s=200, label='Цель', 
           edgecolors='black', linewidth=2, zorder=5)
ax9.scatter([endPos[0]], [endPos[1]], [endPos[2]], 
           c='blue', marker='s', s=150, label='Конец', 
           edgecolors='black', linewidth=2, zorder=5)

# # Вектор от начальной к целевой точке
# ax9.quiver(startPos[0], startPos[1], startPos[2],
#           targetPos[0]-startPos[0], targetPos[1]-startPos[1], targetPos[2]-startPos[2],
#           color='orange', arrow_length_ratio=0.1, linewidth=2, alpha=0.5,
#           label='Направление к цели')

ax9.set_xlabel('X', fontsize=10)
ax9.set_ylabel('Y', fontsize=10)
ax9.set_zlabel('Z', fontsize=10)
ax9.set_title('Траектория движения в пространстве', fontsize=12, fontweight='bold')
ax9.legend(fontsize=9)

cbar = plt.colorbar(scatter, ax=ax9, shrink=0.6)
cbar.set_label('Время', fontsize=9)

max_range = max([
    max(logX)-min(logX), 
    max(logY)-min(logY), 
    max(logZ)-min(logZ)
])
mid_x = (max(logX)+min(logX)) * 0.5
mid_y = (max(logY)+min(logY)) * 0.5
mid_z = (max(logZ)+min(logZ)) * 0.5
ax9.set_xlim(mid_x - max_range/1.5, mid_x + max_range/1.5)
ax9.set_ylim(mid_y - max_range/1.5, mid_y + max_range/1.5)
ax9.set_zlim(mid_z - max_range/1.5, mid_z + max_range/1.5)

ax9.view_init(elev=25, azim=45)

plt.tight_layout()
plt.suptitle(f'Движение 3-х звенного манипулятора\nЦелевая позиция: {targetPos}', 
             fontsize=14, fontweight='bold', y=1.02)
plt.show()

# # Просто приколюха, работает через раз
# fig_anim, ax_anim = plt.subplots(figsize=(10, 6))
# ax_anim.set_xlim(min(logX) - 0.1, max(logX) + 0.1)
# ax_anim.set_ylim(min(logY) - 0.1, max(logY) + 0.1)

# line, = ax_anim.plot([], [], 'b-', linewidth=2, alpha=0.7)
# point, = ax_anim.plot([], [], 'ro', markersize=8)
# target_point, = ax_anim.plot([targetPos[0]], [targetPos[1]], 'gX', markersize=12, 
#                             markeredgewidth=2, label='Цель')
# start_point, = ax_anim.plot([startPos[0]], [startPos[1]], 'gs', markersize=10, 
#                            markeredgewidth=2, label='Старт')
# time_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes, fontsize=12,
#                         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
# error_text = ax_anim.text(0.02, 0.88, '', transform=ax_anim.transAxes, fontsize=10,
#                          bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7))

# ax_anim.set_xlabel('X')
# ax_anim.set_ylabel('Y')
# ax_anim.set_title('Движение в плоскости XY')
# ax_anim.grid(True, alpha=0.3)
# ax_anim.legend()

# def init_anim():
#     line.set_data([], [])
#     point.set_data([], [])
#     time_text.set_text('')
#     error_text.set_text('')
#     return line, point, time_text, error_text

# def update_anim(frame):
#     start_idx = max(0, frame - 50)
#     line.set_data(logX[start_idx:frame+1], logY[start_idx:frame+1])
#     point.set_data([logX[frame]], [logY[frame]])
    
#     current_error = np.sqrt((logX[frame] - targetPos[0])**2 + 
#                            (logY[frame] - targetPos[1])**2 + 
#                            (logZ[frame] - targetPos[2])**2)
    
#     time_text.set_text(f'Время: {logTime[frame]} с')
#     error_text.set_text(f'Ошибка: {current_error} м')
    
#     return line, point, time_text, error_text


# anim = FuncAnimation(fig_anim, update_anim, frames=range(0, sz, 10),
#                     init_func=init_anim, blit=True, interval=50, repeat=False)
# plt.show()

stats_data = {
    'Параметр': ['Начальная X', 'Начальная Y', 'Начальная Z',
                 'Конечная X', 'Конечная Y', 'Конечная Z',
                 'Целевая X', 'Целевая Y', 'Целевая Z',
                 'Макс. ошибка', 'Мин. ошибка', 'Ср. ошибка',
                 'Макс. скорость', 'Ср. скорость', 'Время движения'],
    'Значение': [
        f"{startPos[0]} м", f"{startPos[1]} м", f"{startPos[2]} м",
        f"{endPos[0]} м", f"{endPos[1]} м", f"{endPos[2]} м",
        f"{targetPos[0]} м", f"{targetPos[1]} м", f"{targetPos[2]} м",
        f"{np.max(logError)} м", f"{np.min(logError)} м", f"{np.mean(logError)} м",
        f"{np.max(speed)} м/с", f"{np.mean(speed)} м/с", f"{maxTime:.2f} с"
    ]
}

df_stats = pd.DataFrame(stats_data)
print(df_stats)

# Выводы
final_error = np.linalg.norm(np.array(targetPos) - np.array(endPos))
if final_error < 0.01:
    print("Какая ты умничка")
elif final_error < 0.05:
    print("Все ок")
else:
    print("Цель не достигнута")
    print("Увеличь коэффициент kp, чтоб быстрее сходилось или уменьши демпфирование")

if np.max(speed) > 2.0:
    print("Превышена максимально допустимая скорость\nУменьши макс допустиму скорость джоинтов или добавь ограничения на ускорение")
else:
    print("Все ок")

if np.mean(logError[-100:]) < 0.02:
    print("Все ок")
else:
    print("Колебания возле цели")
data_dict = {
    'time': logTime,
    'x': logX, 'y': logY, 'z': logZ,
    'Vx': logVx, 'Vy': logVy, 'Vz': logVz,
    'error': logError,
    'Звено 1 угол': logJointAngles[:, 0],
    'Звено 2 угол': logJointAngles[:, 1],
    'Звено 3 угол': logJointAngles[:, 2],
    'Звено 1 велосити': logJointVelocities[:, 0],
    'Звено 2 велосити': logJointVelocities[:, 1],
    'Звено 3 велосити': logJointVelocities[:, 2]
}

df_data = pd.DataFrame(data_dict)
df_data.to_excel('manipulator_data.xlsx', index=False)
print("Мучения закончены!")