import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

def SystemOfEquations(y, t):
    # y[0]=x, y[1]=phi, y[2]=x', y[3]=phi'
    # yt[0]=x', yt[1]=phi', yt[2]=x'', yt[3]=phi''
    global m1, m2, k, g
    yt = np.zeros_like(y)

    yt[0] = y[2]
    yt[1] = y[3]

    a11 = m1 + m2
    a12 = 0
    a21 = 0
    a22 = m1*y[0]

    b1 = m1*y[0]*y[3]**2-k*y[2]-m2*g
    b2 = -2*m1*y[3]*y[2]-k*y[0]*y[3]

    yt[2] = (b1 * a22 - a12 * b2) / (a11 * a22 - a12 * a21)
    yt[3] = (b2 * a11 - a21 * b1) / (a11 * a22 - a12 * a21)

    return yt

global m1, m2, k, g

m1 = 4
m2 = 1
k = 0
g = 9.81

X0 = 0.5
Phi0 = 0
DX0 = 2
DPhi0 = 5
y0 = [X0, Phi0, DX0, DPhi0]

Tfin = 100
NT = 1000
t = np.linspace(0, Tfin, NT)

Y = odeint(SystemOfEquations, y0, t)

fig = plt.figure(figsize=[10, 10])
ax = fig.add_subplot(1, 1, 1)
ax.set(xlim=[-15, 20], ylim=[-10, 20])

x = Y[:, 0]
phi = Y[:, 1]
Dx = Y[:, 2]
Dphi = Y[:, 3]
DDx = [SystemOfEquations(y, t)[2] for y, t in zip(Y, t)]
DDphi = [SystemOfEquations(y, t)[3] for y, t in zip(Y, t)]
Nt = m1*(x*Dphi**2-DDx)-k*Dx

fig_for_graphs = plt.figure(figsize=[13, 7])
ax_graphs = fig_for_graphs.add_subplot(1, 3, 1)
ax_graphs.plot(t, x, color="blue")
ax_graphs.set_title("x(t)")
ax_graphs.set(xlim=[0, Tfin])
ax_graphs.grid("True")

ax_graphs = fig_for_graphs.add_subplot(1, 3, 2)
ax_graphs.plot(t, phi, color="green")
ax_graphs.set_title("phi(t)")
ax_graphs.set(xlim=[0, Tfin])
ax_graphs.grid("True")

ax_graphs = fig_for_graphs.add_subplot(1, 3, 3)
ax_graphs.plot(t, Nt, color="red")
ax_graphs.set_title("N(t)")
ax_graphs.set(xlim=[0, Tfin])
ax_graphs.grid("True")

Ex = 10; Ey = 0; Fx = 8; Fy = 10
Cx = 25; Cy = 10; Dx = 27; Dy = 0
Ox = 4; Oy = 5; a = 0.3; b = 0.2
betta = np.linspace(0, 6.28, 1000)
X_h = a * np.sin(betta) + Ox
Y_h = b * np.cos(betta) + Oy

l = 10
lOA = x
lOB = l - lOA
Bx = Ox; By = Oy - lOB
Ax = lOA * np.sin(phi) + X_h
Ay = lOA * np.cos(phi) + Y_h

P_X = np.array([Ex - 20, Fx - 10, Cx - 10, Dx - 20, Ex - 20])
P_Y = np.array([Ey, Fy, Cy, Dy, Ey])

OB = ax.plot([Ox, Bx], [Oy, By[0]], 'red', ls='--')[0]
B = ax.plot(Bx, By[0], 'blue', marker='o', ms=17)[0]
Plane = ax.plot(P_X, P_Y, 'black')
O = ax.plot(X_h, Y_h, 'black')
OA = ax.plot([Ox, Ax[0]], [Oy, Ay[0]], 'red')[0]
A = ax.plot(Ax[0], Ay[0], 'purple', marker='o', ms=19)[0]

def Animation(i):
    OB.set_data([Ox, Bx], [Oy, By[i]])
    B.set_data(Bx, By[i])
    OA.set_data([Ox, Ax[i]], [Oy, Ay[i]])
    A.set_data(Ax[i], Ay[i])
    return [OB, B, OA, A]

a = FuncAnimation(fig, Animation, frames=NT, interval=10)

plt.show()