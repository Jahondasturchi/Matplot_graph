import numpy as np
from matplotlib import pyplot as plt, cm

# =====================
# PARAMETERS
# =====================
nx = 41
ny = 41
nt =600
nit = 100

dx = 1.0 / (nx - 1)
dy = 2.0 / (ny - 1)

rho = 1.0
nu  = 0.1
D   = 0.01     # diffusivity
dt  = 0.001

x = np.linspace(0, 1, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

# =====================
# INITIAL FIELDS
# =====================
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

C = np.zeros((ny, nx))   # concentration
C0 = C.copy()

# =====================
# POISSON RHS
# =====================
def build_up_b(b, rho, dt, u, v, dx, dy):
    b[1:-1,1:-1] = rho * (
        ( (u[1:-1,2:] - u[1:-1,0:-2]) / (2*dx)
        + (v[2:,1:-1] - v[0:-2,1:-1]) / (2*dy) ) / dt
        - ((u[1:-1,2:] - u[1:-1,0:-2]) / (2*dx))**2
        - 2*((u[2:,1:-1] - u[0:-2,1:-1]) / (2*dy) *
             (v[1:-1,2:] - v[1:-1,0:-2]) / (2*dx))
        - ((v[2:,1:-1] - v[0:-2,1:-1]) / (2*dy))**2
    )
    return b

# =====================
# PRESSURE POISSON
# =====================
def pressure_poisson(p, dx, dy, b):
    pn = np.empty_like(p)

    for _ in range(nit):
        pn[:] = p[:]

        p[1:-1,1:-1] = (
            (pn[1:-1,2:] + pn[1:-1,0:-2]) * dy**2 +
            (pn[2:,1:-1] + pn[0:-2,1:-1]) * dx**2
        ) / (2*(dx**2 + dy**2)) \
        - dx**2 * dy**2 / (2*(dx**2 + dy**2)) * b[1:-1,1:-1]

        p[:,0]  = p[:,1]
        p[:,-1] = p[:,-2]
        p[0,:]  = p[1,:]
        p[-1,:] = 0.0

    return p

# =====================
# MAIN SOLVER
# =====================
def cavity_flow(nt, u, v, p, C):

    for n in range(nt):

        un = u.copy()
        vn = v.copy()
        Cn = C.copy()

        # ---- PRESSURE ----
        b = build_up_b(np.zeros_like(p), rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b)

        # ---- VELOCITY ----
        u[1:-1,1:-1] = (
            un[1:-1,1:-1]
            - un[1:-1,1:-1]*dt/dx*(un[1:-1,1:-1]-un[1:-1,0:-2])
            - vn[1:-1,1:-1]*dt/dy*(un[1:-1,1:-1]-un[0:-2,1:-1])
            - dt/(2*rho*dx)*(p[1:-1,2:]-p[1:-1,0:-2])
            + nu*dt*(
                (un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,0:-2])/dx**2 +
                (un[2:,1:-1]-2*un[1:-1,1:-1]+un[0:-2,1:-1])/dy**2
            )
        )

        v[1:-1,1:-1] = (
            vn[1:-1,1:-1]
            - un[1:-1,1:-1]*dt/dx*(vn[1:-1,1:-1]-vn[1:-1,0:-2])
            - vn[1:-1,1:-1]*dt/dy*(vn[1:-1,1:-1]-vn[0:-2,1:-1])
            - dt/(2*rho*dy)*(p[2:,1:-1]-p[0:-2,1:-1])
            + nu*dt*(
                (vn[1:-1,2:]-2*vn[1:-1,1:-1]+vn[1:-1,0:-2])/dx**2 +
                (vn[2:,1:-1]-2*vn[1:-1,1:-1]+vn[0:-2,1:-1])/dy**2
            )
        )

        # ---- VELOCITY BC ----
        u[0, :]  = 0
        u[:, 0]  = 0
        u[:, -1] = 0
        u[-1, :] = u[-2, :]    # set velocity on cavity lid equal to 1
        v[0, :]  = 2.0
        v[-1, :] = v[-2, :]
        v[:, 0]  = 0
        v[:, -1] = 0

        # ============================
        # ADVECTION–DIFFUSION (UPWIND)
        # ============================

        dCdx = (C[1:-1,1:-1] - C[1:-1,0:-2]) / dx
        dCdy = (C[1:-1,1:-1] - C[0:-2,1:-1]) / dy

        adv = -(u[1:-1,1:-1]*dCdx + v[1:-1,1:-1]*dCdy)

        d2Cdx2 = (C[1:-1,2:] - 2*C[1:-1,1:-1] + C[1:-1,0:-2]) / dx**2
        d2Cdy2 = (C[2:,1:-1] - 2*C[1:-1,1:-1] + C[0:-2,1:-1]) / dy**2

        diff = D * (d2Cdx2 + d2Cdy2)

        Cn[1:-1,1:-1] = C[1:-1,1:-1] + dt*(adv + diff)

        # ---- SOURCE (MANBA) ----
        Cn[0,:] = 0.8   # doimiy kirish konsentratsiyasi

        # ---- C BC ----
        Cn[-1,:] = Cn[-2,:]
        Cn[:,0]  = Cn[:,1]
        Cn[:,-1] = Cn[:,-2]

        C = Cn.copy()

    return u, v, p, C

# =====================
# RUN
# =====================
u, v, p, C = cavity_flow(nt, u, v, p, C)
print(u,v,p,C)
# =====================
# PLOTS
# =====================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.contourf(X, Y, np.sqrt(u**2 + v**2), cmap=cm.viridis)
plt.colorbar()
plt.title(f"Oqim tezligini o'zgarishi {nt} sekundda")
plt.xlabel("X [m]")
plt.ylabel("y [m]")
plt.subplot(1,2,2)
plt.contourf(X, Y, C, cmap=cm.viridis)
plt.colorbar()
plt.title(f"Konsentratsiyaning o'zgarishi {nt} sekundda")
plt.xlabel("X [m]")
plt.ylabel("y [m]")
plt.tight_layout()
plt.show()
