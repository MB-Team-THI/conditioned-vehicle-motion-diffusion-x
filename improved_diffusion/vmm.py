import numpy as np
import matplotlib.pyplot as plt

dt = 1.0/25

def vmm(acc, dPsi, x_init=0, y_init=0, v_init=30, psi_init=0):
    def update_state(pos_x, pos_y, vel, heading, a_x, dPsi):
        pos_x = pos_x + vel * np.cos(heading) * dt + a_x*np.cos(heading) *(dt**2/2) - dPsi*vel*np.sin(heading)* (dt**2/2)
        pos_y = pos_y + vel * np.sin(heading) * dt + a_x*np.sin(heading) *(dt**2/2) + dPsi*vel*np.cos(heading)* (dt**2/2)
        velocity = vel + a_x*dt
        heading = heading + dPsi*dt
        return pos_x, pos_y, velocity, heading
   
    # Update npe state for 10 time steps
    xn = x_init
    yn = y_init
    veln = v_init
    psin = psi_init

    x=np.zeros_like(acc)
    y=np.zeros_like(acc) 
    vel=np.zeros_like(acc)
    psi = np.zeros_like(acc)
    for i in range(acc.shape[-1]):
        xn, yn, veln, psin = update_state(xn, yn, veln, psin, acc[i].reshape(-1,1), dPsi[i].reshape(-1,1))
        x[i] = xn[0,0]
        y[i] = yn[0,0]
        psi[i] = psin[0,0]
        vel[i] = veln[0,0]
    return x, y, vel, psi

def plot_xy_vmm(x_gt, y_gt, v_gt, psi_gt, x_hat, y_hat, v_hat, psi, maneuver='', psi0=0):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)#, layout='constrained')
    fig.suptitle('Scneario plot ' + maneuver)
    ax1.set_title('Position')
    ax1.set_xlim(-3, 350)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_ylim(-5, 5)
    ax1.plot(x_gt, y_gt, label='gt')
    ax1.plot(x_hat, y_hat, label='pred')
    ax2.set_title('Velocity vx')
    ax2.plot(v_gt, label='gt')
    ax2.plot(v_hat, label='pred')
    ax2.set_ylim(10, 50)
    ax2.set_ylabel('v_x')
    ax2.set_ylabel('step t')
    ax3.plot(psi_gt,  label='gt')
    ax3.plot(psi, label='pred')
    ax1.legend()
    ax2.legend()
    plt.show()
