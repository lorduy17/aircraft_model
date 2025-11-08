
    #%%
import numpy as np
import pyswarms as ps
import matplotlib.pyplot as plt

def x_dot(state_var, control_var):
    """
    control_vars: Variables of control, how:

    u1: delta_aleron, rad
    u2: delta_elevator, rad
    u3: delta_rudder, rad
    u4: thurst 1
    u5: thurst 2

    state_var: Variables of state, define the state of the A/C.

    x1: u
    x2: v
    x3: w
    x4: p
    x5: q
    x6: r
    x7: phi
    x8: theta
    x9: psi

    where body speed, m/s: x1,x2,x3  and angle rates, rad/s: x4,x5,x6
    """

    ### deg2rad
    deg2rad = np.pi/180
    
    # STEP 1
    # Control limits
    

    for i in range(len(control_var)):
        if i < 3:
            control_var[i] = np.clip(control_var[i],-25*deg2rad,25*deg2rad)
        else:
            control_var[i] = np.clip(control_var[i],0,1)

    u1,u2,u3,u4,u5 = control_var


    # STEP 2
    ## Variables intermedias
    x1,x2,x3,x4,x5,x6,x7,x8,x9 = state_var

    body_speed = np.array([x1,x2,x3])
    angle_rates = np.array([x4,x5,x6])
    euler_angles = np.array([x7,x8,x9])


    Va = np.sqrt(x1**2 + x2**2 + x3**2)
    if Va < 1e-6:
        Va = 1e-6

    alpha = np.arctan2(x3,x1)
    beta = np.arcsin(np.clip(x2/Va, -1, 1))
    rho = 1.225
    Q = 0.5*rho*Va**2
    g = 9.80665 # m/s2
    

    # Aircraft parameters
    m = 120e3 # kg
    innertia_body = m* np.array([
        [40.07,0,2.098],
        [0,64,0],
        [2.098,0,99.92]
    ])

    s = 260 # m2
    mac = 4.2 # m
    x_apt1 = 0
    y_apt1 = 7.94
    z_apt1 = 1.9
    x_apt2 = 0
    y_apt2 = -7.94
    z_apt2 = 1.9

    # STEP 3
    ## Nondimensional Aero Forces coefficientes in Fs
    alpha0 = -11.5/180*np.pi # rad
    if alpha <= 14.5/180*np.pi:
        
        cl_wb = 5.5*(alpha-alpha0)
    else:
        a1 = -155.2
        a2 = 609.2
        a3 = -768.5
        a0 = 15.212
        cl_wb = a0 + a1*alpha + a2*alpha**2 + a3*alpha**3


    # Tail
    s_t = 64 # m2
    l_t = 24.8 #m
    deda = 0.25
    epsilon = deda*(alpha -alpha0)
    alpha_t = alpha-epsilon+u2+1.3*x5*l_t/Va
    cl_t = s_t/s*3.1*alpha_t
    cl = cl_wb+cl_t

    # Forces
    c_d = 0.13+0.0061*(cl_wb-0.45)**2
    c_y = -1.6*beta+0.24*u3
    

    ## Aerodynamic Force in Fb
    non_dim_forces = [-c_d,c_y,-cl]

    c_ws = np.array([[np.cos(beta),np.sin(beta),0],
                    [-np.sin(beta),np.cos(beta),0],
                    [0,0,1]])

    # STEP 4
    forces_s = Q*s*np.array(non_dim_forces)
    c_bs = np.array([
        [np.cos(alpha),0,-np.sin(alpha)],
        [0,1,0],
        [np.sin(alpha),0,np.cos(alpha)]
    ])
    forces_a = c_bs@forces_s


    # STEP 5
    ## Nondimensional Aero Moment Coefficient about AC in Fb
    # Defube parametters for calc cm

    n_bar = np.array([
        -1.4*beta,
        -0.59-3.1*s_t*l_t/(s*mac)*(alpha-epsilon),
        (1-alpha*180/(np.pi*15))*beta
    ])
    cm_x = np.array([
        [-11,0,5],
        [0,-4.03*s_t*l_t**2/(s*mac**2),0],
        [1.7,0,-11.5]
    ])
    cm_u = np.array([
        [-0.6,0,0.22],
        [0,-3.1*(s_t*l_t/(s*mac)),0],
        [0,0,-0.63]
    ])


    cm_ac = n_bar +cm_x@(mac/Va*angle_rates)+cm_u@[u1,u2,u3]
    moments_ac = mac*cm_ac*Q*s
    
    # STEP 7
    ## Aero moment about cg in Fb
    """
    Define parammetters (r_cg,r_ac) for calculate mmoment cg in Fb
    """

    r_cg = np.array([0.23*mac,0,0.1*mac])
    r_ac = np.array([0.12*mac,0,0])
    moments_cg = moments_ac+np.cross(forces_a,(r_cg-r_ac))

    # STEP 8
    ## Propulsion effects
    F_trust_max_per_engine = 0.175*m*g # N
    f1 = u4*F_trust_max_per_engine
    f2 = u5*F_trust_max_per_engine

    """Define parammetters (u_bar1,u_bar2) for the propulsion effects"""
    
    f_e = f1+f2

    u_bar1 = np.array([
        x_apt1,
        y_apt1,
        z_apt1
    ])
    u_bar2 = np.array([
        x_apt2,
        y_apt2,
        z_apt2
    ])

    
    m_ecg = np.cross(u_bar1,[f1,0,0]) + np.cross(u_bar2,[f2,0,0])

    # STEP 9
    ## Gravity effects
    fg_bar = m*np.array([
        -g*np.sin(x8),
        g*np.cos(x8)*np.sin(x7),
        g*np.cos(x8)*np.cos(x7)
    ]) 

    # STEP 10
    forces = forces_a + fg_bar +[f_e,0,0]
    moments =  m_ecg + moments_cg

    accel_body = forces/m-np.cross(angle_rates,body_speed)

    innertia_body_inv = np.linalg.inv(innertia_body)
    accel_angles = innertia_body_inv@(moments-np.cross(angle_rates,innertia_body@angle_rates))
    euler_derivate = np.array([
        [1,np.sin(x7)*np.tan(x8),np.cos(x7)*np.tan(x8)],
        [0,np.cos(x7),-np.sin(x7)],
        [0,np.sin(x7)/np.cos(x8),np.cos(x7)/np.cos(x8)]
    ])@angle_rates

    x_dot_r = np.hstack((accel_body,accel_angles,euler_derivate))
    
    x_dot_r = np.array(x_dot_r,dtype=float)

    return x_dot_r

def simulate(X,U,time,dt,da,eg):
    """

    The function simnulate the behavior of the A/C (aircraft) with
    cases like if the pilot deflect aleron, shut off a any engine

    X: state variables.
    X = ["u", "v", "w", "p", "q", "r", "phi", "theta", "psi"]
    U: control variables
    U = [delta_e, delta_a, delta_r, delta_t1, delta_t2]
    time: Time that the user wanna simulate
    dt: How many each do calculate the behavior
    da: Aleron deflection, degs
    eg: shutoff the engine. 1: shut off engine 1 , 2: shut off engine 2, 0: none
    
    
    """

    deg2rad = np.pi/180

    ## Fix unit's control variables 
    u1,u2,u3 = U[0:3]
    U = np.array([u1,u2,u3,U[3],U[4]])
    ## First simulation state   
    states = [X] # A/C in time 0
    time_vector = [0]

    iter_counter = 1

    counter_time = dt
    
    x_current = X.copy()
    U_initial = U.copy()

    iter_fail = None

    print("Starting simulation...")


    while counter_time <= time:


        U = U_initial.copy()

        if da is not None:
            if 30 <= counter_time <= 32:
                U[0] += da*deg2rad # rad
        
        if eg is not None:
            if counter_time > 60:
                if eg == 1:
                    U[3] = 0 # shut off engine 1
                elif eg == 2:
                    U[4] = 0 # shut off engine 2
            


        x_dot_current = x_dot(x_current,U)
        x_next = x_current + x_dot_current*dt


        Va = np.sqrt(x_next[0]**2 + x_next[1]**2 + x_next[2]**2)
        if Va > 300 or np.any(np.abs(x_next[6:9]) > np.pi/2):

            iter_fail = [iter_counter, counter_time]

            

        states.append(x_next)
        time_vector.append(counter_time)

        x_current = x_next
        counter_time += dt
        iter_counter += 1


    print("iterations: ",iter_counter)
    print("Time simulated: ",counter_time," s")
    if iter_fail is not None:
        print("Simulation can not converge in iteration",iter_fail[0]," and time ",iter_fail[1]," s")

    print("Simulation finished")
    states = np.array(states)

    ## Plotter

    labels_y  = ["u, m/s", "v, m/s", "w, m/s", "p, rad/s", "q, rad/s", "r, rad/s", "phi, rad", "theta, rad", "psi, rad"]
    subtitle = ["Body Speeds","Angle Rates","Euler Angles"]

    var = 0

    for i in range(3):
        fig, axs = plt.subplots(3,sharex = True)
        for j in range(3):
            axs[j].plot(time_vector,states[:,var])
            axs[j].set_ylabel(labels_y[var])
            axs[j].grid()
            var += 1
        axs[2].set_xlabel("Time, s")
        fig.suptitle(subtitle[i])
        
        
    plt.show()
    return states[:,iter_counter-1]

def cost_function(Va_ideal,x_ideal,x_current,U_current):
    """

    The function calc the cost function for the state desiered

    x_ideal: state vector where we want to evaluate the cost,
    U_current: control vector
    x_current: current state vector
    Va_ideal: air speed ideal

    syntaxis format:
    x_ideal = [u,v,w,p,q,r,phi,theta,psi] m/s,rad/s,rad
    x_current = [u,v,w,p,q,r,phi,theta,psi] m/s,rad/s,rad
    U_current = [delta_e, delta_a, delta_r, delta_t1, delta_t2] rad, thrust: [0,1]
    Va_ideal, m/s

    """

    deg2rad = np.pi/180

    # Current vars 
    xdot_current = x_dot(x_current,U_current)
    phi_current = x_current[6]
    psi_current = x_current[8]
    Va_current = np.sqrt(x_current[0]**2,x_current[1]**2,x_current[2]**2)
    climb_angle_current = x_current[7] - np.arctan2(x_current[0]/x_current[2])

    # Erros calcs
    error_xdot = x_ideal-xdot_current
    error_Va = Va_ideal - Va_current
    error_climb_angle = - climb_angle_current
    error_psi = -psi_current
    error_phi = -phi_current
    
    error_vector = error_xdot
    error_vector.append(error_Va)
    error_vector.append(error_climb_angle)
    error_vector.append(error_psi)
    error_vector.append(error_phi)
#%%
    # Scaling factors
    scaling_factor_vector = np.array([
        1,1,1,                          # body accelaration (ba) [m/s^2]
        0.1,0.1,0.1,                    # angles accelarations (ac) [rad/s^2]
        0.1,0.1,0.1,                    # angle rates ar [rad/s]
        1,                              # Va error [m/s]
        0.01,                           # Climb angle (gamma) [rad]
        0.1,                            # psi [rad]
        0.05                            # pho [rad]
    ])

    # Weight importance
    weight_vector = np.array([
        50,50,50,                       # ba: important
        150,150,150,                    # ac: very important, is to sensible with stability
        50,50,50,                       # ar: moderate
        200,                            # Va: Very important, is the base of forces and accelerations of system
        100,                            # gamma: Important, need equal 0 for keep the atitude
        150,                            # psi: important, keep hading
        100                             # phi: important
])
#%%
    # cost calc
    norm = (error_vector/scaling_factor_vector)**2
    J_terms = weight_vector*norm
    J = np.sum(J_terms)

    return J



if __name__ == "__main__":
    # Initial conditions
    X0 = np.array([85,0,0,0,0,0,0,0.1,0]) # m/s , rad/s , rad
    U0 = np.array([0,-0.1,0,0.8,0.8]) # deg , deg , deg , thrust
    

    # Simulation parameters
    time = 60*3 # s
    dt = 0.001 # s

    eg = 0
    da = 0 # deg

    simulate(X0,U0,time,dt,da,eg)

# %%
