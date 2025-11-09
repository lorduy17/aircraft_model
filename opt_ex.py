#%%
import numpy as np
import pyswarms as ps
import matplotlib.pyplot as plt

def x_dot(state_var, control_var):

    """

    Inputs:
    control_vars: Variables of control, how:

    control_var = [u1,u2,u3,u4,u5]

    u1: delta_aleron, rad
    u2: delta_elevator, rad
    u3: delta_rudder, rad
    u4: thurst 1
    u5: thurst 2

    state_var: Variables of state, define the state of the A/C.

    state_var = [x1,x2,x3,x4,x5,x6,x7,x8,x9]

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

    Output:
    xdot = [accel_body,accel_angles,euler_rates_rate]

    accel_body = [u_dot,v_dot,w_dot] m/s**2
    accel_angles = [p_dot,q_dot,r_dot] rad/s**2
    euler_rates_rate = [phi_dot,theta_dot,psi_dot] rad/s
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
        [40.07,0,-2.098],
        [0,64,0],
        [-2.098,0,99.92]
    ])

    s = 260 # m2
    mac = 6.6 # m
    x_apt1 = 0
    y_apt1 = 7.94
    z_apt1 = 1.9
    x_apt2 = 0
    y_apt2 = -7.94
    z_apt2 = 1.9

    # STEP 3
    ## Nondimensional Aero Forces coefficientes in Fs
    alpha0 = -11.5/180*np.pi # rad
    n = 5.5

    if alpha <= 14.5/180*np.pi:  
        cl_wb = n*(alpha-alpha0)
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
    

    # Forces
    cl = cl_wb+cl_t
    c_d = 0.13+0.0061*(n*alpha+0.645)**2
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
        -0.59-(3.1*s_t*l_t)/(s*mac)*(alpha-epsilon),
        (1-alpha*180/(np.pi*15))*beta
    ])
    cm_x = mac/Va*np.array([
        [-11,0,5],
        [0,-4.03*s_t*l_t**2/(s*mac**2),0],
        [1.7,0,-11.5]
    ])
    cm_u = np.array([
        [-0.6,0,0.22],
        [0,-3.1*(s_t*l_t/(s*mac)),0],
        [0,0,-0.63]
    ])


    cm_ac = n_bar +cm_x@(angle_rates)+cm_u@[u1,u2,u3]
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

    f1 = min(u4*m*g, 0.175*m*g) 
    f2 = min(u5*m*g, 0.175*m*g)

    """Define parammetters (u_bar1,u_bar2) for the propulsion effects"""
    
    f_e = f1+f2

    u_bar1 = np.array([
        x_apt1,
        y_apt1,
        z_apt1
])
    u_bar1 = u_bar1 - r_cg
    u_bar2 = np.array([
        x_apt2,
        y_apt2,
        z_apt2
    ])
    u_bar2 = u_bar2 - r_cg

    
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

def simulate(X,U,time,dt,da=False,eg=False):
    """

    The function simnulate the behavior of the A/C (aircraft) with
    cases like if the pilot deflect aleron, shut off a any engine

    Input:
        X: state variables.
        X = ["u", "v", "w", "p", "q", "r", "phi", "theta", "psi"]
        U: control variables
        U = [delta_e, delta_a, delta_r, delta_t1, delta_t2]
        time: Time that the user wanna simulate
        dt: How many each do calculate the behavior
        da: Aleron deflection, degs
        eg: shutoff the engine. 1: shut off engine 1 , 2: shut off engine 2, 1: none

    Output:
        states: Matrix with A/C in each dt.
        states = size: [time/dt,9] 
        
        states = [
        u0, v0, w0, p0, q0, r0, phi0, theta0, psi0
        u1, v1, w1, p1, q1, r1, phi1, theta1, psi1
        u2, v2, w2, p2, q2, r2, phi2, theta2, psi2
        u3, v3, w3, p3, q3, r3, phi3, theta3, psi3
        .
        .
        .
        ]
    
    """

    deg2rad = np.pi/180

     
    u1,u2,u3 = U[0:3]
    U = np.array([u1,u2,u3,U[3],U[4]])
    
    # Save initial moment
    states = [X] 
    time_vector = [0]

    iter_counter = 1

    counter_time = dt
    
    # Save x,U for start simulation.
    x_current = X.copy()
    U_initial = U.copy()

    # Fron terminal
    strs = " Started simulation "
    print(
        f"~"*50,"\n",
        f"-"*10,strs
    )
        
    
    iter_fail = None ## If simulation tend to inf -> Show warning in the terminal

    while counter_time <= time:
        U = U_initial.copy()

        # If the user wanna case simulate with aleron deflection 
        if da is not None:
            if 30 <= counter_time <= 32:
                U[0] += da*deg2rad # rad
        
        # If the user wanna case simulate with engine fail 
        if eg is not None:
            if eg == 1:
                U[3] = 0 # shut off engine 1
            elif eg == 2:
                U[4] = 0 # shut off engine 2
            

        # Euler explicit integration
        x_dot_current = x_dot(x_current,U)
        x_next = x_current + x_dot_current*dt

        # Conditions for active warning
        Va = np.sqrt(x_next[0]**2 + x_next[1]**2 + x_next[2]**2)
        if Va > 300 or np.any(np.abs(x_next[6:9]) > np.pi/2):
            iter_fail = True

        states.append(x_next) # Add X to state
        time_vector.append(counter_time)

        x_current = x_next # Fresh x_current
        counter_time += dt
        

        if iter_counter%5000 == 0:
            print(
            f"Simulation: {round((counter_time/time)*100,2)}%","\t", f"time simulated: {round(counter_time,0)}s"
            )
    
        if iter_fail is not None:
            if iter_counter%2500 == 0:
                print(50*"*")
                print(2*"\t","WARNING")
                print("Simulation can not converge for")
                print(f"Iteration:{iter_counter}","\twith\t",f"time: {round(counter_time,0)}s")
                print(50*"*")
    print(
    f"-"*10," Simulation finished","\n",
    f"~"*50
    )
    states = np.array(states)

    ## Plotter
    fig, axs = plt.subplots(9, 1, figsize=(10, 15), sharex=True)
    labels_y = ["u, m/s", "v, m/s", "w, m/s", "p, rad/s", "q, rad/s", "r, rad/s", "phi, rad", "theta, rad", "psi, rad"]
    
    for i in range(9):
        axs[i].plot(time_vector, states[:,i], label=labels_y[i],)
        axs[i].legend()
        axs[i].grid(True)
        if i == 8:  # Last subplot
            axs[i].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.show(block=True)

    return states

def cost_function(Va_ideal,x_ideal,x_last,U_current=None):
    """
    The function calc the cost function for the state desiered

    Input:
        x_ideal: state vector where we want to evaluate the cost,
        U_current: control vector
        x_current: current state vector
        Va_ideal: air speed ideal

        syntaxis format:
        x_ideal = [u,v,w,p,q,r,phi,theta,psi] m/s,rad/s,rad
        x_current = [u,v,w,p,q,r,phi,theta,psi] m/s,rad/s,rad
        U_current = [delta_e, delta_a, delta_r, delta_t1, delta_t2] rad, thrust: [0,1]
        Va_ideal= float, m/s

    Output:
        J: Escalar

    """

    deg2rad = np.pi/180

    # Current vars 
    xdot_current = x_dot(x_last,U_current)
    euler_angle_current = x_last[6:]
    Va_current = np.linalg.norm(x_last[0:3])
    if Va_ideal == None:
        Va_ideal = np.linalg.norm(x_ideal[0:3])
    alpha_current = np.arctan2(x_last[2], x_last[0])
    climb_angle_current = euler_angle_current[1] - alpha_current


    # Erros calcs
    error_dynamic = xdot_current
    error_Va = Va_ideal - Va_current
    error_climb_angle = climb_angle_current
    error_psi = x_ideal[8] - x_last[8]
    error_phi = x_ideal[6] - x_last[6]
    
    error_vector = np.hstack((
        error_dynamic,
        error_Va,
        error_climb_angle,
        error_psi,
        error_phi
    ))

    # Scaling factors
    scaling_factor_vector = np.array([  
        20,20,20,                   # body accelaration (ba) [m/s^2]
        2,2,2,                      # angles accelarations (ac) [rad/s^2]
        0.2,0.2,0.2,                # eangs]
        10,                         # Va error [m/le (gamma) [rad]
        0.1,                        # Climb angle rates ar [rad/s]
        10,                         # psi [rad]
        0.3                         # pho [rad]
    ])

    # Weight importance
    weight_vector = np.array([ 
        50,50,50,                       # ba: important
        80,100,80,                      # ac: very important, is to sensible with stability
        30,30,30,                       # ar: moderate
        120,                            # Va: Very important, is the base of forces and accelerations of system
        60,                             # gamma: Important, need equal 0 for keep the atitude
        40,                             # psi: important, keep hading
        40                              # phi: important
    ])
    
    error_scaling_sq = (error_vector / scaling_factor_vector)**2
    J = np.sum(weight_vector * error_scaling_sq)
    
    return J



def found_opt(Va_ideal,x_ideal,x_last,n_particles,options,iters,show=False):
    """
    Function search the optimal configuration of control for state ideal

    Input:
        Va_ideal: Air speed desierd for the user
        x_ideal: Desierd contion for the user
        x_last: Last state of simulate
        n_particles: Number of particles searching optimal point
        options: Set the coeficients for PSO.
        iters: Number of iterations.
        show: If is True, show the user the optimization % in the terminal

        Syntaxis:
        Va_idea = float, m/s
        x_ideal = [u, v, w, p, q, r, phi, theta, psi], m/s,rad/s,rad for [0:2],[3:5],[5:8]
        x_last = [u, v, w, p, q, r, phi, theta, psi], m/s,rad/s,rad for [0:2],[3:5],[5:8]
        n_particles = float
        options= {
        "c1":float ; [0,2]
        "c2":float ; [0,2]
        "w":float
        }
        iters = float
        show = Boolean

    Output: cost,U_opt
        cost: Cost of fuction to optmize
        U_opt: Controls vars optimize

        Syntaxis:
        cost: float
        U_opt = [delta_e, delta_a, delta_r, delta_t1, delta_t2] rad, thrust: [0,1]
    """

    x_last = np.array(x_last,dtype=float)
    x_ideal = np.array(x_ideal,dtype=float)

    # Create the functions of control to optimize
    def objective(U_vector):
        dt = 0.1
        t = 30
        steps = max(1, int(t /dt))
        n = U_vector.shape[0]
        costs = np.full(n,np.inf,dtype=float)

        # Little simulation for predic x for evaluates J.
        for i, U in enumerate(U_vector):
            x_temp = x_last.copy()
            for _ in range(steps):
                x_temp = x_temp + x_dot(x_temp,U)*dt
            costs[i] = cost_function(Va_ideal, x_ideal, x_temp, U)
        return costs

    
    deg2rad = np.pi/180
    
    # Define the constains.
    lower_bounds = [-25*deg2rad,-25*deg2rad,-25*deg2rad,0.5,0.5]
    upper_bounds = [25*deg2rad,25*deg2rad,25*deg2rad,1,1]
    bounds = (np.array(lower_bounds),np.array(upper_bounds))

    # Optimizer
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=5,
        options=options,
        bounds=bounds
    )
    # Search started
    cost, U_opt = optimizer.optimize(objective,iters=iters,verbose=show)
    x_dot_r = x_dot(x_ideal,U_opt)
    Va_result = np.sqrt(x_ideal[0]**2+x_ideal[1]**2+x_ideal[2]**2)

    # Front terminal
    print("~"*50)
    print(f" xdor result: {x_dot_r}")
    print(f" Va resurt: {Va_result}")


    return cost, U_opt


if __name__ == "__main__":

    ### Parametters for simulations:
    # - First simulate with 
    #       x = [85,0,0,0,0,0,0,0.10,0]
    #       u = [0,-0.1,0,0.8,0.8]

    x,u = [85,0,0,0,0,0,0,0.10,0],[0,-0.1,0,0.8,0.8]
    ### Parametters for simulations:
    time = 60*3 # s
    dt = 0.005 # s
    
    strs = " FIRST SIMULATION "

    print(
        f"_"*50,"\n",
        f"-"*16,strs,"\n",
        f"-"*10," Simulate normal fligth","\n",
        f"-"*10,f" Simlate with: ","\n",
        "-"*5,"x = [85,0,0,0,0,0,0,0.10,0]","\n",
        "-"*5,"and","\n",
        "-"*5,"u = [0,-0.1,0,0.8,0.8]"
    )

    simulate(x,u,time,dt)

    # - Second simulate for:
    #       first simulation adding
    #       aleron deflection of 5°

    da = 5*np.pi/180
    strs = "SECOND SIMULATION"
    
    print(
        f"_"*50,"\n",
        f"-"*17,strs,"\n",
        f"-"*5," Aleron deflected for 2 seconds \n",
        f"-"*5," at 30 seconds of simulation",
    )
    simulate(x,u,time,dt,da=da)

    # - Third simulatiom
    #       Simulate a engine fail
    print("_"*50)
    print("-"*17," THIRD SIMULATION")
    print("-"*5," Simulate a engine fail ")

    ### Parametters
    eg = 2
    simulate(x,u,time,dt,eg=eg)

    # - Fourth simulation
    #       Found conditions for flight trim
    print("_"*50)
    print("-"*17," FORTH SIMULATION")
    print("-"*5," Found conditions for flight trim")

    ### Parametters
    
    x_ideal = [95,0,0,0,0,0,0,0,-np.pi]
    Va = 95
    options = {"c1":1.2 , "c2":1.8 , "w": 0.7}
    n_particles = 30
    iters = 300
    show = True
    x_last = x.copy()

    cost , U_opt = found_opt(Va,x_ideal,x_last,n_particles=n_particles,iters=iters,options=options,show=show)
    print("~"*50)
    print("-"*10,"Optmization results:")
    print(f"J:\t{cost}","\n",f"Control optimal set:\t{U_opt}")

    simulate(x_last,U_opt,time,dt) 