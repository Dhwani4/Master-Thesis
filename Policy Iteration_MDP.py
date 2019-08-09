

"""
A small grid world with a "mother" cat and a lot of "baby" cats.
The cats get positive reinforcement for eating.
Food appears on the grid at one of four corners.
The mother can see the entire grid, the babies only see nearby.

The babies have a fixed policy:
If they see food, they go to the food.
Else, if they see the mother, they go to the mother.
Else, they stay put.

The mother can have varying degrees of selfishness and empathy.
Selfishness determines how much she wants to eat.
Empathy determines how much she wants the babies to eat.
Both can be positive or negative numbers.
Her policy is to be determined by MDP value iteration.

State representation:
(
    mother x, mother y,
    (..., baby x, ...), (..., baby y, ...)),
    food
)

where food is a length-4 boolean tuple which is True for each corner that has food.
Corners are listed clockwise in the order (0,0), (0,5), (5,5), (5,0)
"""
import itertools as it
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as pt
from scipy.sparse import identity
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve
from scipy.sparse import vstack
# Initial parameters.  P tables need to be rebuilt when these are changed.
grid_cols = 6 # size of grid world
grid_rows = 6 # size of grid world
baby_view = 1 # how far babies can see
num_babies = 2 # number of babies

# total number of possible states (mother/baby/food positions)
num_states = (grid_cols*grid_rows)**(1+num_babies) * 2**4

def food_coords(food):
    """
    Convert a length 4 boolean food array into x/y coordinates of food.
    For example, if food == (True, False, True, True), then
    f_x = (0, 5, 5), f_y = (0, 5, 0)
    """
    f_x = tuple(np.array([0, 0, grid_cols-1, grid_cols-1])[np.array(food, dtype=bool)])
    f_y = tuple(np.array([0, grid_rows-1, grid_rows-1, 0])[np.array(food, dtype=bool)])
    return f_x, f_y

def plot_state(state):
    """
    Visualize a state with matplotlib:
        Green circles are food
        Red circle is mother
        Blue circles are babies
    A little noise is added to each baby position to see distinct babies
    """
    m_x, m_y, b_x, b_y, food = state
    f_x, f_y = food_coords(food)
    pt.scatter(f_x, f_y, s=400, c='g')
    pt.scatter(m_x, m_y, s=200, c='r')
    pt.scatter(
        np.array(b_x)+np.random.randn(num_babies)*0.1,
        np.array(b_y)+np.random.randn(num_babies)*0.1, s=50, c='b')
    pt.xlim([-1, grid_cols])
    pt.ylim([-1, grid_rows])
    pt.grid()

def state_to_index(state):
    """
    For initial parameters there are ~ 700K possible states.
    This assigns each of those states a unique index in the range 0,...,700K-1.
    """
    m_x, m_y, b_x, b_y, food = state
    digits = (m_x, m_y) + b_x + b_y + food
    factors = [grid_cols, grid_rows]*(num_babies+1) + [2]*4
    idx = 0
    coef = 1
    for i in range(2*(1 + num_babies) + 4):
        idx += digits[i]*coef
        coef *= factors[i]
    return int(idx)

def index_to_state(idx):
    """
    This method is the inverse of "state_to_index":
    Given an integer index, it reconstructs the corresponding state.
    """
    factors = [grid_cols, grid_rows]*(num_babies+1) + [2]*4
    digits = []
    for i in range(2*(1 + num_babies) + 4):
        digit = idx % factors[i]
        idx = (idx - digit) / factors[i]
        digits.append(digit)
    m_x, m_y = tuple(digits[:2])
    b_x = tuple(digits[2:2+num_babies])
    b_y = tuple(digits[2+num_babies:-4])
    food = tuple(digits[-4:])
    return (m_x, m_y, b_x, b_y, food)

def move(state, dx, dy):
    """
    This moves the mother's position by dx,dy grid units.
    Babies will also move according to their fixed policy.
    It returns the new state after the mother has moved.
    Typically dx and dy will be in [-1,0,1].
    So the mother moves one unit at a time horizontally/vertically/diagonally.
    """
    m_x, m_y, b_x, b_y, food = state
    f_x, f_y = food_coords(food)

    # move babies
    b_x_new, b_y_new = [], []
    for bx, by in zip(b_x, b_y):

        # if food is in this baby's view, it steps towards food
        food_step = False
        for fx, fy in zip(f_x, f_y):
            if max(abs(fx-bx), abs(fy-by)) <= baby_view:
                food_step = True
                b_x_new.append(bx + np.sign(fx-bx))
                b_y_new.append(by + np.sign(fy-by))
#                if max(abs(fx-bx), abs(fy-by)) == 0:
#                    food = (False, False, False, False)
                    

                
                break
        if food_step: continue
        
        # otherwise, if mother is in this baby's view, it steps towards mother
        if max(abs(m_x-bx), abs(m_y-by)) <= baby_view:
            b_x_new.append(bx + np.sign(m_x-bx))
            b_y_new.append(by + np.sign(m_y-by))
            
            continue
        
        # otherwise, this baby stays at its current position
        b_x_new.append(bx)
        b_y_new.append(by)
    
    # move mother
    m_x_new = max(0, min(m_x + dx, grid_cols-1))
    m_y_new = max(0, min(m_y + dy, grid_rows-1))
#    if max(abs(fx-m_x_new), abs(fy-m_y_new)) == 0:
#        food = (False, False, False, False)       
    new_state = (m_x_new, m_y_new, tuple(b_x_new), tuple(b_y_new), food)
    return new_state


def reward(state, selfish=0., empathy=4.):
    """
    This computes the mother's reward in the given state.
    By default, the reward is zero.
    If the mother is colocated with food, 'selfish' is added to her reward.
    If a baby is colocated with food, 'empathy' is added to her reward.
    Either 'selfish' or 'empathy' can be negative:
    selfish < 0: the mother wants starve
    empathy < 0: the mother wants babies to starve
    """
    m_x, m_y, b_x, b_y, food = state
    f_x, f_y = food_coords(food)
    r = 0.
    for fx, fy in zip(f_x, f_y):
        if m_x == fx and m_y == fy:
            r += selfish
        for bx, by in zip(b_x, b_y):
            if bx == fx and by == fy:
                r += empathy
    return r

def build_mdp():
    """
    Builds the transition arrays P defining the MDP:
    P[(dx,dy)] is a 2d probability matrix when action (dx,dy) is taken.
    If the mother is in state with index i, and moves by (dx,dy),
    there is some probability the mother will end up in another state with index j.
    P[(dx,dy)][i,j] is that probability.
    So P[(dx,dy)] is a 700K x 700K matrix, which is huge.
    However, most probabilities are zero (there is no chance the mother will jump 3 spaces for example).
    So P[(dx,dy)] can be feasibly stored as a *sparse* matrix.
    For this we use scipy's "CSR" format to build the sparse matrices.
    """
    
    # Initial input data for CSR construction:
    # P[row_ind[k], col_ind[k]] = data[k]
    # Any position not in row_ind, col_ind are zero.
    drc = {
        (dx,dy): {"data": [], "row_ind": [], "col_ind": []}
        for dx in [-1,0,1] for dy in [-1,0,1]}

    print("Building MDP")
    # Enumerate each possible state by its index
    for idx in range(num_states):

        if idx % (num_states / 100) == 0: print("%d of %d..." % (idx, num_states))

        # Get state corresponding to current index
        # This corresponds to one row of the P table
        state = index_to_state(idx)
        m_x, m_y, b_x, b_y, food = state
        f_x, f_y = food_coords(food)

        # Try each possible move from this state
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:

                # Get new state and its index, corresponding to a column of the P table
                new_state = move(state, dx, dy)
                new_idx = state_to_index(new_state)
                
                # Taking action dx,dy in current state will produce new_state with probability 1
                # So add it to the P table (all other probabilities in this row will be zero)
                drc[(dx,dy)]["data"].append(1.)
                drc[(dx,dy)]["row_ind"].append(idx)
                drc[(dx,dy)]["col_ind"].append(new_idx)

    # Use scipy constructor to build sparse P tables
    P = {(dx,dy): sp.csr_matrix(
            (drc[(dx,dy)]["data"], (drc[(dx,dy)]["row_ind"], drc[(dx,dy)]["col_ind"])),
            shape=(num_states, num_states))
        for (dx,dy) in drc.keys()}
    
    return P

def build_reward(selfish=1., empathy=0.):
    """
    Builds the reward vector of the MDP
    r[i] is the reward for state with index i
    """

    # Initialize reward vector and enumerate possible states
    r = np.empty(num_states)
    for idx in range(num_states):

        if idx % (num_states / 100) == 0: print("%d of %d..." % (idx, num_states))

        # Construct current state from index and all reward function on it
        state = index_to_state(idx)
        r[idx] = reward(state, selfish, empathy)
        #print(idx)
        #print(r[idx])
    return r

def value_iteration(r, P, g = 0.9, target_error=0, num_iters=None):
    """
    MDP value iteration to "learn" the optimal policy for the mother.
    Conceptually this is an efficient way to run all possible simulations in parallel.
    It uses a "discount factor" g to balance instant and delayed rewards
    It returns a "utility" vector: u[k] is the utility of state with index k.
    "Utility" is the net long-term reward the mother can get from a given state.
    Net long-term rewards put more weight on short-term rewards according to the discount factor.
    Discount factor closer to 1 weights distant rewards more similarly to immediate rewards.
    The more "value iterations", the closer this algorithm gets to approximating true utility.
    It is possible to bound the error between the approximation and the truth.

    Input:
        r: the reward vector from build_reward
        P: the P tables from build_mdp
        g: the "discount" factor in (0,1)
        target_error: using the bound, try to get this close to the true utility vector
        num_iters: how many iterations to perform (None goes until target error achieved)
    Output:
        u: the utility vector 
    """
    u = np.zeros(num_states) # utility vector
    error = np.inf

    for i in it.count():

        if i == num_iters: break
        if error <= target_error: break

        P_u = np.array([P[a].dot(u) for a in P])
        u_new = r + g * P_u.max(axis=0)
        error = np.fabs(u_new - u).max() * (1 - g) / g
        u = u_new
        
        print("iter %d error = %f"%(i,error))
    
    return u

def policy_iteration(r, P, g = 0.9):
    # Initialize thel value function
    U = np.zeros(num_states)

    is_value_changed = True
    iterations = 0
    I = identity(num_states,format = 'csr')
    
    action = [P[-1,-1],P[-1,0],P[-1,1],P[0,-1],P[0,0],P[0,-1],P[1,-1],P[1,0],P[1,1]]
    
    Y = I - ( g * P[-1,-1])
    U = spsolve(Y,r)
    p = np.array([P[a].dot(U) for a in P]).argmax(axis=0)
   
    while is_value_changed:
        is_value_changed = False
        iterations += 1       
       
        A =[]
        B=[]
        for s in range(num_states):
            a =p[s]
            B=action[a][s,:]
            A.append(B)
        
        O_P=vstack(A)    
            
        
            
        Y1= I - (g*O_P)
        U_new = spsolve(Y1,r)
        if (np.all(U==U_new)):
            print("Yes")
        print("Different Utility",np.sum(U!=U_new))
        p_new= np.array([P[a].dot(U_new) for a in P]).argmax(axis=0)
        print("Differen Actions:",np.sum(p!=p_new))
        if (np.all(p== p_new)):
            is_value_changed = False
        else:
            
            U = U_new
            p = p_new
            is_value_changed = True
        print(iterations)
    print("Final policy")
    print(p)
    print(U_new)
    return p

def build_policy(P, u):
    """
    Determines mother's final policy based on P tables and u vector.
    Mother will choose the action with highest expected utility.
    Inputs: P and u
    Returns a policy array p
        a[k,:] = [dx,dy], the optimal move to take in state with index k    
    """
    A = P.keys()
    p = np.array(list(A))[
        np.array([P[a].dot(u) for a in A]).argmax(axis=0)
    ,:]
    return p

if __name__ == "__main__":

    # # test state/index mapping
    # for idx in range(num_states):
    #     if idx % int(num_states/20) == 0: print(idx)
    #     assert(idx == state_to_index(index_to_state(idx)))

    #### !!! Do this once to build P tables and save to disk !!! ###
    P = build_mdp()
    for dx,dy in P:
        sp.save_npz("P_%d_%d.npz" % (dx,dy), P[dx,dy])

   # Load P tables from disk, after previous code has been executed once.
    P = {(dx,dy): sp.load_npz("P_%d_%d.npz" % (dx,dy))
        for dx in [-1,0,1] for dy in [-1,0,1]}
   # print(P[(-1,-1)])
    # Build reward vector with desired parameters
    r = build_reward(selfish=1, empathy=-1)
    # r = build_reward(selfish=1., empathy=-1.)
    # r = build_reward(selfish=1., empathy=1.)

    # Load P tables from disk, after previous code has been executed once.
    #P = {(dx,dy): sp.load_npz("P_%d_%d.npz" % (dx,dy))
       # for dx in [-1,0,1] for dy in [-1,0,1]}

    # Build reward vector with desired parameters !!! Do this whenever changing selfish/empathy
    # r = build_reward(selfish=-.1, empathy=.9)
    # r = build_reward(selfish=1., empathy=-1.)
    # r = build_reward(selfish=1., empathy=1.)
    #r = build_reward(selfish=.6, empathy=.5)
    # np.savez("r.npz", r=r)

    # Load the built reward vector from disk
   # r = np.load("r.npz")["r"]

    # Do value iteration to get optimal mother policy
    #u = value_iteration(r, P, g = 0.95, target_error=0, num_iters=50)
    p = policy_iteration(r, P, g = 0.9)
    #p = build_policy(P, u)

    # Run a simulation with optimal policy to check that it works well
    # Initial state
    state = (1, 0, (0, 0), (5, 4), (False, False, True, False))
    #state = (0, 0, (0, 0), (0, 0), (False, False, False, True))
    # state = (5, 5, (5, 5), (5, 5), (True, True, True, True))
    
    # distance = []
    # for episode in range(30):
    
    #state = index_to_state(np.random.randint(num_states)) # random initial

    # Run simulation
    pt.ion()
    for t in range(12): # number of time-steps

        # show current state
        print(t)
        pt.cla()
        plot_state(state)
        pt.show()
        pt.pause(10.)

        # move and update according to optimal policy
        dx, dy = p[state_to_index(state),:]
        state = move(state, dx, dy)

    
    # np.mean(distance)
    # np.std(distance)
