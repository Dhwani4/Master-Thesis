# Calling Dijkstra's algorithm to find shortest path using code from https://www.bogotobogo.com/python/python_Dijkstras_Shortest_Path_Algorithm.php
import numpy as np
import heapq
import matplotlib.pyplot as pt
grid_cols = 6 # size of grid world
grid_rows = 6 # size of grid world
baby_view = 1 # how far babies can see
num_babies = 2
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

def index_to_state(idx):
    """
    This method is the inverse of "state_to_index":
    Given an integer index, it reconstructs the corresponding state.
    """
    factors = [grid_cols, grid_rows]*(num_babies+1) + [2]*4
    digits = []
    for i in range(2*(1 + num_babies) + 4):
        digit =int( idx % factors[i])
        idx = int((idx - digit) / factors[i])
        digits.append(digit)
    m_x, m_y = tuple(digits[:2])
    b_x = tuple(digits[2:2+num_babies])
    b_y = tuple(digits[2+num_babies:-4])
    food = tuple(digits[-4:])
    return (m_x, m_y, b_x, b_y, food)

class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}
        # Set distance to infinity for all nodes
        self.distance = float("inf")
        # Mark all nodes unvisited        
        self.visited = False  
        # Predecessor
        self.previous = None

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()  

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

    def set_distance(self, dist):
        self.distance = dist

    def get_distance(self):
        return self.distance

    def set_previous(self, prev):
        self.previous = prev

    def set_visited(self):
        self.visited = True

    def __lt__(self, other):
        return self.distance < other.distance

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        
        else:
            return None

    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()

    def set_previous(self, current):
        self.previous = current

    def get_previous(self, current):
        return self.previous

def shortest(v, path):
    if v.previous:
        path.append(v.previous.get_id())
        shortest(v.previous, path)
        
    return


def dijkstra(aGraph, start, target):
    #print ('''Dijkstra's shortest path''')
    # Set the distance for the start node to zero 
    start.set_distance(0)
    unvisited_queue = [(v.get_distance(),v) for v in aGraph]
    heapq.heapify(unvisited_queue)
    

    while len(unvisited_queue):
        # Pops a vertex with the smallest distance 
        uv = heapq.heappop(unvisited_queue)
        current = uv[1]
        current.set_visited()

        
        for next in current.adjacent:
            # if visited, skip
            if next.visited:
                continue
            new_dist = current.get_distance() + current.get_weight(next)
            
            if new_dist < next.get_distance():
                next.set_distance(new_dist)
                next.set_previous(current)
#               

        # Rebuild heap
        # 1. Pop every item
        while len(unvisited_queue):
            heapq.heappop(unvisited_queue)
        # 2. Put all vertices not visited into the queue
        unvisited_queue = [(v.get_distance(),v) for v in aGraph if not v.visited]
        heapq.heapify(unvisited_queue)
    
def move(state):
    
    m_x, m_y, b_x, b_y, food = state
    f_x, f_y = food_coords(food)
    #print(m_x)
    b_x_new, b_y_new = [], []
    for bx, by in zip(b_x, b_y):

        # if food is in this baby's view, it steps towards food
        food_step = False
        for fx, fy in zip(f_x, f_y):
            
            if max(abs(fx-bx), abs(fy-by)) <= baby_view:
                food_step = True
                
                b_x_new.append(bx + np.sign(fx-bx))
                b_y_new.append(by + np.sign(fy-by))
                
                
                break
        if food_step: continue
        
        # otherwise, if mother is in this baby's view, it steps towards mother
        if max(abs(m_x-bx), abs(m_y-by)) <= baby_view:
            #print("*")
            b_x_new.append(bx + np.sign(m_x-bx))
            b_y_new.append(by + np.sign(m_y-by))
            continue
        b_x_new.append(bx)
        b_y_new.append(by)
        

#    m_x_new = max(0, min(m_x, grid_cols-1))
#    m_y_new = max(0, min(m_y, grid_rows-1))

    new_state = (m_x,m_y, tuple(b_x_new), tuple(b_y_new), food)
    return new_state

def make_graph(g,state,flag=False):
    
    grid_row = 6
    grid_column = 6
    for r in range(grid_row):
        for c in range(grid_column):
            g.add_vertex((r,c))
   
    m_x, m_y, b_x, b_y, food = state
    f_x, f_y = food_coords(food)
    B1 =[]
    B2=[]
    for bx, by in zip(b_x, b_y):
        B1.append(bx)
        B2.append(by)

    #print(B1)
    #print(B2)
    p1,p2 = (B1[0],B2[0]),(B1[1],B2[1])

    if flag == True:
        
        neww = set()
        for p in [p1,p2]:
            neww.add(p)
            neww.add((p[0]+1,p[1]+1))
            neww.add((p[0]+1,p[1]))
            neww.add((p[0],p[1]+1))
            neww.add((p[0]-1,p[1]+1))
            neww.add((p[0]+1,p[1]-1))
            neww.add((p[0],p[1]-1))
            neww.add((p[0]-1,p[1]))
            neww.add((p[0]-1,p[1]-1))
        #print(neww)
    
    
        
        
        for r in range(grid_row):
            for c in range(grid_column):
                #g.add_edge((r,c),(r,c),0)
                if (0<=(r+1)<=grid_row-1) and (0<=c<=grid_column-1) and (r+1,c) not in neww :
                    g.add_edge((r,c),(r+1,c),1)
                if (0<=(r)<=grid_row-1) and (0<=(c+1)<=grid_column-1) and (r,c+1) not in neww :
                    g.add_edge((r,c),(r,c+1),1)
                if (0<=(r+1)<=grid_row-1) and (0<=(c+1)<=grid_column-1) and (r+1,c+1) not in neww :
                    g.add_edge((r,c),(r+1,c+1),1)
                if (0<=(r-1)<=grid_row-1) and (0<=(c)<=grid_column-1) and (r-1,c) not in neww :
                    g.add_edge((r,c),(r-1,c),1)
                if (0<=(r)<=grid_row-1) and (0<=(c-1)<=grid_column-1) and (r,c-1) not in neww :
                    g.add_edge((r,c),(r,c-1),1)
                if (0<=(r-1)<=grid_row-1) and (0<=(c-1)<=grid_column-1) and (r-1,c-1) not in neww :
                    g.add_edge((r,c),(r-1,c-1),1)
                if (0<=(r-1)<=grid_row-1) and (0<=(c+1)<=grid_column-1) and (r-1,c+1) not in neww :
                    g.add_edge((r,c),(r-1,c+1),1)
                if (0<=(r+1)<=grid_row-1) and (0<=(c-1)<=grid_column-1) and (r+1,c-1) not in neww :
                    g.add_edge((r,c),(r+1,c-1),1)   

    else:
        for r in range(grid_row):
            for c in range(grid_column):
                        
                if (0<=(r+1)<=grid_row-1) and (0<=c<=grid_column-1):
                        g.add_edge((r,c),(r+1,c),1)
                if (0<=(r)<=grid_row-1) and (0<=(c+1)<=grid_column-1):
                    g.add_edge((r,c),(r,c+1),1)
                if (0<=(r+1)<=grid_row-1) and (0<=(c+1)<=grid_column-1):
                    g.add_edge((r,c),(r+1,c+1),1)
                if (0<=(r-1)<=grid_row-1) and (0<=(c)<=grid_column-1):
                    g.add_edge((r,c),(r-1,c),1)
                if (0<=(r)<=grid_row-1) and (0<=(c-1)<=grid_column-1):
                    g.add_edge((r,c),(r,c-1),1)
                if (0<=(r-1)<=grid_row-1) and (0<=(c-1)<=grid_column-1):
                    g.add_edge((r,c),(r-1,c-1),1)
                if (0<=(r-1)<=grid_row-1) and (0<=(c+1)<=grid_column-1):
                    g.add_edge((r,c),(r-1,c+1),1)
                if (0<=(r+1)<=grid_row-1) and (0<=(c-1)<=grid_column-1):
                    g.add_edge((r,c),(r+1,c-1),1)

#def path_1(state,mx_new,my_new):
#    
#    
#    a = len(mx_new)
#    print(a)
#    m_x, m_y, b_x, b_y, food = state           
def path_1(state,mx_new,my_new):
    
    
    a = len(mx_new)
    print(a)
    m_x, m_y, b_x, b_y, food=state
    f_x, f_y = food_coords(food)            
                    
    
    for t in range(a):
            print(t)
            if (t < len(mx_new)):
                
                b_x_new =[]
                b_y_new=[]
                state1 = (mx_new[t],my_new[t],b_x,b_y,food)
                mx,my,b_x,b_y, food= state1
                
                for bx, by in zip(b_x, b_y):
                    food_step = False
                    for fx, fy in zip(f_x, f_y):
    
                        if max(abs(fx-bx), abs(fy-by)) <= baby_view:
                            food_step = True  
                            b_x_new.append(bx + np.sign(fx-bx))
                            b_y_new.append(by + np.sign(fy-by))
                            break
                    if food_step: continue
                    if max(abs(mx-bx), abs(my-by)) <= baby_view:
                        #print("*")
                        print("oops")
                        b_x_new.append(bx + np.sign(mx-bx))
                        b_y_new.append(by + np.sign(my-by))
                        continue
                    b_x_new.append(bx)
                    b_y_new.append(by)
                    
                print(b_x_new,b_y_new)
                if (t < len(mx_new)-1):
                    state = (mx_new[t+1],my_new[t+1],tuple(b_x_new),tuple(b_y_new),food)
                    plot_state(state)
                    pt.pause(10.)
                
            
                
                
           # pt.pause(10.)
        
    m_x, m_y, b_x, b_y, food = state           
    f_x, f_y = food_coords(food)            
    #print("hi")       
    #print(m_x,m_y) 
    #for fx, fy in zip(f_x, f_y):
    f = Graph()
    make_graph(f,state)
    for fx, fy in zip(f_x, f_y):
        food_1 = True
        #print("hello")
        dijkstra(f, f.get_vertex((m_x,m_y)), f.get_vertex((fx,fy))) 
        target = f.get_vertex((fx,fy))
        path1 = [target.get_id()]
        shortest(target, path1)
        D= (path1[::-1])
        mx_new=[]
        my_new=[]
        for i in range(len(D)):
            mx_new.append(D[i][0])
            my_new.append(D[i][1])
        #print(mx_new,my_new)
        for t in range(12):
            #print(t+a)
            
            if (t < len(mx_new)):
                
                b_x_new =[]
                b_y_new=[]
                state1 = (mx_new[t],my_new[t],b_x,b_y,food)
                mx,my,b_x,b_y, food= state
                
                for bx, by in zip(b_x, b_y):
                    #print("there")
                    food_step = False

                    for fx, fy in zip(f_x, f_y):
    
                        if max(abs(fx-bx), abs(fy-by)) <= baby_view:
                            food_step = True  
                            b_x_new.append(bx + np.sign(fx-bx))
                            b_y_new.append(by + np.sign(fy-by))
                            #print("here")
                            #print(b_x_new)
                            #print(b_y_new)
                            break

                    if food_step: continue
                    if max(abs(mx-bx), abs(my-by)) <= baby_view:
                        #print("*")
                        b_x_new.append(bx + np.sign(mx-bx))
                        b_y_new.append(by + np.sign(my-by))
                        continue
                    #print(bx)
                    b_x_new.append(bx)
                    #print(by)
                
                    b_y_new.append(by)
                #print(b_x_new)
                #print(b_y_new)
                if (t < len(mx_new)-1):
                    state = (mx_new[t+1],my_new[t+1],tuple(b_x_new),tuple(b_y_new),food)
                    plot_state(state)
                    pt.pause(10.)
                else:
                    
                    state =(mx_new[t],my_new[t],tuple(b_x_new),tuple(b_y_new),food)
                    #state = move(state)

                    plot_state(state)
                    pt.pause(10.)
                
                    pt.show()
        if food_1 : break
        
    return state

def path_2(state,mx_new,my_new):
    a = len(mx_new)
            #prin(a)
    b_x_new=[]
    b_y_new=[]
    m_x, m_y, b_x, b_y, food=state
    f_x, f_y = food_coords(food)          
    for t in range(a):
        #print(t)
        if (t < len(mx_new)):
            
            b_x_new =[]
            b_y_new=[]
            state1 = (mx_new[t],my_new[t],b_x,b_y,food)
            mx,my,b_x,b_y, food= state1
            
            for bx, by in zip(b_x, b_y):
                food_step = False
                for fx, fy in zip(f_x, f_y):
    
                    if max(abs(fx-bx), abs(fy-by)) <= baby_view:
                        food_step = True  
                        b_x_new.append(bx + np.sign(fx-bx))
                        b_y_new.append(by + np.sign(fy-by))
                        break
                if food_step: continue
                if max(abs(mx-bx), abs(my-by)) <= baby_view:
                    #print("*")
                    #print("oops")
                    b_x_new.append(bx + np.sign(mx-bx))
                    b_y_new.append(by + np.sign(my-by))
                    continue
                b_x_new.append(bx)
                b_y_new.append(by)
                
            #print(b_x_new,b_y_new)
            if (t < len(mx_new)-1):
                state = (mx_new[t+1],my_new[t+1],tuple(b_x_new),tuple(b_y_new),food)
                plot_state(state)
                pt.pause(10.)
            
    m_x, m_y, b_x, b_y, food = state           
    f_x, f_y = food_coords(food)            
            
    #print(m_x,m_y) 
    #for fx, fy in zip(f_x, f_y):
    f = Graph()
    make_graph(f,state)
    for fx, fy in zip(f_x, f_y):
        food_1 = True
        dijkstra(f, f.get_vertex((m_x,m_y)), f.get_vertex((fx,fy))) 
        target = f.get_vertex((fx,fy))
        path1 = [target.get_id()]
        shortest(target, path1)
        D= (path1[::-1])
        D.pop()
        mx_new=[]
        my_new=[]
        for i in range(len(D)):
            mx_new.append(D[i][0])
            my_new.append(D[i][1])
        
        for t in range(12):
            #print(t+a)
            
            if (t < len(mx_new)):
                
                b_x_new =[]
                b_y_new=[]
                state1 = (mx_new[t],my_new[t],b_x,b_y,food)
                mx,my,b_x,b_y, food= state
                for bx, by in zip(b_x, b_y):
                    food_step = False
                    for fx, fy in zip(f_x, f_y):
        
                        if max(abs(fx-bx), abs(fy-by)) <= baby_view:
                            food_step = True  
                            b_x_new.append(bx + np.sign(fx-bx))
                            b_y_new.append(by + np.sign(fy-by))
                            break
                    if food_step: continue
                    if max(abs(mx-bx), abs(my-by)) <= baby_view:
                        #print("*")
                        #print("oops")
                        b_x_new.append(bx + np.sign(mx-bx))
                        b_y_new.append(by + np.sign(my-by))
                        continue
                    b_x_new.append(bx)
                    b_y_new.append(by)
                
                if (t < len(mx_new)-1):
                    state = (mx_new[t+1],my_new[t+1],tuple(b_x_new),tuple(b_y_new),food)
                    plot_state(state)
                    pt.pause(10.)
                else:
                    
                    state =(mx_new[t]-1,my_new[t]-1,tuple(b_x_new),tuple(b_y_new),food)
                    m_x,m_y,b_x_new,b_y_new,food = state1
                    pt.pause(10.)
                    plot_state(state)
                
                    pt.show()
                    m_x_new = mx_new[t]-1
                    m_y_new = my_new[t]-1
            
                
        if food_1:break
    return state
                     

def Hand_Coded_Policy(state,selfish=1,empathy=-1):
    m_x, m_y, b_x, b_y, food = state
    f_x, f_y = food_coords(food)
    if selfish > 0:
        if empathy < 0 or empathy ==0:
            
            for fx, fy in zip(f_x, f_y):
                food_1 = True
                g = Graph()
                flag = True
                make_graph(g,state,flag)
                dijkstra(g, g.get_vertex((m_x,m_y)), g.get_vertex((fx,fy))) 
                target = g.get_vertex((fx,fy))
                path = [target.get_id()]
                #print(path)
                shortest(target, path)
                A=[]
                A= (path[::-1])
                #print(A)
                mx_new = []
                my_new = []
                for i in range(len(A)):
                    mx_new.append(A[i][0])
                    my_new.append(A[i][1])
                for t in range(12):
                    if t < len(mx_new):
                        
                        state = (mx_new[t],my_new[t],b_x,b_y,food)
                        
                    #print(t)
                    plot_state(state)
                    pt.show()
                    pt.pause(10.)
                if food_1:
                    break
#            call Dijkstra’s algorithm with robot as source and food as destination
#            he path i.e sequence of action that robot will take per time step
        elif empathy > 0:
            B1 =[]
            B2 =[]
            for bx, by in zip(b_x, b_y):
                B1.append(bx)
                B2.append(by)
            #print(B1,B2)
            
            g = Graph()
            make_graph(g,state)
            dijkstra(g, g.get_vertex((m_x,m_y)), g.get_vertex((B1[1],B2[1]))) 
            target = g.get_vertex((B1[1],B2[1]))
            path = [target.get_id()]
            shortest(target, path)
            A= (path[::-1])
            A.pop()
            #print(len(A))
            dijkstra(g, g.get_vertex((m_x,m_y)), g.get_vertex((B1[0],B2[0]))) 
            target = g.get_vertex((B1[0],B2[0]))
            path = [target.get_id()]
            shortest(target, path)
            B= (path[::-1])
            B.pop()
            #print(len(B))
            a = len(A)
            b= len(B)
            #print(B)
            if a<=b:
                mx_new = []
                my_new = []
                for i in range(len(A)):
                    mx_new.append(A[i][0])
                    my_new.append(A[i][1])
                c = path_1(state,mx_new,my_new)
                m_x, m_y, b_x, b_y, food = c
                
                dijkstra(g, g.get_vertex((m_x,m_y)), g.get_vertex((B1[0],B2[0]))) 
                target = g.get_vertex((B1[0],B2[0]))
                path = [target.get_id()]
                shortest(target, path)
                A= (path[::-1])
                A.pop()
                a = len(A)
                mx_new = []
                my_new = []
                for i in range(len(A)):
                    mx_new.append(A[i][0])
                    my_new.append(A[i][1])
                s
                c = path_1(state,mx_new,my_new)
            else:
                mx_new = []
                my_new = []
                for i in range(len(B)):
                    mx_new.append(B[i][0])
                    my_new.append(B[i][1])
                #print(mx_new)
                #print(my_new)
                
                c1=path_1(state,mx_new,my_new)
                m_x, m_y, b_x, b_y, food = c1
                #print(m_x)
                #print(m_y)
                h = Graph()
                make_graph(h,c1)
                dijkstra(h, h.get_vertex((m_x,m_y)), h.get_vertex((B1[1],B2[1]))) 
                target = h.get_vertex((B1[1],B2[1]))
                path = [target.get_id()]
                shortest(target, path)
                A= (path[::-1])
                A.pop()
               # print(A)
                a = len(A)
                mx_new = []
                my_new = []
                for i in range(len(A)):
                    mx_new.append(A[i][0])
                    my_new.append(A[i][1])
                c = path_1(c1,mx_new,my_new)
                
                
                                            
                        
                        

                    
    elif selfish < 0:
        
        if empathy < 0 or empathy == 0:
            #print("0")
            plot_state(state)
            pt.show()
            pt.pause(10.)
            state = move(state)
    
            for t in range(12):

                #print(t+1)
                state = move(state)

                plot_state(state)
                pt.show()
                pt.pause(10.)
            
            
        elif empathy > 0:
            B1 =[]
            B2 =[]
            for bx, by in zip(b_x, b_y):
                B1.append(bx)
                B2.append(by)
            #print(B1,B2)
            
            g = Graph()
            make_graph(g,state)
            dijkstra(g, g.get_vertex((m_x,m_y)), g.get_vertex((B1[1],B2[1]))) 
            target = g.get_vertex((B1[1],B2[1]))
            path = [target.get_id()]
            shortest(target, path)
            A= (path[::-1])
            A.pop()
            #print(len(A))
            dijkstra(g, g.get_vertex((m_x,m_y)), g.get_vertex((B1[0],B2[0]))) 
            target = g.get_vertex((B1[0],B2[0]))
            path = [target.get_id()]
            shortest(target, path)
            B= (path[::-1])
            B.pop()
            #print(len(B))
            a = len(A)
            b= len(B)
            #print(B)
            if a<=b:
                mx_new = []
                my_new = []
                for i in range(len(A)):
                    mx_new.append(A[i][0])
                    my_new.append(A[i][1])
                c1 = path_2(state,mx_new,my_new)
                m_x, m_y, b_x, b_y, food = c1
                
                dijkstra(g, g.get_vertex((m_x,m_y)), g.get_vertex((B1[0],B2[0]))) 
                target = g.get_vertex((B1[0],B2[0]))
                path = [target.get_id()]
                shortest(target, path)
                A= (path[::-1])
                A.pop()
                a = len(A)
                mx_new = []
                my_new = []
                for i in range(len(A)):
                    mx_new.append(A[i][0])
                    my_new.append(A[i][1])
                
                c = path_2(c1,mx_new,my_new)
            else:
                mx_new = []
                my_new = []
                for i in range(len(B)):
                    mx_new.append(B[i][0])
                    my_new.append(B[i][1])
                #print(mx_new)
                #print(my_new)
                
                c1=path_2(state,mx_new,my_new)
                m_x, m_y, b_x, b_y, food = c1
                #print(m_x)
                #print(m_y)
                h = Graph()
                make_graph(h,c1)
                dijkstra(h, h.get_vertex((m_x,m_y)), h.get_vertex((B1[1],B2[1]))) 
                target = h.get_vertex((B1[1],B2[1]))
                path = [target.get_id()]
                shortest(target, path)
                A= (path[::-1])
                A.pop()
                #print(A)
                a = len(A)
                mx_new = []
                my_new = []
                for i in range(len(A)):
                    mx_new.append(A[i][0])
                    my_new.append(A[i][1])
                c = path_2(c1,mx_new,my_new)
                mx,my,b_x,b_y,food = c
                b_x_new =[]
                b_y_new=[]
                for bx, by in zip(b_x, b_y):
                    food_step = False
                    for fx, fy in zip(f_x, f_y):
        
                        if max(abs(fx-bx), abs(fy-by)) <= baby_view:
                            food_step = True  
                            b_x_new.append(bx + np.sign(fx-bx))
                            b_y_new.append(by + np.sign(fy-by))
                            break
                    if food_step: continue
                    if max(abs(mx-bx), abs(my-by)) <= baby_view:
                        #print("*")
                        #print("oops")
                        b_x_new.append(bx + np.sign(mx-bx))
                        b_y_new.append(by + np.sign(my-by))
                        continue
                    b_x_new.append(bx)
                    b_y_new.append(by)
                state = mx,my,tuple(b_x_new),tuple(b_y_new),food
#                pt.show()
                t=0
                while my > 0 or mx >0:
                    
                    t=t+1
                    if mx >0:
                        mx= mx-1 
                    else:
                        mx =0
                    if my >0:
                        my = my-1
                    else:
                        my = 0
                    state = mx,my,b_x_new,b_y_new,food
                    if t >5:
                        break
                    
                    pt.pause(10.)
                    plot_state(state)
                    pt.show()
#            
            
            
            
                
#		call Dijkstra’s algorithm with robot as source and subject1/subject 2 as destination
#		call Dijkstra’s algorithm with robot as source and food as destination
#		feed the path i.e sequence of action that robot will take per time step

    elif selfish == 0:
        if empathy < 0 or empathy == 0:
            #print("")
            #print("0")
            plot_state(state)
            pt.show()
            pt.pause(10.)
            state = move(state)
    
            for t in range(12):
                    
                print(t+1)
                plot_state(state)
                pt.show()
                pt.pause(10.)
#		Robot stays at the same location
        elif empathy > 0:
            B1 =[]
            B2 =[]
            for bx, by in zip(b_x, b_y):
                B1.append(bx)
                B2.append(by)
            #print(B1,B2)
            
            g = Graph()
            make_graph(g,state)
            dijkstra(g, g.get_vertex((m_x,m_y)), g.get_vertex((B1[1],B2[1]))) 
            target = g.get_vertex((B1[1],B2[1]))
            path = [target.get_id()]
            shortest(target, path)
            A= (path[::-1])
            A.pop()
            #print(len(A))
            dijkstra(g, g.get_vertex((m_x,m_y)), g.get_vertex((B1[0],B2[0]))) 
            target = g.get_vertex((B1[0],B2[0]))
            path = [target.get_id()]
            shortest(target, path)
            B= (path[::-1])
            B.pop()
            #print(len(B))
            a = len(A)
            b= len(B)
            #print(B)
            if a<=b:
                mx_new = []
                my_new = []
                for i in range(len(A)):
                    mx_new.append(A[i][0])
                    my_new.append(A[i][1])
                c1 = path_2(state,mx_new,my_new)
                m_x, m_y, b_x, b_y, food = c1
                
                dijkstra(g, g.get_vertex((m_x,m_y)), g.get_vertex((B1[0],B2[0]))) 
                target = g.get_vertex((B1[0],B2[0]))
                path = [target.get_id()]
                shortest(target, path)
                A= (path[::-1])
                A.pop()
                a = len(A)
                mx_new = []
                my_new = []
                for i in range(len(A)):
                    mx_new.append(A[i][0])
                    my_new.append(A[i][1])
                
                c = path_2(c1,mx_new,my_new)
            else:
                mx_new = []
                my_new = []
                for i in range(len(B)):
                    mx_new.append(B[i][0])
                    my_new.append(B[i][1])
                #print(mx_new)
                #print(my_new)
                
                c1=path_2(state,mx_new,my_new)
                m_x, m_y, b_x, b_y, food = c1
                #print(m_x)
                #print(m_y)
                h = Graph()
                make_graph(h,c1)
                dijkstra(h, h.get_vertex((m_x,m_y)), h.get_vertex((B1[1],B2[1]))) 
                target = h.get_vertex((B1[1],B2[1]))
                path = [target.get_id()]
                shortest(target, path)
                A= (path[::-1])
                A.pop()
                #print(A)
                a = len(A)
                mx_new = []
                my_new = []
                for i in range(len(A)):
                    mx_new.append(A[i][0])
                    my_new.append(A[i][1])
                c = path_2(c1,mx_new,my_new)
                mx,my,b_x,b_y,food = c
                b_x_new =[]
                b_y_new=[]
                for bx, by in zip(b_x, b_y):
                    food_step = False
                    for fx, fy in zip(f_x, f_y):
        
                        if max(abs(fx-bx), abs(fy-by)) <= baby_view:
                            food_step = True  
                            b_x_new.append(bx + np.sign(fx-bx))
                            b_y_new.append(by + np.sign(fy-by))
                            break
                    if food_step: continue
                    if max(abs(mx-bx), abs(my-by)) <= baby_view:
                        #print("*")
                        #print("oops")
                        b_x_new.append(bx + np.sign(mx-bx))
                        b_y_new.append(by + np.sign(my-by))
                        continue
                    b_x_new.append(bx)
                    b_y_new.append(by)
                state = mx,my,tuple(b_x_new),tuple(b_y_new),food
#                pt.show()
                t=0
                while my > 0 or mx >0:
                    
                    t=t+1
                    if mx >0:
                        mx= mx-1 
                    else:
                        mx =0
                    if my >0:
                        my = my-1
                    else:
                        my = 0
                    state = mx,my,b_x_new,b_y_new,food
                    if t >5:
                        break
                    
                    pt.pause(10.)
                    plot_state(state)
                    pt.show()
                        

#		call Dijkstra’s algorithm with robot as source and subject1/subject 2 as destination
#		call Dijkstra’s algorithm with robot as source and food as destination
#		feed the path i.e sequence of action that robot will take per time step
    else :
        print("wrong value of selfish and empathy")

#def make_graph(state,obstacle= True):
#    m_x, m_y, b_x, b_y, food = state
    
                    
  
if __name__ == "__main__":
    
    

    
#    for v in g:
#        for w in v.get_connections():
#            vid = v.get_id()
#            wid = w.get_id()
#            print( '() %s , %s, %3d)'  % ( vid, wid, v.get_weight(w)))

    state = (1, 0, (0, 0), (4,5), (False, False,True, False))
    #state = (5, 0, (0, 0), (5, 5), (False, False,True, False))
    #state = (0, 0, (0, 0), (5, 5), (False, False, True, False))
    # state = (0, 0, (0, 0), (0, 0), (False, False, False, True))
    # state = (5, 5, (5, 5), (5, 5), (True, True, True, True))
    # state = (0, 0, (0, 0), (0, 0), (False, False, False, True))
    # state = (5, 5, (5, 5), (5, 5), (True, True, True, True))
    #state = index_to_state(np.random.randint(num_states)) # random initial
    m_x, m_y, b_x, b_y, food = state
    f_x, f_y = food_coords(food)
   
# Var selfish and empathy value from -1 to 1
    Hand_Coded_Policy(state,selfish =1,empathy = 1)
