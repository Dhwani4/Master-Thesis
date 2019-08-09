
import numpy as np
import matplotlib.pyplot as pt
grid_cols = 6 # size of grid world
grid_rows = 6 # size of grid world
baby_view = 1 # how far babies can see
num_babies = 2

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
    ''' make shortest path from v.previous'''
    if v.previous:
        path.append(v.previous.get_id())
        shortest(v.previous, path)
        
    return

import heapq

def dijkstra(aGraph, start, target):
    print ('''Dijkstra's shortest path''')
    # Set the distance for the start node to zero 
    start.set_distance(0)
    unvisited_queue = [(v.get_distance(),v) for v in aGraph]
    heapq.heapify(unvisited_queue)
    # unvisited_queue=[]
    # for v in aGraph:
    #     heapq.heappush(unvisited_queue, (v.get_distance(), v))

    while len(unvisited_queue):
        # Pops a vertex with the smallest distance 
        uv = heapq.heappop(unvisited_queue)
        current = uv[1]
        current.set_visited()

        #for next in v.adjacent:
        for next in current.adjacent:
            # if visited, skip
            if next.visited:
                continue
            new_dist = current.get_distance() + current.get_weight(next)
            
            if new_dist < next.get_distance():
                next.set_distance(new_dist)
                next.set_previous(current)
#                #print ('updated : current = %s next = %s new_dist = %s' \
#                        %(current.get_id(), next.get_id(), next.get_distance()))
        
#                #print ('not updated : current = %s next = %s new_dist = %s' \
#                        %(current.get_id(), next.get_id(), next.get_distance()))

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
    print(m_x)
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
            print("*")
            b_x_new.append(bx + np.sign(m_x-bx))
            b_y_new.append(by + np.sign(m_y-by))
        b_x_new.append(bx)
        b_y_new.append(by)
        

    m_x_new = max(0, min(m_x, grid_cols-1))
    m_y_new = max(0, min(m_y, grid_rows-1))

    new_state = (m_x,m_y, tuple(b_x_new), tuple(b_y_new), food)
    return new_state

def make_graph(g):
    
    grid_row = 6
    grid_column = 6
    for r in range(grid_row):
        for c in range(grid_column):
            g.add_vertex((r,c))
   
        

    for r in range(grid_row+1):
        for c in range(grid_column+1):
                    
            if (0<=(r+1)<=grid_row) and (0<=c<=grid_column):
                    g.add_edge((r,c),(r+1,c),1)
            if (0<=(r)<=grid_row) and (0<=(c+1)<=grid_column):
                g.add_edge((r,c),(r,c+1),1)
            if (0<=(r+1)<=grid_row) and (0<=(c+1)<=grid_column):
                g.add_edge((r,c),(r+1,c+1),1)
            if (0<=(r-1)<=grid_row) and (0<=(c)<=grid_column):
                g.add_edge((r,c),(r-1,c),1)
            if (0<=(r)<=grid_row) and (0<=(c-1)<=grid_column):
                g.add_edge((r,c),(r,c-1),1)
            if (0<=(r-1)<=grid_row) and (0<=(c-1)<=grid_column):
                g.add_edge((r,c),(r-1,c-1),1)
            if (0<=(r-1)<=grid_row) and (0<=(c+1)<=grid_column):
                g.add_edge((r,c),(r-1,c+1),1)
            if (0<=(r+1)<=grid_row) and (0<=(c-1)<=grid_column):
                g.add_edge((r,c),(r+1,c-1),1)
            
    

def Hand_Coded_Policy(state,selfish=1,empathy=-1):
    m_x, m_y, b_x, b_y, food = state
    f_x, f_y = food_coords(food)
    if selfish > 0:
        if empathy < 0 or empathy ==0:
            for fx, fy in zip(f_x, f_y):
                g = Graph()
                make_graph(g)
                dijkstra(g, g.get_vertex((m_x,m_y)), g.get_vertex((fx,fy))) 
                target = g.get_vertex((fx,fy))
                path = [target.get_id()]
                print(path)
                shortest(target, path)
                A=[]
                A= (path[::-1])
                print(A)
                mx_new = []
                my_new = []
                for i in range(len(A)):
                    mx_new.append(A[i][0])
                    my_new.append(A[i][1])
                print(mx_new,my_new)
                for t in range(12):
                    if t < len(mx_new):
                        
                        state = (mx_new[t],my_new[t],b_x,b_y,food)
                        
                    print(t)
                    plot_state(state)
                    pt.show()
                    pt.pause(10.)
                    
#            call Dijkstra’s algorithm with robot as source and food as destination
#            he path i.e sequence of action that robot will take per time step
        elif empathy > 0:
            B1 =[]
            B2 =[]
            for bx, by in zip(b_x, b_y):
                B1.append(bx)
                B2.append(by)
            print(B1,B2)
            g = Graph()
            make_graph(g)
            dijkstra(g, g.get_vertex((m_x,m_y)), g.get_vertex((B1[1],B2[1]))) 
            target = g.get_vertex((B1[1],B2[1]))
            path = [target.get_id()]
            shortest(target, path)
            A= (path[::-1])
            A.pop()
            print(A)
            mx_new = []
            my_new = []
            for i in range(len(A)):
                mx_new.append(A[i][0])
                my_new.append(A[i][1])
            print(mx_new,my_new)
            
            a = len(mx_new)
            print(a)
            
            for t in range(a):
                    print(t)
                    state = (mx_new[t],my_new[t],b_x,b_y,food)
                    plot_state(state)
                    pt.show()
                   # pt.pause(10.)
                
            m_x, m_y, b_x, b_y, food = state           
            f_x, f_y = food_coords(food)            
                    
            print(m_x,m_y) 
            #for fx, fy in zip(f_x, f_y):
            f = Graph()
            make_graph(f)
            dijkstra(f, f.get_vertex((m_x,m_y)), f.get_vertex((5,5))) 
            target = f.get_vertex((5,5))
            path1 = [target.get_id()]
            shortest(target, path1)
            D= (path1[::-1])
            D.pop()
            print(D)
            mx_new=[]
            my_new=[]
            for i in range(len(D)):
                mx_new.append(D[i][0])
                my_new.append(D[i][1])
            print(mx_new,my_new)
            for t in range(12):
                print(t)
                
                if (t < len(mx_new)):
                    
                    b_x_new =[]
                    b_y_new=[]
                    state1 = (mx_new[t],my_new[t],b_x,b_y,food)
                    mx,my,b_x,b_y, food= state
                    
                    for bx, by in zip(b_x, b_y):
                        if max(abs(mx_new[t]-bx), abs(my_new[t]-by)) <= baby_view:
                            print("*")
                            b_x_new.append(bx + np.sign(mx_new[t]-bx))
                            b_y_new.append(by + np.sign(my_new[t]-by))
                    print(b_x_new)
                    print(b_y_new)
                    if (t < len(mx_new)-1):
                        state = (mx_new[t+1],my_new[t+1],tuple(b_x_new),tuple(b_y_new),food)
                        plot_state(state)
                        pt.pause(10.)
                    else:
                        a = (mx_new[t]-1)
                        
                        state =(mx_new[t]-1,my_new[t]-1,tuple(b_x_new),tuple(b_y_new),food)
                        plot_state(state)
                        state = move(state)
                        pt.pause(10.)
                        plot_state(state)
                    
                        pt.show()
                else:
                    state = move(state)
                    plot_state(state)
                    pt.show()
                    
#                    for i in range(len(mx_new)):
#                        
#                    
#                        for bx, by in zip(b_x_new, b_y_new):
#                            if max(abs(mx_new[t]-bx), abs(my_new[t]-by)) <= baby_view:
#                                print("*")
#                                b_x_new.append(bx + np.sign(mx_new[t]-bx))
#                                b_y_new.append(by + np.sign(my_new[t]-by))
#                        pt.pause(.10)
#                        print(b_x_new)
#                        print(b_y_new)
#                        state = (mx_new[t+1],my_new[t+1],tuple(b_x_new),tuple(b_y_new),food)
#                        plot_state(state)
#                        pt.show()    
                        
                    
            
#		call Dijkstra’s algorithm with robot as source and subject1/subject 2 as destination
#		call Dijkstra’s algorithm with robot as source and food as destination
#		feed the path i.e sequence of action that robot will take per time step
    elif selfish < 0:
        
        if empathy < 0 or empathy == 0:
            print("0")
            plot_state(state)
            pt.show()
            pt.pause(10.)
            state = move(state)
    
            for t in range(12):
                    
                print(t+1)
                plot_state(state)
                pt.show()
                pt.pause(10.)
            
            
        elif empathy > 0:
            B1 =[]
            B2 =[]
            for bx, by in zip(b_x, b_y):
                B1.append(bx)
                B2.append(by)
            print(B1,B2)
            g = Graph()
            make_graph(g)
            dijkstra(g, g.get_vertex((m_x,m_y)), g.get_vertex((B1[1],B2[1]))) 
            target = g.get_vertex((B1[1],B2[1]))
            path = [target.get_id()]
            shortest(target, path)
            A= (path[::-1])
            A.pop()
            print(A)
            mx_new = []
            my_new = []
            for i in range(len(A)):
                mx_new.append(A[i][0])
                my_new.append(A[i][1])
            print(mx_new,my_new)
            
            a = len(mx_new)
            print(a)
            
            for t in range(a):
                    print(t)
                    state = (mx_new[t],my_new[t],b_x,b_y,food)
                    plot_state(state)
                    pt.show()
                   # pt.pause(10.)
                
            m_x, m_y, b_x, b_y, food = state           
            f_x, f_y = food_coords(food)            
                    
            print(m_x,m_y) 
            #for fx, fy in zip(f_x, f_y):
            f = Graph()
            make_graph(f)
            dijkstra(f, f.get_vertex((m_x,m_y)), f.get_vertex((5,5))) 
            target = f.get_vertex((5,5))
            path1 = [target.get_id()]
            shortest(target, path1)
            D= (path1[::-1])
            D.pop()
            print(D)
            mx_new=[]
            my_new=[]
            for i in range(len(D)):
                mx_new.append(D[i][0])
                my_new.append(D[i][1])
            print(mx_new,my_new)
            for t in range(12):
                print(t)
                
                if (t < len(mx_new)):
                    
                    b_x_new =[]
                    b_y_new=[]
                    state1 = (mx_new[t],my_new[t],b_x,b_y,food)
                    mx,my,b_x,b_y, food= state
                    
                    for bx, by in zip(b_x, b_y):
                        if max(abs(mx_new[t]-bx), abs(my_new[t]-by)) <= baby_view:
                            print("*")
                            b_x_new.append(bx + np.sign(mx_new[t]-bx))
                            b_y_new.append(by + np.sign(my_new[t]-by))
                    print(b_x_new)
                    print(b_y_new)
                    if (t < len(mx_new)-1):
                        state = (mx_new[t+1],my_new[t+1],tuple(b_x_new),tuple(b_y_new),food)
                        plot_state(state)
                        pt.pause(10.)
                    else:
                        a = (mx_new[t]-1)
                        
                        state =(mx_new[t]-1,my_new[t]-1,tuple(b_x_new),tuple(b_y_new),food)
                        plot_state(state)
                        state = move(state)
                        pt.pause(10.)
                        plot_state(state)
                    
                        pt.show()
            
#		call Dijkstra’s algorithm with robot as source and subject1/subject 2 as destination
#		call Dijkstra’s algorithm with robot as source and food as destination
#		feed the path i.e sequence of action that robot will take per time step

    elif selfish == 0:
        if empathy < 0 or empathy == 0:
            print("")
            print("0")
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
            print("")
        
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

    #state = (1, 0, (0, 0), (5, 5), (False, False,True, False))
    state = (5, 0, (0, 0), (5, 5), (False, False,True, False))
    m_x, m_y, b_x, b_y, food = state
    f_x, f_y = food_coords(food)
    B1= set()
    
    for bx, by in zip(b_x, b_y):
        B1.add((bx,by))
    print(B1)
    
        
    print(m_x)
    print(m_y)
    for fx, fy in zip(f_x, f_y):
        print("food x",fx)
        print("food y",fy)
    for bx, by in zip(b_x,b_y):
        print(bx)
        print(by)
    
    Hand_Coded_Policy(state,selfish =-1,empathy = 1)
#    for t in range(12): # number of time-steps
#
#        # show current state
#        print(t)
#        pt.cla()
#        plot_state(state)
#        pt.show()
#        pt.pause(10.)