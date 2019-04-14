from search import Problem
from search import breadth_first_tree_search
from search import depth_first_tree_search
from search import iterative_deepening_search
from search import astar_search
import numpy as np
import matplotlib.pyplot as plt
import time
import math

class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __lt__(self,node):
        return self.x < node.x
    

class MatrixProblem(Problem):
    """ Problem of finding a certain spot in a matrix """
    def __init__(self, initial, goal, matrix, heuristic = 1):
        Problem.__init__(self, initial, goal)
        self.matrix = matrix
        self.heuristic = heuristic

    def actions(self, state):
        a = []

        if self.matrix.item(state.x+1,state.y) != -1 and self.matrix.item(state.x+1,state.y) != 1:
            a.append(Point(state.x+1,state.y))

        if self.matrix.item(state.x-1,state.y) != -1 and self.matrix.item(state.x-1,state.y) != 1:
            a.append(Point(state.x-1,state.y))

        if self.matrix.item(state.x,state.y+1) != -1 and self.matrix.item(state.x,state.y+1) != 1:
            a.append(Point(state.x,state.y+1))

        if self.matrix.item(state.x,state.y-1) != -1 and self.matrix.item(state.x,state.y-1) != 1:
            a.append(Point(state.x,state.y-1))
        
        # Diagonals
        if self.matrix.item(state.x+1,state.y+1) != -1 and self.matrix.item(state.x+1,state.y+1) != 1:
            a.append(Point(state.x+1,state.y+1))
        
        if self.matrix.item(state.x+1,state.y-1) != -1 and self.matrix.item(state.x+1,state.y-1) != 1:
            a.append(Point(state.x+1,state.y-1))
        
        if self.matrix.item(state.x-1,state.y-1) != -1 and self.matrix.item(state.x-1,state.y-1) != 1:
            a.append(Point(state.x-1,state.y-1))
        
        if self.matrix.item(state.x-1,state.y+1) != -1 and self.matrix.item(state.x-1,state.y+1) != 1:
            a.append(Point(state.x-1,state.y+1))  

        return a

    def h(self, node):
        x1,y1 = matrix_to_coordinates(node.state.x,node.state.y)
        x2,y2 = matrix_to_coordinates(self.goal.x,self.goal.y)
        if self.heuristic == 1:
            return x2-x1
        else: 
            return math.sqrt((x2-x1)**2+(y2-y1)**2)

    def goal_test(self, state):
        return state.x == goal.x and state.y == goal.y
    
    def result(self, state, action):
        state = action
        self.matrix[state.x][state.y] = 1
        return action
        

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end=" ")
        print("")

def matrix_to_coordinates(i,j):
    x = j-10
    y = 60-i
    return x,y

def init_matrix(initial,goal):
    matrix = np.zeros((61,81),dtype=int)
    matrix[:,10] = -1
    matrix[:,70] = -1
    matrix[20:60,30] = -1
    matrix[0:40,50] = -1
    matrix[0,10:70] = -1
    matrix[60,10:70] = -1
    matrix[initial.x,initial.y] = 1
    matrix[goal.x,goal.y] = 2
    return matrix

def paint_solution(node,matrix):
    for point in node.solution():
        matrix[point.x][point.y] = 3

def plot_result(matrix,initial,goal,title):
    plt.axis([-10,70,0,60])
    plt.grid()

    for i in range(61):
        for j in range(81):
            if matrix.item(i,j) != 0:
                x,y = matrix_to_coordinates(i,j)
                if matrix.item(i,j) == -1:
                    plt.plot([x],[y],marker='.',color='k')
                elif matrix.item(i,j) == 1:
                    plt.plot([x],[y],marker='x',color='c')
                else:
                    plt.plot([x],[y],marker='x',color='r')

    x,y = matrix_to_coordinates(initial.x,initial.y)
    plt.plot([x],[y],marker='x',color='y')
    x,y = matrix_to_coordinates(goal.x,goal.y)   
    plt.plot([x],[y],marker='x',color='b')

    plt.title(title)

    plt.show()
    plt.clf()

# Initial
initial = Point(50,20)
goal = Point(10,60)
matrix = init_matrix(initial,goal)
plot_result(matrix,initial,goal,'Mapa do ambiente inicial')

# BFS
initial = Point(50,20)
goal = Point(10,60)

matrix = init_matrix(initial,goal)

search_problem = MatrixProblem(initial,goal,matrix)
start = time.time()

node = breadth_first_tree_search(search_problem)

end = time.time()
paint_solution(node,matrix)
print(''.join(['BFS search time: ',str(end - start)]))
plot_result(matrix,initial,goal,''.join(['Busca em expansão (t = ', str(round(end - start,4)), ' s)']))

# DFS
initial = Point(50,20)
goal = Point(10,60)
matrix = init_matrix(initial,goal)
search_problem = MatrixProblem(initial,goal,matrix)
start = time.time()

node = depth_first_tree_search(search_problem)

end = time.time()
paint_solution(node,matrix)
print(''.join(['DFS time: ',str(end - start)]))
plot_result(matrix,initial,goal,''.join(['Busca em profundidade (t = ', str(round(end - start,4)), ' s)']))

# BFS
initial = Point(20,60)
goal = Point(10,60)

matrix = init_matrix(initial,goal)

search_problem = MatrixProblem(initial,goal,matrix)
start = time.time()

node = breadth_first_tree_search(search_problem)

end = time.time()
paint_solution(node,matrix)
print(''.join(['BFS easy search time: ',str(end - start)]))
plot_result(matrix,initial,goal,''.join(['Busca em expansão (t = ', str(round(end - start,4)), ' s)']))

# DFS
initial = Point(20,60)
goal = Point(10,60)
matrix = init_matrix(initial,goal)
search_problem = MatrixProblem(initial,goal,matrix)
start = time.time()

node = depth_first_tree_search(search_problem)

end = time.time()
paint_solution(node,matrix)
print(''.join(['DFS easy time: ',str(end - start)]))
plot_result(matrix,initial,goal,''.join(['Busca em profundidade (t = ', str(round(end - start,4)), ' s)']))


# A*
initial = Point(50,20)
goal = Point(10,60)
matrix = init_matrix(initial,goal)
search_problem = MatrixProblem(initial,goal,matrix)
start = time.time()

node = astar_search(search_problem)

end = time.time()
paint_solution(node,matrix)
print(''.join(['A* time bad heuristic: ',str(end - start)]))
plot_result(matrix,initial,goal,''.join(['Busca A* (t = ', str(round(end - start,4)), ' s)']))

# A*
initial = Point(20,60)
goal = Point(10,60)
matrix = init_matrix(initial,goal)
search_problem = MatrixProblem(initial,goal,matrix)
start = time.time()

node = astar_search(search_problem)

end = time.time()
paint_solution(node,matrix)
print(''.join(['A* easy time bad heuristic: ',str(end - start)]))
plot_result(matrix,initial,goal,''.join(['Busca A* (t = ', str(round(end - start,4)), ' s)']))

# A*
initial = Point(50,20)
goal = Point(10,60)
matrix = init_matrix(initial,goal)
search_problem = MatrixProblem(initial,goal,matrix,2)
start = time.time()

node = astar_search(search_problem)

end = time.time()
paint_solution(node,matrix)
print(''.join(['A* time: ',str(end - start)]))
plot_result(matrix,initial,goal,''.join(['Busca A* (t = ', str(round(end - start,4)), ' s)']))

# A*
initial = Point(20,60)
goal = Point(10,60)
matrix = init_matrix(initial,goal)
search_problem = MatrixProblem(initial,goal,matrix,2)
start = time.time()

node = astar_search(search_problem)

end = time.time()
paint_solution(node,matrix)
print(''.join(['A* easy time: ',str(end - start)]))
plot_result(matrix,initial,goal,''.join(['Busca A* (t = ', str(round(end - start,4)), ' s)']))

