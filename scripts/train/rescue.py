import sys
import time

import numpy as np

class Point:
    def __init__(self, x = None, y = None, parent = None, cost = 0):
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = cost

class AStar:
    def __init__(self, map,obstacle_map, target_x, target_y, start_x, start_y):
        self.map=map
        self.obstacle_map = obstacle_map
        self.open_set = []
        self.close_set = []
        self.point = Point
        self.target_x = target_x
        self.target_y = target_y
        self.start_x = start_x
        self.start_y = start_y

    def BaseCost(self, p):
        x_dis = abs(p.x-self.start_x)
        y_dis = abs(p.y - self.start_y)
        # Distance to start point
        # print("base cost",x_dis + y_dis)
        return x_dis + y_dis

    def HeuristicCost(self, p):
        x_dis = abs(self.target_x - p.x)
        y_dis = abs(self.target_y - p.y)
        # Distance to end point
        # print("HeuristicCost",x_dis + y_dis)
        return x_dis + y_dis

    def TotalCost(self, p):
        return self.BaseCost(p) + self.HeuristicCost(p)

    def IsValidPoint(self, x, y):
        if x < 0 or y < 0:
            return False
        elif x >= self.map.shape[0] or y >= self.map.shape[0]:
            return False
        elif self.obstacle_map[x, y] >= 1:
            return False
        elif self.map[x, y] == 0 and x != self.target_x and y != self.target_y:
            return False
        return True

    def IsInPointList(self, p, point_list):
        for point in point_list:
            if point.x == p.x and point.y == p.y:
                return True
        return False

    def IsInOpenList(self, p):
        return self.IsInPointList(p, self.open_set)

    def IsInCloseList(self, p):
        return self.IsInPointList(p, self.close_set)

    def IsStartPoint(self, p):
        return p.x == self.start_x and p.y == self.start_y

    def IsEndPoint(self, p):
        return p.x == self.target_x and p.y == self.target_y

    def RunAndSaveImage(self):
        start_time = time.time()
        start_point = self.point(x=self.start_x, y=self.start_y)
        start_point.cost = self.TotalCost(p=start_point)
        self.open_set.append(start_point)

        while True:
            index = self.SelectPointInOpenList()
            if index < 0:
                print('No path found, algorithm failed!!!')
                return
            p = self.open_set[index]
            # rec = Rectangle((p.x, p.y), 1, 1, color='c')
            # ax.add_patch(rec)
            # self.SaveImage(plt)
            if self.IsEndPoint(p):
                return self.BuildPath(p, start_time)

            del self.open_set[index]
            self.close_set.append(p)

            # Process all neighbors
            x = p.x
            y = p.y

            self.ProcessPoint(x - 1, y, p)
            self.ProcessPoint(x + 1, y, p)
            self.ProcessPoint(x, y-1, p)
            self.ProcessPoint(x, y+1, p)
    # def SaveImage(self, plt):
    #     millis = int(round(time.time() * 1000))
    #     filename = './' + str(millis) + '.png'
    #     plt.savefig(filename)

    def ProcessPoint(self, x, y, parent):
        if not self.IsValidPoint(x, y):
            return  # Do nothing for invalid point
        p = self.point(x=x, y=y)
        if self.IsInCloseList(p):
            return  # Do nothing for visited point
        if not self.IsInOpenList(p):
            p.parent = parent
            p.cost = self.TotalCost(p)
            self.open_set.append(p)
            # print('Process Point [', p.x, ',', p.y, ']', ', cost: ', p.cost)


    def SelectPointInOpenList(self):
        index = 0
        selected_index = -1
        # 初始化为系统最大整数
        min_cost = sys.maxsize
        for p in self.open_set:
            cost = self.TotalCost(p)
            if cost < min_cost:
                min_cost = cost
                selected_index = index
            index += 1
        return selected_index

    def BuildPath(self, p, start_time):
        path = []
        while True:
            path.insert(0, p)  # Insert first
            if self.IsStartPoint(p):
                break
            else:
                p = p.parent
                if p is None:
                    print("Finish the path and Return")
        end_time = time.time()
        actions = []
        # print("!!!!!!!!!len path",len(path))
        for i_path in range(len(path)-1):
            # print(f"({i_path.x},{i_path.y})->")
            actions.append(self.get_action([path[i_path].x, path[i_path].y], [path[i_path+1].x, path[i_path+1].y]))
        path_list = [(i_path.x, i_path.y) for i_path in path]
        return path_list, actions
    def get_action(self,p1, p2):
        if p2[0] - p1[0] == -1:
            action = [1, 0, 0, 0]
        elif p2[0] - p1[0] == 1:
            action = [0, 1, 0, 0]
        elif p2[1] - p1[1] == -1:
            action = [0, 0, 1, 0]
        else:
            action = [0, 0, 0, 1]

        # if p2[1] - p1[1] == -1:
        #     action = [1, 0, 0, 0]
        # elif p2[1] - p1[1] == 1:
        #     action = [0, 1, 0, 0]
        # elif p2[0] - p1[0] == -1:
        #     action = [0, 0, 1, 0]
        # else:
        #     action = [0, 0, 0, 1]
        return action

if __name__ == '__main__':

    map = np.ones((50,50))
    obstacle = np.zeros((50,50))
    for i in range(15):
        x = np.random.randint(low=0,high=50)
        y = np.random.randint(low=0,high=50)
        print(x,y)
        obstacle[x,y] = 2
    print("map",map)

    astar = AStar(obstacle_map=obstacle,map=map,target_x=45,target_y=45,start_x=0,start_y=0)
    astar.RunAndSaveImage()
