import matplotlib.pyplot as plt
import numpy as np


img='new output/figure_1_0.jpg'

image=plt.imread(img)
x_len=len(image)
y_len=len(image[0])

for i in range(x_len):
    for j in range(y_len):
        if image[i][j]>200:
            image[i][j]=1
        else:
            image[i][j]=0



def neighbours(array, x, y):
    """Return 8-neighbours of point p1 of array."""
    return [array[x2, y2] for x2 in range(x-1, x+2)
                            for y2 in range(y-1, y+2)
                            if (-1 < x < array.shape[0] and
                                -1 < y < array.shape[1] and
                                (x != x2 or y != y2) and
                                (0 <= x2 < array.shape[0]) and
                                (0 <= y2 < array.shape[1]))]

def transitions(neighbours):
    n = neighbours + neighbours[0:1]      
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))

def zhangSuen(array):
    changing1 = changing2 = [(-1,-1)]    
    while changing1 or changing2:
        # Step 1
        changing1 = []
        for x in range(1, array.shape[0]-1):
            for y in range(1, array.shape[1]-1):
                p2, p3, p4, p5, p6, p7, p8, p9 = n = neighbours(array, x, y)
                if (array[x, y] == 1 and    
                    2 <= sum(n) <= 6 and    
                    transitions(n) == 1 and 
                    p2 * p4 * p6 == 0 and   
                    p4 * p6 * p8 == 0):     

                    changing1.append((x,y))
        for x, y in changing1: 
            array[x, y] = 0
        # Step 2
        changing2 = []
        for x in range(1, array.shape[0]-1):
            for y in range(1, array.shape[1]-1):
                p2, p3, p4, p5, p6, p7, p8, p9 = n = neighbours(array, x, y)
                if (array[x, y] == 1 and    
                    2 <= sum(n) <= 6 and    
                    transitions(n) == 1 and 
                    p2 * p4 * p8 == 0 and   
                    p2 * p6 * p8 == 0):     
                    changing2.append((x,y))
        for x, y in changing2: 
            array[x, y] = 0
    return array

image=np.array(image)
thinned_image = zhangSuen(image)

for i in range(x_len):
    for j in range(y_len):
        if thinned_image[i][j]==1:
            image[i][j]=255
        else:
            image[i][j]=0
plt.imshow(image)
plt.show()
