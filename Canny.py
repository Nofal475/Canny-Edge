import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import sys

def filter_size_and_gradient(sigma,T):
    if sigma < 0.5:
        sigma = 0.5
    if T > 1 and T < 0:
        T = 0.1
    
    shalf = round(math.sqrt(-math.log(T) * 2 * sigma ** 2))
    N = 2 * shalf + 1  # N is total mask size
    print(f"Mask Size: {N}")

    map = np.linspace(-shalf,shalf, N)
    [Y, X] = np.meshgrid (map, map)
    print(f"Horizontal Mask:\n\n{X}")
    print(f"\nVertical Mask:\n\n{Y}")
    
    gx = []
    gy = []
    gx = Gaus_x(X,Y,sigma)
    print(f"\nGx:\n\n{gx}")
    gy = Gaus_y(X,Y,sigma)
    print(f"\nGy:\n\n{gy}")
    return N,gx, gy

def Gaus_x(X,Y,sigma):
    Gx = []
    scale = 255
    for i in range(0, len(X)):
        row = []
        for j in range(0, len(X[0])):
            e = math.exp((-X[i][j]**2 -Y[i][j]**2)/2*sigma**2)
            deriv=((-X[i][j]/sigma**2)*e)
            row.append(round(scale * deriv)) # for scaling
        Gx.append(row)
    
    return Gx
        
def Gaus_y(X,Y,sigma):
    Gy = []
    scale = 255
    for i in range(0, len(X)):
        row = []
        for j in range(0, len(X[0])):
            e = math.exp((-X[i][j]**2 -Y[i][j]**2)/2*sigma**2)
            deriv=((-Y[i][j]/sigma**2)*e)
            row.append(round(scale * deriv)) # for scaling
        Gy.append(row)
    
    return Gy    


def conv(N,image,g):
    
    g  = np.flipud(g) # flipped due to no maxima supression
    image_columns = image.shape[1]
    image_rows = image.shape[0]
    
    output_rows = image_rows - N + 1
    output_columns = image_columns - N + 1
    output = np.zeros((output_rows, output_columns)) 
    
    for i in range(output_rows):
        for j in range(output_columns):
            output[i, j] = np.sum(g * image[i: i + N, j: j + N])

    return output


def magnitude(fx,fy):
    
    magnew = [] 
    scale = 255
    
    for i in range(0, len(fx)+2):  #zero padding - this is enough for mag supressed also
        row = []
        for j in range(0, len(fy[1])+2):
                row.append(0)
        magnew.append(row)
        
    for i in range(0, len(fx)):
        row = []
        for j in range(0, len(fy[1])):
            mag = math.sqrt((fx[i][j]**2)+(fy[i][j]**2))
            magnew[i+1][j+1] = (mag/scale) # for scaling
      
    return magnew


def non_maxima_suppression(fx, fy, mag_suppressed):
    
    mag = mag_suppressed
    rows = len(fx)
    columns = len(fy[1])
        
    for i in range(1, len(fx)):
        for j in range(1, len(fy[1])):
            angle =  np.arctan2(fy[i][j], fx[i][j])
            angle = np.rad2deg(angle) 
            if angle < 0:
                angle = angle + 360 
            if ((angle > 0 and angle < 22.5) or (angle > 157.5 and angle < 202.5) or (angle > 337.5 and angle < 360)):              
                if (mag[i][j] > mag[i][j+1]) or (mag[i][j] > mag[i][j-1]):
                    mag_suppressed[i][j] = mag_suppressed[i][j]
                else:
                    mag_suppressed[i][j] = 0
            if ((angle > 22.5 and angle < 67.5) or (angle > 202.5 and angle < 247.5)):          
                if (mag[i][j] > mag[i+1][j+1]) or (mag[i][j] > mag[i-1][j-1]):
                    mag_suppressed[i][j] = mag_suppressed[i][j]
                else:
                    mag_suppressed[i][j] = 0
            if ((angle > 67.5 and angle < 112.5) or (angle > 247.5 and angle < 292.5)):
                if (mag[i][j] > mag[i+1][j]) or (mag[i][j] > mag[i-1][j]):
                    mag_suppressed[i][j] = mag_suppressed[i][j]
                else:
                    mag_suppressed[i][j] = 0
            if ((angle > 112.5 and angle < 157.5) or (angle > 292.5 and angle < 337.5)):   
                if (mag[i][j] > mag[i+1][j-1]) or (mag[i][j] > mag[i-1][j+1]):
                    mag_suppressed[i][j] = mag_suppressed[i][j]
                else:
                    mag_suppressed[i][j] = 0

              
    return mag_suppressed


def hysteresis(th, tl, mag_suppressed):
    
    th = int(th)
    tl = int(tl)
    
    ced = np.zeros((len(mag_suppressed),len(mag_suppressed[1]),3))
    
    for i in range(len(mag_suppressed)):
        for j in range(len(mag_suppressed[1])-1):
            if  mag_suppressed[i][j] > th:
                ced[i][j][0] = 1
                ced[i][j][1] = 1
                ced[i][j][2] = 1  
            
    while True:
        
        done = 1
        for i in range(len(mag_suppressed)):
            for j in range(len(mag_suppressed[1])-1):
                if (ced[i][j][0] == 1):
                    if (mag_suppressed[i+1][j] > tl and ced[i+1][j][0] == 0):
                        ced[i+1][j][0] = 1
                        ced[i+1][j][1] = 1
                        ced[i+1][j][2] = 1
                        done = 0
                    if (mag_suppressed[i-1][j] > tl and ced[i-1][j][0] == 0):
                        ced[i-1][j][0] = 1
                        ced[i-1][j][1] = 1
                        ced[i-1][j][2] = 1
                        done = 0
                    if (mag_suppressed[i][j+1] > tl and ced[i][j+1][0] == 0):
                        ced[i][j+1][0] = 1
                        ced[i][j+1][1] = 1
                        ced[i][j+1][2] = 1
                        done = 0
                    if (mag_suppressed[i][j-1] > tl and ced[i][j-1][0] == 0):
                        ced[i][j-1][0] = 1
                        ced[i][j-1][1] = 1
                        ced[i][j-1][2] = 1
                        done = 0
                    if (mag_suppressed[i-1][j-1] > tl and ced[i-1][j-1][0] == 0):
                        ced[i-1][j-1][0] = 1
                        ced[i-1][j-1][1] = 1
                        ced[i-1][j-1][2] = 1
                        done = 0
                    if (mag_suppressed[i+1][j+1] > tl and ced[i+1][j+1][0] == 0):
                        ced[i+1][j+1][0] = 1
                        ced[i+1][j+1][1] = 1
                        ced[i+1][j+1][2] = 1
                        done = 0
                    if (mag_suppressed[i+1][j-1] > tl and ced[i+1][j-1][0] == 0):
                        ced[i+1][j-1][0] = 1
                        ced[i+1][j-1][1] = 1
                        ced[i+1][j-1][2] = 1
                        done = 0
                    if (mag_suppressed[i-1][j+1] > tl and ced[i-1][j+1][0] == 0):
                        ced[i-1][j+1][0] = 1
                        ced[i-1][j+1][1] = 1
                        ced[i-1][j+1][2] = 1
                        done = 0
                        
        if done == 1:
            break       
    return ced


def canny(img):
    
    sigma = float(input("sigma: "))
    T = float(input("T: "))
    N, gx, gy = filter_size_and_gradient(sigma,T)
    np.shape(gy)
    
    fx = conv(N, img, gx)
    fy = conv(N, img, gy)        
    
    mag = magnitude(fx,fy)
    
    mag_suppressed = non_maxima_suppression(fx, fy, mag)
    
    max_val = np.max(mag_suppressed)

    th = input(f"input th value when max value is {max_val} : ")
    tl = input(f"input tl value when max value is {max_val} : ")

    ced = hysteresis(th, tl, mag_suppressed)
    plt.imshow(ced)

    return ced

target_file = input("type in your target file name with extension: ")
img = cv2.imread(target_file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edge = canny(img)