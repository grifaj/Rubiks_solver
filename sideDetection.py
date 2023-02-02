import cv2 as cv
import numpy as np
import time

global max_val
global min_val
def find_closest_color(input_colour):
    colour_names =  ['white', 'red', 'blue', 'orange', 'green', 'yellow']
    colour_list = [(255,255,255),(20,18,137),(172,72,13),(37,85,255),(76,155,25),(47,213,254)]
    #print(input_colour)
    l, a, b = cv.split(input_colour)
    l = l[0][0]
    a = a[0][0]
    b = b[0][0]

    for i in range(3):
        c = input_colour[0][0][i]
        if c > max_val[i]: 
            max_val[i] = c
        if c < min_val[i]: 
            min_val[i] = c

     # white
    if l in range(84,184) and a in range(109,133) and b in range(135,155):
        return colour_list[0]
    # orange
    if l in range(76,133) and a in range(129,150) and b in range(163,182):
        return colour_list[3]
    # yellow
    if l in range(101,175) and a in range(107,124) and b in range(170,195):
        return colour_list[5]
    # green
    if l in range(79,118) and a in range(95,108) and b in range(145,163):
        return colour_list[4]    
    # blue
    if l in range(54,103) and a in range(115,131) and b in range(107,130):
        return colour_list[2]
    # red
    if l in range(70,100) and a in range(132,150) and b in range(156,166):
        return colour_list[1]
    #print('not known',input_colour)


def sort_by_location(c):
    print(c)
    x, y, w, h = cv.boundingRect(c)
    return (y, x)

# scale and show image for printing
def showImg(label, img):
    #scale image
    scale = 1
    width = int(img.shape[1] *scale)
    height = int(img.shape[0] *scale)
    dimensions = (width, height)
    img = cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
    cv.imshow(label,img)

def getColours(cube):
    
    grey = cv.cvtColor(cube, cv.COLOR_BGR2GRAY)
    blur = cv.blur(cube,(3,3))
    canny = cv.Canny(blur, 55, 100, L2gradient = True) #60, 100

    showImg('edges',canny)

    # get contours
    contours, hierarchies = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # display contours
    blank = np.zeros(cube.shape, dtype='uint8')
    cv.drawContours(blank, contours, -1, (0,255,0), 1)
    showImg('contours',blank)

    if hierarchies is None:
        return cube
    else:
        hierarchies = hierarchies[0]

    output = cube
    colours = []
    hsv_vals =[]
    used_contours =[]
    # for each contour check if it is a square
    for i in range(len(contours)):
        contour = contours[i]
        hierarchy = hierarchies[i]
        parent = hierarchy[3]

        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        #epsilon = 0.01 * perimeter
        #approx = cv.approxPolyDP(contour, epsilon, True)
        squareness = cv.norm(((perimeter / 4) * (perimeter / 4)) - area)

        # likely candidate for piece
        if parent == -1 and area > 2000 and area < 6000 and squareness < 200:#150
            
            print(area, squareness)

            # add contour to image
            cv.drawContours(output, contours, i, (0,255,0), 2)
            used_contours.append(contour)

            #get average colour
            mask = np.zeros(grey.shape, np.uint8) 
            cv.drawContours(mask, [contour], 0, 255, -1)
            #mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            lab_img = cv.cvtColor(cube, cv.COLOR_BGR2LAB)
            mean = cv.mean(lab_img, mask=mask)[:-1]
            hsv_vals.append([int(a) for a in mean])
            mean = np.uint8([[mean]])
            colours.append(find_closest_color(mean))
    
    if len(colours) == 4:
        # order colours by contour location
        #[colours for _, colours in sorted(zip(used_contours, colours))]
        #print(hsv_vals)
        #print('max', max_val)
        #print('min', min_val)
        #create box in corner for colours
        side_len = 50
        pos = [[0,2],[1,3]]
        for i in range(len(pos)):
            for j in range(len(pos[i])):
                cv.rectangle(output, (i*side_len,j*side_len),((i+1)*side_len,(j+1)*side_len), colours[pos[i][j]],-1)

        return True, output
    
    return False, output

max_val = [0,0,0]
min_val = [255,255,255]

video = True

if video:
    frame_rate = 20 # set frame rate
    prev = 0
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        time_elapsed = time.time() - prev
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if time_elapsed > 1./frame_rate:
            prev = time.time()
            face,output = getColours(frame)
            # Display the resulting frame
            showImg('frame',output)

        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

    cv.waitKey(0)

else: #photo only
    #cube = cv.imread('C:\\Users\\Alfie\\Documents\\uni_work\\year3\\cs310\\github\Rubiks_solver\\good521.JPG')
    cube = cv.imread('/home/grifaj/Documents/y3project/Rubiks_solver/test1.jpg')

    output = getColours(cube)
    showImg('output', output)

    cv.waitKey(0)