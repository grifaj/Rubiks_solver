import cv2 as cv
import numpy as np
import time
import random
from centroidtracker import CentroidTracker
import joblib

global max_val
global min_val

# finds the most sutible colour based on average pixel value
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
    # perhaps get avergae luminiace and calculate based on relative l for total

     # white
    if l in range(84,250) and a in range(105,133) and b in range(125,165):
        return colour_list[0]
    # orange
    if l in range(76,200) and a in range(129,160) and b in range(163,205):
        return colour_list[3]
    # yellow
    if l in range(101,235) and a in range(95,124) and b in range(170,204):
        return colour_list[5]
    # green
    if l in range(79,200) and a in range(66,108) and b in range(145,170):
        return colour_list[4]    
    # blue
    if l in range(54,165) and a in range(112,135) and b in range(89,130):
        return colour_list[2]
    # red
    if l in range(70,180) and a in range(132,180) and b in range(147,185):
        return colour_list[1]
    #print('not known',input_colour)

# gets the 4 colours detected on the face and orders them to add to the state
def order_faces(face_pos, colours):
    combined = zip(face_pos, colours)
    combined = list(combined)

    #order by y, gives top and bottom row
    combined = sorted(combined, key=lambda x: x[0][1])
    top = combined[:2]
    bottom = combined[2:]

    #order top and bottom by x value to give left and right
    top = sorted(top, key=lambda x: x[0][0])
    bottom = sorted(bottom, key=lambda x: x[0][0])

    #combine to give full sorted list
    new_order = top + bottom

    # seperate colours out in new order
    colours = [y for x,y in new_order]

    return colours

# scale and show image for printing
def showImg(label, img):
    #scale image
    scale = 1
    width = int(img.shape[1] *scale)
    height = int(img.shape[0] *scale)
    dimensions = (width, height)
    img = cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
    cv.imshow(label,img)

# returns the colours of the shown face
def getColours(cube):
    
    grey = cv.cvtColor(cube, cv.COLOR_BGR2GRAY)
    blur = cv.blur(cube,(3,3))
    canny = cv.Canny(blur, 55, 100, L2gradient = True) #60, 100

    showImg('edges',canny)

    # get contours
    contours, hierarchies = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    if hierarchies is None:
        return cube
    else:
        hierarchies = hierarchies[0]

    output = cube
    colours = []
    centroids = []
    hsv_vals =[]
    blank = np.zeros(cube.shape, dtype='uint8')
    # for each contour check if it is a square
    for i in range(len(contours)):
        contour = contours[i]
        hierarchy = hierarchies[i]
        parent = hierarchy[3]

        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        squareness = cv.norm(((perimeter / 4) * (perimeter / 4)) - area)

        if parent == -1 and area > 750 and area < 4000 and squareness < 250:

            # display contours
            cv.drawContours(blank, contours, i, (random.randint(0,255), random.randint(0, 255), random.randint(0, 255)), 1)
            showImg('contours',blank)
            #print(area, squareness, perimeter)

        # likely candidate for piece
        #if parent == -1 and area > 2000 and area < 8000 and squareness < 120:

            # add contour to image
            cv.drawContours(output, contours, i, (0,255,0), 2)
            # get coords of top left corner for ordering
            x, y, _, _ = cv.boundingRect(contour)
            #cv.putText(img=output, text=(str(x)+' '+str(y)), org=(x, y), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=0.6, color=(225, 0, 255),thickness=1)

            # get centroid of contour
            M = cv.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))

            #get average colour
            mask = np.zeros(grey.shape, np.uint8) 
            cv.drawContours(mask, [contour], 0, 255, -1)
            #mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            lab_img = cv.cvtColor(cube, cv.COLOR_BGR2LAB)
            mean = cv.mean(lab_img, mask=mask)[:-1]
            hsv_vals.append([int(a) for a in mean])
            mean = np.uint8([[mean]])
            #colours.append(find_closest_color(mean))
            
            #try knn
            colour_names =  ['w', 'r', 'b', 'o', 'g', 'y']
            colour_list = [(255,255,255),(20,18,137),(172,72,13),(37,85,255),(76,155,25),(47,213,254)]

            index = knn.predict(mean[0])[0]
            colour = colour_list[index]
            colours.append(colour)
    
    if len(colours) == 4:
        # order colours by contour location 

        #print('min', min_val)
        #print('max',max_val)

        colours = order_faces(centroids, colours)

        #create box in corner for colours
        side_len = 50
        pos = [[0,2],[1,3]]
        for i in range(len(pos)):
            for j in range(len(pos[i])):
                cv.rectangle(output, (i*side_len,j*side_len),((i+1)*side_len,(j+1)*side_len), colours[pos[i][j]],-1)

        # return face in the form of an array
        colour_names =  ['w', 'r', 'b', 'o', 'g', 'y']
        colour_list = [(255,255,255),(20,18,137),(172,72,13),(37,85,255),(76,155,25),(47,213,254)]
        try:
            face = [colour_names[colour_list.index(c)] for c in colours]
            face = [face[:2], face[2:]]
        except:
            print('unknown colour')
    
    return centroids, hsv_vals, output

def getState(cube):
    # get colours of displayed face
    centroids, colours, frame = getColours(cube)
    #print(side)

    #print(centroids)

    #update positions of centroids
    '''objects = ct.update(centroids)
    #print(objects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
'''

    #pass frame back to main for display
    return frame, colours


knn = joblib.load('knn.joblib')
# max and min values for colour updating
max_val = [0,0,0]
min_val = [255,255,255]
# set video flag
video = True
update_colours  = False
# initilise centroid tracker
ct = CentroidTracker()
# colour data file
f = open("/home/grifaj/Documents/y3project/Rubiks_solver/colour_data.txt", "a")

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

            ## per frame operations ##
            output, colours = getState(frame)

            if update_colours:
                label  = 'o'
                for c in colours:
                    out = ''
                    for i in c:
                        out += str(i)+','
                    f.write(out+label+'\n')

            # Display the resulting frame
            showImg('frame',output)

        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    f.close()

    cv.waitKey(0)

else: #photo only
    #cube = cv.imread('C:\\Users\\Alfie\\Documents\\uni_work\\year3\\cs310\\github\Rubiks_solver\\good521.JPG')
    cube = cv.imread('/home/grifaj/Documents/y3project/Rubiks_solver/test1.jpg')

    output = getColours(cube)
    showImg('output', output)

    cv.waitKey(0)