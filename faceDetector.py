import cv2 as cv
import numpy as np
import time
import random
from centroidtracker import CentroidTracker
import joblib
import math as maths

# groups faces based on closness of anlges to each other
def group_cubes(contours, threshold):
    groups = []
    areas = [cv.contourArea(c) for c in contours]
    combined = list(zip(contours, areas))
    combined = sorted(combined, key=lambda x: x[1])
    current_group = [combined[0][1]]
    for i in range(1, len(combined)):
        if abs(combined[i][1] - current_group[-1][1]) <= threshold:
            current_group.append(combined[i])
        else:
            groups.append(current_group)
            current_group = [combined[i]]
    groups.append(current_group)
    return groups

# gets the 4 colours detected on the face and orders them to add to the state
# TODO will need to amend this to partion contours into faces 
def order_faces(contours, colours):

    # split cubes into faces
    groups = group_cubes(contours,150)

    print(len(groups), groups)

    # order faces in each group if it has all 4 faces
    for group in groups:
        if len(group) == 4:

            #get postions of faces
            face_pos = [get_centre(c) for c in contours]

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

# returns centre of the contour
def get_centre(contour):
    M = cv.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return (cX, cY)

'''def drawAxis(img, p_, q_, color, scale):
  p = list(p_)
  q = list(q_)
  pi = maths.pi
 
  ## [visualization1]
  angle = maths.atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
  hypotenuse = maths.sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
 
  # Here we lengthen the arrow by a factor of scale
  q[0] = p[0] - scale * hypotenuse * maths.cos(angle)
  q[1] = p[1] - scale * hypotenuse * maths.sin(angle)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
 
  # create the arrow hooks
  p[0] = q[0] + 9 * maths.cos(angle + pi / 4)
  p[1] = q[1] + 9 * maths.sin(angle + pi / 4)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
 
  p[0] = q[0] + 9 * maths.cos(angle - pi / 4)
  p[1] = q[1] + 9 * maths.sin(angle - pi / 4)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
  ## [visualization1]


def getOrientation(pts, img):
  ## [pca]
  # Construct a buffer used by the pca analysis
  sz = len(pts)
  data_pts = np.empty((sz, 2), dtype=np.float64)
  for i in range(data_pts.shape[0]):
    data_pts[i,0] = pts[i,0,0]
    data_pts[i,1] = pts[i,0,1]
 
  # Perform PCA analysis
  mean = np.empty((0))
  mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
 
  # Store the center of the object
  cntr = (int(mean[0,0]), int(mean[0,1]))
  ## [pca]
 
  ## [visualization]
  # Draw the principal components
  cv.circle(img, cntr, 3, (255, 0, 255), 2)
  p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
  p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
  drawAxis(img, cntr, p1, (255, 255, 0), 1)
  drawAxis(img, cntr, p2, (0, 0, 255), 5)
 
  angle = maths.atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
  ## [visualization]
 
  # Label with the rotation angle
  #label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
  #textbox = cv.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
  #cv.putText(img, label, (cntr[0], cntr[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
 
  return angle'''


# returns angle of contour
def get_angle(contour):
     # Get the approximate polygon around the contour
    epsilon = 0.04 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)
    # Get the centroid of the polygon
    M = cv.moments(approx)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    # Get the orientation angle of the polygon
    angle = np.arctan2(cy - approx[0][0][1], cx - approx[0][0][0])
    angle = np.degrees(angle)
    if angle < 0:
        angle += 180
    return angle



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
    shapes = []
    angles = []
    blank = np.zeros(cube.shape, dtype='uint8')
    # for each contour check if it is a square
    for i in range(len(contours)):
        contour = contours[i]
        hierarchy = hierarchies[i]
        parent = hierarchy[3]

        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        squareness = cv.norm(((perimeter / 4) * (perimeter / 4)) - area)

        epsilon = 0.04 * perimeter
        approx = cv.approxPolyDP(contour, epsilon, True)
        hull = cv.convexHull(contour)

        #if parent == -1 and area > 750 and area < 4000 and squareness < 250:
        if parent == -1 and area > 750 and area < 4000: #and squareness < 400:


           ''' # display contours
            cv.drawContours(blank, contours, i, (random.randint(0,255), random.randint(0, 255), random.randint(0, 255)), 1)
            showImg('contours',blank)
            #print(area, squareness, perimeter)'''

            #cv.drawContours(output, [approx], 0, (0,0,255), 2)

        # likely candidate for piece
        if parent == -1 and area > 750 and area < 4000 and squareness < 210 and len(approx) == 4:
        #if parent == -1 and area > 2000 and area < 8000 and squareness < 120:

            # add contour to image
            cv.drawContours(output, contours, i, (0,255,0), 2)
            # get coords of top left corner for ordering
            x, y, _, _ = cv.boundingRect(contour)
            #cv.putText(img=output, text=(str(x)+' '+str(y)), org=(x, y), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=0.6, color=(225, 0, 255),thickness=1)

            # add contour to list
            shapes.append(contour)

            #cv.putText(img=output, text=str(int(get_angle(contour))), org=(x, y), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=0.6, color=(225, 0, 255),thickness=1)
            angles.append(int(get_angle(contour)))


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
            #std = cv.meanStdDev(lab_img, mask-mask)
            #std = np.mean(std[1])
            hsv_vals.append([int(a) for a in mean])
            mean = np.uint8([[mean]])
            
            #try knn
            colour_names =  ['w', 'r', 'b', 'o', 'g', 'y']
            colour_list = [(255,255,255),(20,18,137),(172,72,13),(37,85,255),(76,155,25),(47,213,254)]

            index = knn.predict(mean[0])[0]
            colour = colour_list[index]
            colours.append(colour)

            #getOrientation(contour, output)
    
    if len(shapes) > 0:
        order_faces(shapes, colours)
    #print(angles)

    # won't work for 4 colours anyomore
    if len(colours) == 4:
        # order colours by contour location 

        #colours = order_faces(shapes, colours)

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