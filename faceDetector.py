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

'''# gets the 4 colours detected on the face and orders them to add to the state
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

    return colours'''

# assume 4 cubies
def order_faces(contours, colours):

    #get postions of faces
    centres = [get_centre(c) for c in contours]

    combined = zip(centres, colours)
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
    [centres, colours] = map(list, zip(*new_order))

    return centres, colours



# returns centre of the contour
def get_centre(contour):
    M = cv.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return (cX, cY)

# scale and show image for printing
def showImg(label, img):
    #scale image
    scale = 1
    width = int(img.shape[1] *scale)
    height = int(img.shape[0] *scale)
    dimensions = (width, height)
    img = cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
    cv.imshow(label,img)

def inferCubie(centres):
    # find greatest distance between 2 cubes
    return



# returns data on the shown face
def getFace(cube, verbose=True, update_colours=False):
    
    grey = cv.cvtColor(cube, cv.COLOR_BGR2GRAY)
    blur = cv.blur(cube,(3,3))
    canny = cv.Canny(blur, 55, 100, L2gradient = True) #60, 100

    if verbose: showImg('edges',canny)

    # get contours
    contours, hierarchies = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    if hierarchies is None:
        return cube
    else:
        hierarchies = hierarchies[0]

    output = cube
    colours = []
    centres = []
    hsv_vals =[]
    shapes = [] # list of contours
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

        # likely candidate for piece
        if parent == -1 and area > 750 and area < 4000 and squareness < 210 and len(approx) == 4:
        #if parent == -1 and area > 2000 and area < 8000 and squareness < 120:

            # add contour to image
            cv.drawContours(output, contours, i, (0,255,0), 2)
            # get coords of top left corner for ordering
            x, y, _, _ = cv.boundingRect(contour)
            #cv.putText(img=output, text=(str(x)+' '+str(y)), org=(x, y), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=0.6, color=(225, 0, 255),thickness=1)

            shapes.append(contour) # add contour to list
            centres.append(get_centre(contour)) # add square centre

            #get average colour
            mask = np.zeros(grey.shape, np.uint8) 
            cv.drawContours(mask, [contour], 0, 255, -1)
            lab_img = cv.cvtColor(cube, cv.COLOR_BGR2LAB)
            mean = cv.mean(lab_img, mask=mask)[:-1]
            hsv_vals.append([int(a) for a in mean])
            mean = np.uint8([[mean]])
            
            #try knn for classifiing colours
            colour_list = [(255,255,255),(20,18,137),(172,72,13),(37,85,255),(76,155,25),(47,213,254)]
            index = knn.predict(mean[0])[0]
            colour = colour_list[index]
            colours.append(colour)

    # outputs objects to use, might create cubie class
        # hsv_vals: colour of face
        # shapes: contours of face
        # centres: centre point of each face

    # should be able to approximate final cubie
    if len(colours) == 3:
        hidden_centre = inferCubie(centres)

        cv.circle(output,hidden_centre,5,(0,0,0), 5)


    # won't work for 4 colours anyomore
    if len(colours) == 4:
        # order colours by contour location 
        centres, colours = order_faces(shapes, colours)

        #create box in corner for colours
        side_len = 50
        pos = [[0,2],[1,3]]
        for i in range(len(pos)):
            for j in range(len(pos[i])):
                cv.rectangle(output, (i*side_len,j*side_len),((i+1)*side_len,(j+1)*side_len), colours[pos[i][j]],-1)

        # return face in the form of an array
        colour_names =  ['w', 'r', 'b', 'o', 'g', 'y']
        try:
            face = [colour_names[colour_list.index(c)] for c in colours]
            face = [face[:2], face[2:]]
        except:
            print('unknown colour')
        
        if update_colours:
            return hsv_vals
    
        return centres, face
    
    #return centroids, hsv_vals, output

def getState(cube):
    # get colours of displayed face
    centroids, colours, frame = getFace(cube)
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

# file setups
knn = joblib.load('knn.joblib')


if __name__ == '__main__':

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
                #output, colours = getState(frame) bypass get state for now
                colours = getFace(frame, update_colours=update_colours)

                if update_colours:
                    if colours is not None:
                        label  = 'w'
                        for c in colours:
                            out = ''
                            for i in c:
                                out += str(i)+','
                            f.write(out+label+'\n')

                # Display the resulting frame
                showImg('frame',frame)

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

        output = getFace(cube)
        showImg('output', output)

        cv.waitKey(0)