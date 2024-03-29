import cv2 as cv
import numpy as np
import time
from cubie import Cubie
import globals
from cube import RubiksCube
from orientFaces import orient
import random

# groups faces based on closness of anlges to each other
def group_cubes(cubies, threshold):
    if len(cubies) > 0:
        areas = [cv.contourArea(c.contour) for c in cubies]

        groups = []
        combined = list(zip(cubies, areas))
        combined = sorted(combined, key=lambda x: x[1])
        current_group = [combined[0]]
        for i in range(1, len(combined)):
            if abs(combined[i][1] - current_group[-1][1]) <= threshold:
                current_group.append(combined[i])
            else:
                groups.append(current_group)
                current_group = [combined[i]]
        groups.append(current_group)
        return groups

# assume 4 cubies
def order_faces(cubies):

    #get postions of faces
    centres = [c.centre for c in cubies]
    colours = [c.colour for c in cubies]

    combined = zip(centres, colours)
    combined = list(combined)

    #order by y, gives top and bottom row
    combined = sorted(combined, key=lambda x: x[0][1])
    top = combined[:3]
    middle = combined[3:6]
    bottom = combined[6:]

    #order top and bottom by x value to give left and right
    top = sorted(top, key=lambda x: x[0][0])
    middle = sorted(middle, key=lambda x: x[0][0])
    bottom = sorted(bottom, key=lambda x: x[0][0])

    #combine to give full sorted list
    new_order = top + middle + bottom

    # seperate colours out in new order
    [centres, colours] = map(list, zip(*new_order))

    return centres, colours

# returns data on the shown face
def getFace(frame, verbose=False, update_colours=False):
    
    cv.imshow('orgin', frame)
    # blur and get edges from frame
    blur = cv.blur(frame,(3,3))
    canny = cv.Canny(blur, 50, 100, L2gradient = True) #60, 100
    
    if verbose: cv.imshow('edges',canny)

    # get contours
    contours, hierarchies = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    if hierarchies is None:
        return
    else:
        hierarchies = hierarchies[0]

    cubies  = []
    blank = np.zeros(frame.shape, dtype='uint8')
    # select candidates for cubies
    for i in range(len(contours)):
        contour = contours[i]
        hierarchy = hierarchies[i]
        parent = hierarchy[3]

        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        squareness = cv.norm(((perimeter / 4) * (perimeter / 4)) - area)

        epsilon = 0.04 * perimeter
        approx = cv.approxPolyDP(contour, epsilon, True)

        # likely candidate for piece
        if area > 250 and area < 5000 and squareness < 250:
            # create new cubie object
            cubies.append(Cubie(frame=frame, contour=contour))
        

        if verbose:
            colour = random.sample(range(0, 255), 3)
            cv.drawContours(blank, contours=contours, contourIdx=i, color=colour, thickness=2)
            cv.imshow('Contours', blank)

    # group cubies by area to get main face
    avg = cv.mean(np.array([cv.contourArea(c.contour) for c in cubies]))[0]
    groups = group_cubes(cubies, avg*0.25)
    
    # display groups as different colours
    if groups is not None:
        group_colours = [(0,255,0),(255,0,255),(255,255,0),(0,255,255)]
        groups = sorted(groups, key=lambda x : len(x), reverse=True)
        #display current groups
        for i in range(len(groups)):
            face = [c[0].contour for c in groups[i]]
            cv.drawContours(frame, face, -1, group_colours[i], 2)

        # select main face
        cubies = [c[0] for c in groups[0]]

    if len(cubies) == 9:
        # order colours by contour location 
        centres, colours = order_faces(cubies)

        #create box in corner for colours
        side_len = 50
        pos = [[0,1,2],[3,4,5],[6,7,8]]
        for i in range(len(pos)):
            for j in range(len(pos[i])):
                cv.rectangle(frame, (i*side_len,j*side_len),((i+1)*side_len,(j+1)*side_len), colours[pos[j][i]],-1)

        # if updating colours 
        if update_colours:
            return [c.hsvVal for c in cubies]

        # return face in the form of an array
        colour_dict = {(255,255,255):'w', (20,18,137):'r', (172,72,13):'b', (37,85,255):'o', (76,155,25):'g', (47,213,254):'y'}
        face = [colour_dict[c] for c in colours]
        face = [face[:3], face[3:6],face[6:]]

        return [centres, face]

def placeCross(frame, centres):
    cv.line(frame, centres[0], centres[8], (255,0,255),6)
    cv.line(frame, centres[2], centres[6], (255,0,255),6)


def getState(frame):
    # state complete
    if globals.faceNum == 6:
        print('done')
        # need to rotate faces to correct orientation
        tmp = RubiksCube(array=orient(globals.detectedCube))
        return tmp.stringify()

    # get colours of displayed face
    result = getFace(frame, verbose=False)
    if result is not None:
        [centres, face] = result
    else:
        return
    
    centre_face = face[1][1]
    colours = ["w", "y", "r", "o", "b", "g"] 

    # if face is new face add to state
    if (globals.colours == [] or centre_face not  in globals.colours) and globals.consistentCount > 10:
        globals.colours.append(centre_face)
        globals.detectedCube.array[colours.index(centre_face)] = face
        globals.detectedCube.printCube()
        globals.faceNum +=1
        globals.rotateFlag = True
        print('new face scanned',centre_face)
        globals.consistentCount = 0
    
    # check if face matches last seen face
    elif face == globals.previousFace and (globals.colours == [] or centre_face not  in globals.colours):
        globals.consistentCount +=1
    else:
        globals.consistentCount = 0

    # place cross on cube
    if centre_face in globals.colours:
        placeCross(frame, centres)

    # set current face to previous
    globals.previousFace = face

        
if __name__ == '__main__':

    update_colours  = False
    array = []
    faceNum = 0
    consistentCount = 0
    previousFace = None

    # colour data file
    f = open("/home/grifaj/Documents/y3project/Rubiks_solver/colour_data.txt", "a")

    frame_rate = 20 # set frame rate
    prev = 0
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    index = 0
    prev_colour_time = 0
    while True:
        time_elapsed = time.time() - prev
        ret, frame = cap.read()

        colour_time_elapsed = time.time() - prev_colour_time

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if time_elapsed > 1./frame_rate:
            prev = time.time()

            ## per frame operations ##
            colours = getFace(frame, update_colours=update_colours)

            colours_labels = ['w','r','b','o','g','y']
            scan_time = 10
            if update_colours:
                if colour_time_elapsed > scan_time:
                    print('changing to label', colours_labels[index])
                    prev_colour_time = time.time()
                    label = colours_labels[index]
                    index+=1

                text = str(int(scan_time - colour_time_elapsed))
                cv.putText(img=frame, text=text, org=(150, 80), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 255, 0),thickness=3)
                cv.putText(img=frame, text=colours_labels[index-1], org=(50, 80), fontFace=cv.FONT_HERSHEY_TRIPLEX, fontScale=3, color=(0, 255, 0),thickness=3)
                if colours is not None:
                    for c in colours:
                        out = ''
                        for i in c:
                            out += str(i)+','
                        f.write(out+label+'\n')

                if index > 5:
                    print('ending stream')
                    break

            # Display the resulting frame
            cv.imshow('frame',frame)

        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    f.close()

    cv.waitKey(0)
