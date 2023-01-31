import cv2 as cv
import numpy as np

# assign colour to pixel value of each cubie using closest colour
def getColour(cubie_value):
    names = ['white', 'red', 'blue', 'orange', 'green', 'yellow']
    colours = [[255,255,255],[20,18,137],[172,72,13],[37,85,255],[76,155,25],[47,213,254]]

    best_dist = -1
    best_colour = ''
    for a in range(len(colours)):
        # compute eclidian distance
        sum = 0
        for b in range(len(colours[a])):
            sum += (cubie_value[b] - colours[a][b])**2
        diff = np.sqrt(sum)

        if best_dist == -1 or best_dist > diff:
            best_dist = diff
            best_colour = names[a]

    return colours[names.index(best_colour)]
    #return best_colour

# compare colours in lab colour space using CIE76/CIE94
def getColour_lab(cubie_value):
    names = ['white', 'red', 'blue', 'orange', 'green', 'yellow']
    colours = [[255,255,255],[20,18,137],[172,72,13],[37,85,255],[76,155,25],[47,213,254]]

    #convert colours to lab
    for a in range(len(colours)):
        colours[a] = cv.cvtColor(colours[a], cv.COLOR_BGR2LAB)
    cubie_value = cv.cvtColor(cubie_value, cv.COLOR_BGR2LAB)

    best_dist = -1
    best_colour = ''
    for a in range(len(colours)):
        diff =((cubie_value[0] - colours[a][0])**2 + (cubie_value[1] - colours[a][1])**2 + (cubie_value[2] - colours[a][2])**2)**(1/2)
        if best_dist == -1 or best_dist > diff:
            best_dist = diff
            best_colour = names[a]

    return colours[names.index(best_colour)]

'''   elif method=='CIE94':
        L1, a1, b1 = color1
        L2, a2, b2 = color2
        kl = 1
        kc = 1
        kh = 1
        c1 = (a1**2 + b1**2)**(1/2)
        c2 = (a2**2 + b2**2)**(1/2)
        delta_c = c1 - c2
        delta_a = a1 - a2
        delta_b = b1 - b2
        delta_h = (delta_a**2 + delta_b**2 - delta_c**2)**(1/2)
        delta_l = L1 - L2
        delta_e = ((delta_l/(kl*kl))**2 + (delta_c/(kc*kc))**2 + (delta_h/(kh*kh))**2)**(1/2)
    return delta_e'''


def fractionPoint(ptA, ptB, div):
    return (int(ptA[0]*(1/div) +ptB[0]*(1-(1/div))),int(ptA[1]*(1/div) +ptB[1]*(1-(1/div))))    

def intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = int(det(d, xdiff) / div)
    y = int(det(d, ydiff) / div)
    return x, y
    
# scale and show image for printing
def showImg(label, img):
    #scale image
    scale = 1.2
    width = int(img.shape[1] *scale)
    height = int(img.shape[0] *scale)
    dimensions = (width, height)
    img = cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
    cv.imshow(label,img)

def getColours(cube):
    # set greyscale, blur and find edges
    grey = cv.cvtColor(cube, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(grey, 80, 120)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(21,21))
    dilated = cv.dilate(canny, kernel)

    #showImg('edges',dilated)

    # get contours
    contours, hierarchies = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    blank = np.zeros(cube.shape, dtype='uint8')
    cv.drawContours(blank, contours, -1, (0,255,0), 1)
    #showImg('contours',blank)

    if hierarchies is None:
        return cube
    else:
        hierarchies = hierarchies[0]

    cubies = []
    for i in range(len(contours)):
        contour = contours[i]
        hierarchy = hierarchies[i]
        parent = hierarchy[3]

        area = cv.contourArea(contour)
        perimeter = cv.arcLength(contour, True)
        epsilon = 0.01 * perimeter
        approx = cv.approxPolyDP(contour, epsilon, True)
        squareness = cv.norm((perimeter/4)**2 / area)

        #find the main face
        if parent == -1 and area > 100000 and squareness < 10:

            #blank = np.zeros(cube.shape, dtype='uint8')
            #cv.drawContours(blank, contours, i, (0,255,0), 2)
            #showImg(str(i),blank)

            # draw approx shape
            cv.drawContours(cube, [approx], -1, (0,0,255), 3)


            #split into 9 squares
            square = cv.minAreaRect(approx)
            box = cv.boxPoints(square)
            box = np.int0(box)
            #cv.drawContours(cube,[box],0,(0,255,255),2)

            #find closest hull point to box 
            hull = cv.convexHull(contour)
            for pt in hull:
                cv.circle(cube, pt[0], 2, (255, 0, 0), 2)

            skew_box  = box.copy()
            count = 0
            for vert in box:
                best_dist = -1
                for point in hull:
                    pt = point[0]
                    dist = np.sqrt((pt[0] - vert[0])**2 + (pt[1] - vert[1])**2)
                    if dist < best_dist or best_dist == -1:
                        best_dist = dist
                        best_pt = pt
                skew_box[count] = best_pt
                count +=1
                
            cv.drawContours(cube,[skew_box],0,(0,255,255),2)

            corners = []
            for i in range(4):
                cv.circle(cube, skew_box[i], 10, (255,0,255), 3) # add corners
                (a,b) = fractionPoint(skew_box[i],skew_box[(i+1)%4],2)
                cv.circle(cube, (a,b), 10, (255,0,255), 3) 
                corners.append((a,b))
                cv.putText(cube, str(len(corners)-1), (a,b), cv.FONT_HERSHEY_TRIPLEX, 1, (0,0,0), 2)

            # sixths
            for i in range(4):
                # one side
                (a,b) = fractionPoint(skew_box[i],skew_box[(i+1)%4],6)
                cv.circle(cube, (a,b), 10, (255,0,255), 3) 
                corners.append((a,b))
                cv.putText(cube, str(len(corners)-1), (a,b), cv.FONT_HERSHEY_TRIPLEX, 1, (0,0,0), 2)
                #other side
                (a,b) = fractionPoint(skew_box[(i+1)%4],skew_box[i],6)
                cv.circle(cube, (a,b), 10, (255,0,255), 3) 
                corners.append((a,b))
                cv.putText(cube, str(len(corners)-1), (a,b), cv.FONT_HERSHEY_TRIPLEX, 1, (0,0,0), 2)


            #draw lines between corners
            cv.line(cube,corners[5],corners[8],(255,0,255),3)
            cv.line(cube,corners[0],corners[2],(255,0,255),3)
            cv.line(cube,corners[4],corners[9],(255,0,255),3)
            cv.line(cube,corners[10],corners[7],(255,0,255),3)
            cv.line(cube,corners[3],corners[1],(255,0,255),3)
            cv.line(cube,corners[11],corners[6],(255,0,255),3)

            intersections = [[5,8,10,7],[0,2,10,7],[4,9,10,7],[3,1,5,8],[3,1,0,2],[3,1,4,9],[5,8,11,6],[0,2,11,6],[11,6,4,9]]
            colours = []
            for i in range(9): #for each cubie
                inter = intersection((corners[intersections[i][0]],corners[intersections[i][1]]),(corners[intersections[i][2]],corners[intersections[i][3]]))
                cv.circle(cube, inter, 10, (255,0,255), 3)
                r = 20 #radius
                cv.rectangle(cube, (inter[0]-r,inter[1]-r), (inter[0]+r, inter[1]+r), (0, 255, 255), 2)
                cubie_colour = getColour(np.array(cv.mean(cube[inter[1]-r:inter[1]+r,inter[0]-r:inter[0]+r])).astype(int))
                #cv.putText(cube, cubie_colour, (inter[0]-r,inter[1]-r), cv.FONT_HERSHEY_TRIPLEX,  1, (0,0,0), 2)
                colours.append(cubie_colour)

            #create box in corner for colours
            side_len = 50
            pos = [[0,1,2],[3,4,5],[6,7,8]]
            for i in range(len(pos)):
                for j in range(len(pos[i])):
                    cv.rectangle(cube, (i*side_len,j*side_len),((i+1)*side_len,(j+1)*side_len), colours[pos[i][j]],-1)


    
    return cube

video = False

if video:
    cap = cv.VideoCapture(4)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        output = getColours(frame)
        # Display the resulting frame
        cv.imshow('frame', output)
        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

    cv.waitKey(0)

else: #photo only
    cube = cv.imread('C:\\Users\\Alfie\\Documents\\uni_work\\year3\\cs310\\github\Rubiks_solver\\cube3.JPG')
    #cube = cv.imread('/home/grifaj/Documents/y3project/Rubiks_solver/cube3.jpg')

    output = getColours(cube)
    showImg('output', output)

    cv.waitKey(0)