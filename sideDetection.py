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
    colours = [(255,255,255),(20,18,137),(172,72,13),(37,85,255),(76,155,25),(47,213,254)]

    #convert colours to lab
    for a in range(len(colours)):
        colours[a] = cv.cvtColor(np.uint8([[colours[a]]]), cv.COLOR_BGR2LAB)[0][0]
    cubie_value = cv.cvtColor(np.uint8([[cubie_value]]), cv.COLOR_BGR2LAB)[0][0]

    best_dist = -1
    best_colour = ''
    for a in range(len(colours)):
        diff =((cubie_value[0] - colours[a][0])**2 + (cubie_value[1] - colours[a][1])**2 + (cubie_value[2] - colours[a][2])**2)**(1/2)
        if best_dist == -1 or best_dist > diff:
            best_dist = diff
            best_colour = names[a]

    return list(cv.cvtColor(np.uint8([[colours[names.index(best_colour)]]]), cv.COLOR_LAB2BGR)[0][0])

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
    scale = 1
    width = int(img.shape[1] *scale)
    height = int(img.shape[0] *scale)
    dimensions = (width, height)
    img = cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
    cv.imshow(label,img)

def getColours(cube):
    
    #scale image
    scale = 0.5
    width = int(cube.shape[1] *scale)
    height = int(cube.shape[0] *scale)
    dimensions = (width, height)
    cube = cv.resize(cube, dimensions, interpolation=cv.INTER_AREA)


    # set greyscale, blur and find edges
    grey = cv.cvtColor(cube, cv.COLOR_BGR2GRAY)
    blur = cv.blur(cube,(5,5))
    canny = cv.Canny(blur, 80, 120)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    dilated = cv.dilate(canny, kernel)

    showImg('edges',dilated)

    # get contours
    contours, hierarchies = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

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
    colours_lab =[]
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
        if parent == -1 and area > 1000 and squareness < 150:
            
            #print(i,area, squareness, parent)

            # add contour to image
            cv.drawContours(output, contours, i, (0,255,0), 1)

            #get average colour
            mask = np.zeros(grey.shape, np.uint8)
            cv.drawContours(mask, [contour], 0, 255, -1)
            mean = cv.mean(cube, mask=mask) # might not make sense, try HSV

            print(getColour(mean))
            print(getColour_lab(mean))
            colours.append(getColour(mean))
            colours_lab.append(getColour_lab(mean))


    print(colours)
    print(colours_lab)
    if len(colours) == 4:
        #create box in corner for colours
        side_len = 50
        pos = [[0,1],[2,3]]
        for i in range(len(pos)):
            for j in range(len(pos[i])):
                cv.rectangle(output, (i*side_len,j*side_len),((i+1)*side_len,(j+1)*side_len), colours_lab[pos[i][j]],-1)


    
    return output

video = False

if video:
    cap = cv.VideoCapture(2)
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
    #cube = cv.imread('C:\\Users\\Alfie\\Documents\\uni_work\\year3\\cs310\\github\Rubiks_solver\\good521.JPG')
    cube = cv.imread('/home/grifaj/Documents/y3project/Rubiks_solver/test1.jpg')

    output = getColours(cube)
    showImg('output', output)

    cv.waitKey(0)