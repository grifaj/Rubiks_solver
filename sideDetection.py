import cv2 as cv
import numpy as np

'''def find_closest_color(input_color):
    color_list = [(255,255,255),(20,18,137),(172,72,13),(37,85,255),(76,155,25),(47,213,254)]
    input_color = np.uint8([[input_color]])
    input_hsv = cv.cvtColor(input_color, cv.COLOR_BGR2HSV)
    closest_color = None
    min_dist = float('inf')
    for color in color_list:
        color = np.uint8([[color]])
        color_hsv = cv.cvtColor(color, cv.COLOR_BGR2HSV)
        dist = np.sum((input_hsv - color_hsv) ** 2)
        if dist < min_dist:
            closest_color = color
            min_dist = dist
    return closest_color[0][0].tolist()'''
        
def find_closest_color(input_colour):
    colour_names =  ['white', 'red', 'blue', 'orange', 'green', 'yellow']
    colour_list = [(255,255,255),(20,18,137),(172,72,13),(37,85,255),(76,155,25),(47,213,254)]
    print(input_colour)
    h, s, v = cv.split(input_colour)
    #white, ignore hue ideal (0,0,255)
    if s < 10 and v > 100:
        print(colour_names[0])
        return colour_list[0]
    # red
    if 0 < h and h < 60:
        print(colour_names[1])
        return colour_list[1]
    # blue
    if 240 < h and h < 300:
        print(colour_names[2])
        return colour_list[2]
    #orange
    if 0 < h and h < 120:
        print(colour_names[3])
        return colour_list[3]
    # green
    if 120 < h and h < 180:
        print(colour_names[4])
        return colour_list[4]
    # yellow
    if 60 < h and h < 120:
        print(colour_names[5])
        return colour_list[5]
    print('not known')

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

    #showImg('edges',dilated)

    # get contours
    contours, hierarchies = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # display contours
    blank = np.zeros(cube.shape, dtype='uint8')
    cv.drawContours(blank, contours, -1, (0,255,0), 1)
    #showImg('contours',blank)

    if hierarchies is None:
        return cube
    else:
        hierarchies = hierarchies[0]

    output = cube
    colours = []
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
            hsv_img = cv.cvtColor(np.uint8([[[255,255,255 ]]]), cv.COLOR_BGR2HSV)
            #hsv_img = cv.cvtColor(cube, cv.COLOR_BGR2HSV)
            #mean = cv.mean(hsv_img, mask=mask)
            mean = np.uint8([[cv.mean(hsv_img)[:-1]]])
            colours.append(find_closest_color(mean))

            '''  mask = np.zeros(grey.shape, np.uint8) 
            cv.drawContours(mask, [contour], 0, 255, -1)

            # get pixels inside contour
            pixels = []
            N = 0
            for i in range(cube.shape[0]):
                for j in range(cube.shape[1]):
                    if mask[i][j] > 0 and (N % 5) == 0:
                        pixels.append(cube[i][j])
                        N+=1
            pixels = np.uint8([pixels])
            colours.append(avg_pixel(pixels))'''

    print(colours)
    if len(colours) == 4:
        #create box in corner for colours
        side_len = 50
        pos = [[0,1],[2,3]]
        for i in range(len(pos)):
            for j in range(len(pos[i])):
                cv.rectangle(output, (i*side_len,j*side_len),((i+1)*side_len,(j+1)*side_len), colours[pos[i][j]],-1)


    
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
    cube = cv.imread('C:\\Users\\Alfie\\Documents\\uni_work\\year3\\cs310\\github\Rubiks_solver\\good521.JPG')
    #cube = cv.imread('/home/grifaj/Documents/y3project/Rubiks_solver/test1.jpg')

    output = getColours(cube)
    showImg('output', output)

    cv.waitKey(0)