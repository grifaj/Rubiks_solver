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
    return best_colour

# scale and show image for printing
def showImg(label, img):
    #scale image
    scale = .2
    width = int(img.shape[1] *scale)
    height = int(img.shape[0] *scale)
    dimensions = (width, height)
    img = cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
    cv.imshow(label,img)

cube = cv.imread('cube2.JPG')


# set greyscale, blur and find edges
grey = cv.cvtColor(cube, cv.COLOR_BGR2GRAY)
canny = cv.Canny(grey, 80, 120)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(11,11))
dilated = cv.dilate(canny, kernel)

showImg('edges',dilated)

# get contours
contours, hierarchies = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

blank = np.zeros(cube.shape, dtype='uint8')
cv.drawContours(blank, contours, -1, (0,255,0), 1)
showImg('contours',blank)

hierarchies = hierarchies[0]

print(len(contours), 'contours found')

cubies = []
for i in range(len(contours)):
    contour = contours[i]
    hierarchy = hierarchies[i]
    parent = hierarchy[3]
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    test_area = (perimeter/4)**2
    squareness = test_area/(area+1) if test_area > area else area/(test_area+1) # should be 1 for perfect square

    #check area and squareness is within bounds and that contour has no parent
    if area > 10000 and area < 100000 and squareness < 100 :
        #print(i)
        #print('area',area)
        #print('perim', perimeter)
        #print('squareness',squareness)
        #print('parent', parent)
        #print()
        

       # blank = np.zeros(cube.shape, dtype='uint8')
        #cv.drawContours(blank, contours, i, (0,255,0), 2)
      # showImg(str(i),blank)

        # add selected contours to image
        cv.drawContours(cube,[contour],0,(0, 255, 0),2)

        # get rectangle for colour detection
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(cube, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cubie_colour = getColour(np.array(cv.mean(cube[y:y+h,x:x+w])).astype(int))
        
        #write colour on cubie
        cv.putText(cube, cubie_colour, (x,(y+h//2)), cv.FONT_HERSHEY_TRIPLEX, 5, (0,0,0), 2)
        cubies.append([x, y, cubie_colour])

print(len(cubies), 'cubies found')

showImg('output',cube)

cv.waitKey(0)
