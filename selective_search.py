import numpy as np
import cv2
import scipy.io as sio
import skimage.segmentation
import skimage.feature

def image_segmentation(img, scale = 1.0, sigma = 0.8, min_size = 50):
    '''
    J.R.R. Uijlings's hierarchical grouping algorithm 
    
    == input ==
    img_8bit : shape = (height, width, 3),
               8-bits degital image (each digit ranges between 0 - 255)
    
    == output ==
    img      : shape = (height, width, 4)
    '''
    # convert the image to range between 0 and 1
    im_mask   = skimage.segmentation.felzenszwalb(
                    img, 
                    scale    = scale, 
                    sigma    = sigma,
                    min_size = min_size)
    #img       = np.dstack([img,im_mask])
    return im_mask

def extractRegion(img):
    R = {}

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            l = img[y][x]

            if l not in R:
                R[l] = {"min_x": np.Inf,  "min_y": np.Inf, "max_x": 0, "max_y": 0, "labels": [l]}

            if R[l]["min_x"] > x:
                R[l]["min_x"] = x
            if R[l]["min_y"] > y:
                R[l]["min_y"] = y
            if R[l]["max_x"] < x:
                R[l]["max_x"] = x
            if R[l]["max_y"] < y:
                R[l]["max_y"] = y
    Rcopy = (R)
    print ((R.keys()))
    for key in R.keys():
        r = R[key]
        if (r["min_x"] == r["max_x"]) or (r["min_y"] == r["max_y"]):
            del Rcopy[key]
    return(Rcopy)

def calc_texture_gradient(img):
    """
        calculate texture gradient for entire image

        The original SelectiveSearch algorithm proposed Gaussian derivative
        for 8 orientations, but we use LBP instead.

        output will be [height(*)][width(*)]
    """
    
    ret = np.zeros(img.shape[:3])
    for colour_channel in (0, 1, 2):
        ret[:, :, colour_channel] = skimage.feature.local_binary_pattern(
            img[:, :, colour_channel], 8, 1.0)

    return ret
    

def calc_hist(img, minhist=0, maxhist=1):
    """
        calculate colour histogram for each region

        the size of output histogram will be BINS * COLOUR_CHANNELS(3)

        number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]
        
        len(hist) = BINS * 3
        hist[:BINS] = [0, 10, 20, 0,...,0] meaning that 
           there are 10 pixels that have values between (maxhist - minhist)/BINS*1 and (maxhist - minhist)/BINS*2
           there are 10 pixels that have values between (maxhist - minhist)/BINS*2 and (maxhist - minhist)/BINS*3
    
    """

    BINS = 25
    hist = np.array([])

    for colour_channel in range(3):

        # extracting one colour channel
        c = img[:, colour_channel]

        # calculate histogram for each colour and join to the result
        hist = np.concatenate([hist] + [np.histogram(c, BINS, (minhist, maxhist))[0]])
        #print (np.histogram(c, BINS, (minhist, maxhist)))

    # L1 normalize
    hist = hist / len(img)
    return hist

def extract_neighbours(regions):
    '''
    check if two regions intersect 
    '''

    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"] and a["min_y"] < b["min_y"] < a["max_y"]) or\
           (a["min_x"] < b["max_x"] < a["max_x"] and a["min_y"] < b["max_y"] < a["max_y"]) or\
           (a["min_x"] < b["min_x"] < a["max_x"] and a["min_y"] < b["max_y"] < a["max_y"]) or\
           (a["min_x"] < b["max_x"] < a["max_x"] and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    R = np.array(regions.items())
    neighbours = []
    for cur, a in enumerate(R[:-1]):
        #print (a)
        for b in R[cur + 1:]:
            if intersect(a[1], b[1]):
                neighbours.append((a, b))

    return neighbours


def _sim_colour(r1, r2):
    """
        calculate the sum of histogram intersection of colour
    """
    return sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])


def _sim_texture(r1, r2):
    """
        calculate the sum of histogram intersection of texture
    """
    return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])


def _sim_size(r1, r2, imsize):
    """
        calculate the size similarity over the image
    """
    return 1.0 - (r1["size"] + r2["size"]) / imsize


def _sim_fill(r1, r2, imsize):
    """
        calculate the fill similarity over the image
    """
    bbsize = (
        (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
        * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize

def calc_sim(r1, r2, imsize):
    return (_sim_colour(r1, r2)       +\
            _sim_texture(r1, r2)      +\
            _sim_size(r1, r2, imsize) +\
            _sim_fill(r1, r2, imsize))

def calculate_similarlity(img,neighbours,verbose=False):
    # calculate initial similarities
    imsize = img.shape[0] * img.shape[1]
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = calc_sim(ar, br, imsize)
        if verbose:
            print("S[({:2.0f}, {:2.0f})]={:3.2f}".format(ai,bi,S[(ai, bi)]))
    return(S)


def merge_regions(r1, r2):
    '''
    Suppose r1 and r2 look like: 
    
    (minx1, miny1)
          _____________________
          |                   |
          |     (minx2,maxy2) |
          |           ________|____ 
          |          |        |    |
          |          |        |    |
          |          |________|____|(maxx2, maxy2)
          |___________________|(maxx1, maxy1)
          
    Then for merged region,
    
    minx : minx1
    miny : miny1
    maxx : maxx2
    maxy : maxy1 
    size : the number of pixels in the two regions (intersection is counted twice?)
    
                         
    '''
    new_size = r1["size"] + r2["size"]
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_c": (r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
        "hist_t": (r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }
    return rt



def merge_regions_in_order(S,R,imsize, verbose=False):
    '''
    == Input ==
    S : similarity dictionary
    
    R : dictionary of proposed regions
    
    == Output ==
    
    regions : list of regions
    
    '''
    # hierarchal search
    while S != {}:

        # Step 1: get highest similarity pair of regions from the similarlity dictionary
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # Step 2: marge the region pair and add to the region dictionary
        t = max(R.keys()) + 1.0
        R[t] = merge_regions(R[i], R[j])

        # Step 3: from the similarity dictionary, 
        #         remove all the pair of regions where one of the regions is selected in Step 1
        key_to_delete = []
        for k, v in list(S.items()):
            if (i in k) or (j in k):
                key_to_delete.append(k)
        for k in key_to_delete:
            del S[k]

        # Step 4: calculate similarity with new merged region and the regions and its intersecting region
        #         (intersecting region is the region that are to be deleted)
        for k in key_to_delete:
            if k != (i,j):
                if k[0] in (i, j):
                    n = k[1]
                else:
                    n = k[0]
                S[(t, n)] = calc_sim(R[t], R[n], imsize)
    if verbose:
        print("{} regions".format(len(R)))

    ## finally return list of region proposal
    regions = []
    for k, r in list(R.items()):
            regions.append({
                'rect': (
                    r['min_x'],              # min x
                    r['min_y'],              # min y
                    r['max_x'] - r['min_x'], # width 
                    r['max_y'] - r['min_y']),# height
                'size': r['size'],
                'labels': r['labels']
            })
    return (regions)


scale    = 1.0
sigma    = 0.8
# min_size may be around 50 for better RCNN performance but for the sake of visualization, I will stick to min_size =500
min_size = 500 # 500 3000

# import 8 bits degital image (each digit ranges between 0 - 255)
#img_8bit  = scipy.misc.imread(os.path.join(img_dir,imgnm))
    
image = cv2.imread("dog.jpg")
img  = image_segmentation(image.astype(float), scale, sigma, min_size)
img = img.astype(np.uint8)

R = extractRegion(img)

for item in R.values():
    x1 = item["min_x"]
    y1 = item["min_y"]
    x2 = item["max_x"]
    y2 = item["max_y"]
    cv2.rectangle(image,(x1,y1), (x2,y2), 255,1)
#cv2.imshow("image",image)
#cv2.waitKey(0)
print("{} rectangle regions are found".format(len(R)))
tex_grad = calc_texture_gradient(image)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
calc_hist(image)
for k, v in list(R.items()):
    ## height and width axies are flattened. 
    ## masked_pixel.shape = (N pixel with this mask ID , 3)
    masked_pixels  = hsv[img == k]
    R[k]["size"]   = len(masked_pixels / 4)
    R[k]["hist_c"] = calc_hist(masked_pixels,minhist=0, maxhist=1)
    R[k]["hist_t"] = calc_hist(tex_grad[img == k],minhist=0, maxhist=2**8-1)
neighbours = extract_neighbours(R)
S = calculate_similarlity(img,neighbours,verbose=True)
i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]
print (i,j)
regions = merge_regions_in_order(S,R,img.shape[0]*img.shape[1],verbose=True)
for item in (regions):
    x1, y1, width, height = item["rect"]
    #label = item["labels"][:5]
    x2 = x1 + width
    y2 = y1 + height
    cv2.rectangle(image,(x1,y1), (x2,y2), 255,1)
cv2.imshow("image",image)
cv2.waitKey(0)


'''
brightLAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
Lchannel = brightLAB[:,:,0]   # Lightness component
achannel = brightLAB[:,:,1]	  # Color component
bchannel = brightLAB[:,:,2]   # Color Component

Hchannel = hsvImage[:,:,0]   # Lightness component
Schannel = hsvImage[:,:,1]	  # Color component
Vchannel = hsvImage[:,:,2]   # Color Component

cv2.imshow("image",Hchannel)
cv2.waitKey(0)
cv2.imshow("image1", Schannel)
cv2.waitKey(0)
cv2.imshow("image2", Vchannel)
cv2.waitKey(0)

'''
