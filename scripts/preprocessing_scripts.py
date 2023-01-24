import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np
import cv2
import pydicom
import os
import skimage
from pathlib             import Path
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_windowing, apply_voi_lut, apply_voi
import numpy.ma as ma
from scipy.stats.mstats import mquantiles
from skimage.transform import rescale
from skimage.filters import threshold_otsu
from skimage import exposure
from PIL import Image



def img_original_raw_or_not(fn_or_dicom):
    # Corrects the encoding (MONOCHROME) and applies the rescale slope and intercept if tags are present
    # Using new functions from pydicom library
    # Separate processing if image is 'FOR PRESENTATION' or 'FOR PRESENTATION'
    # Remark: The functions `apply_modality_lut`, `apply_voi` and `apply_windowing` 
    # return the numpy array unchanged on raw DICOMS 
    # because the tags required by these functions are missing.

    if isinstance(fn_or_dicom, str):
        dicom = pydicom.read_file(fn_or_dicom)
    elif isinstance(fn_or_dicom, pydicom.dataset.FileDataset):
        dicom = fn_or_dicom
    else:
        raise Exception(
            'fn_or_dicom is not a file name or dicom object. Type(fn_or_dicom)={}'.format(type(fn_or_dicom)))

    patient = dicom.PatientID
    img = apply_modality_lut(dicom.pixel_array, dicom).astype(int)
    
    if dicom.PresentationIntentType == 'FOR PRESENTATION':

        inverse_pls = False
        if 'PresentationLUTShape' in dicom:
            if dicom.PresentationLUTShape == 'INVERSE':
                img = apply_windowing(img, dicom).astype(int)
                inverse_pls = True

        if inverse_pls:
            
            img = apply_voi(img, dicom)  ## no lut
        else:
            img = apply_voi_lut(img, dicom)

        if 'PhotometricInterpretation' in dicom:
            if dicom.PhotometricInterpretation == 'MONOCHROME1':
                img = img.max() - img
                
        
    elif dicom.PresentationIntentType == 'FOR PROCESSING':

        inverse_pls = False
        if 'PresentationLUTShape' in dicom:
            if dicom.PresentationLUTShape == 'INVERSE':
                # (0028,1050) Window Center and (0028,1051) Window Width not present
                # so leaves the array unchanged
                img = apply_windowing(img, dicom).astype(int)
                inverse_pls = True
                
        # VOI LUT attribute required only if Presentation Intent Type (0008,0068) is FOR PRESENTATION
        # ...so here leaves the array unchanged
        if inverse_pls:
            img = apply_voi(img, dicom)  ## no lut
        else:
            img = apply_voi_lut(img, dicom)
        
        # this is the only bit that does something in 'FOR PROCESSING'
        if 'PhotometricInterpretation' in dicom:
            if dicom.PhotometricInterpretation == 'MONOCHROME1':
                img = img.max() - img
        
    # Some images have 3 channels (error images):
    if img.ndim > 2:
        print('WARNING: Image with more than 1 channel. Applying rgb2gray transformation')
        # img = imgu.rgb2gray(img)



    # print("This image is " + str(dicom.PresentationIntentType) + "; Presentation LUT shape " + str(dicom.PresentationLUTShape) + "; and Photometric Interpretation " + str(dicom.PhotometricInterpretation))
    return [img.astype(int), patient]



def img_norm_raw(fn_or_dicom, ymin=0, ymax=255):
    # For presentation: normalizes the image to the range [ymin,ymax])
    # For processing: removes background, windows, normalizes the image to the range [ymin,ymax]
    # Returns: a list (numpy array, string)
    # where the string is the filename to save it


    if isinstance(fn_or_dicom, str):
        dicom = pydicom.read_file(fn_or_dicom)
    elif isinstance(fn_or_dicom, pydicom.dataset.FileDataset):
        dicom = fn_or_dicom
    else:
        raise Exception(
            'fn_or_dicom is not a file name or dicom object. Type(fn_or_dicom)={}'.format(type(fn_or_dicom)))
    
    img = img_original_raw_or_not(dicom)
        
    # Checking if the image is 'FOR PROCESSING' or 'FOR PRESENTATION'
    if (dicom.PresentationIntentType=='FOR PRESENTATION'):

        imgNorm = cv2.normalize(img, None, ymin, ymax, cv2.NORM_MINMAX) 
  
        return imgNorm.astype('float32')


    
    elif (dicom.PresentationIntentType=='FOR PROCESSING'):
        
        
        img255 = (255*(img/float(np.max(img)))).astype('uint8')  


        #removing the background - thresholding
        ret,th = cv2.threshold(img255,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        binary = 1*(img255<ret)
        imgm = ma.masked_array(img, mask=binary) #apply mask
        #img1 = ma.masked_array(np.maximum(np.ones_like(imgm), imgm)), mask=binary) #log transform

        ## windowing quantiles
        myquants=mquantiles(imgm.compressed(), [0,0.1,0.25,0.5,0.75,0.85,1]).tolist()
    
        ## windowing
        mymin=np.ones_like(imgm) * (myquants[2]) ##don't window white
        #myzero=np.zeros_like(imgm)
        
        img2 = ma.masked_array(np.maximum(mymin, img)- myquants[2] , mask=binary) ##don't window white
        
        ## fill the white 
        imgb = img2.filled(np.min(img))
        imgb = imgb.astype('float32')
        
        ### Normalisation 
        imgNorm = cv2.normalize(imgb, None, ymin, ymax, cv2.NORM_MINMAX) 

        ### Name that we will use to save it
        filename = dicom.PatientID + dicom.ViewPosition + dicom.ImageLaterality + dicom.AcquisitionDate

        return [imgNorm.astype('float32'), filename]
    
    else:
        raise Exception(
        'This dicom is neither FOR PRESENTATION nor FOR PROCESSING')
















 




def CropBorders(img):
    
    '''
    Cuts
    1% of the image’s width from the left and right edges 
    4% of the image’s height from the top and bottom edges
    
    Parameters
    ----------
    img : {numpy.ndarray}
        The image to crop.
        
    Returns
    -------
    cropped_img: {numpy.ndarray}
        The cropped image.
    '''
    nrows, ncols = img.shape

    # Get the start and end rows and columns
    l_crop = int(ncols * 0.01)
    r_crop = int(ncols * (1 - 0.01))
    u_crop = int(nrows * 0.04)
    d_crop = int(nrows * (1 - 0.04))

    cropped_img = img[u_crop:d_crop, l_crop:r_crop]
    
    return cropped_img


def OwnGlobalBinarise(img, thresh, maxval):
    
    '''
    This function takes in a numpy array image and
    returns a corresponding mask that is a global
    binarisation on it based on a given threshold
    and maxval. Any elements in the array that is
    greater than or equals to the given threshold
    will be assigned maxval, else zero.
    
    Parameters
    ----------
    img : {numpy.ndarray}
        The image to perform binarisation on.
    thresh : {int or float}
        The global threshold for binarisation.
    maxval : {np.uint8}
        The value assigned to an element that is greater
        than or equals to `thresh`.
        
        
    Returns
    -------
    binarised_img : {numpy.ndarray, dtype=np.uint8}
        A binarised image of {0, 1}.
    '''
    
    binarised_img = np.zeros(img.shape, np.uint8)
    binarised_img[img >= thresh] = maxval
    
    return binarised_img

def OpenMask(mask, ksize=(23, 23), operation="open"):

    '''
    This function edits a given mask (binary image) by performing
    closing then opening morphological operations.
    
    Parameters
    ----------
    mask : {numpy.ndarray}
        The mask to edit.
        
    Returns
    -------
    edited_mask : {numpy.ndarray}
        The mask after performing close and open morphological
        operations.
    '''
        
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=ksize)
    
    if operation == "open":
        edited_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == "close":
        edited_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Then dilate
    edited_mask = cv2.morphologyEx(edited_mask, cv2.MORPH_DILATE, kernel)    
    # Then erode the edge of the breast

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (150,150))
    edited_mask = cv2.erode(edited_mask,kernel2,iterations = 2)
    edited_mask = cv2.GaussianBlur(edited_mask, (5, 5), cv2.BORDER_DEFAULT)
    return edited_mask

def SortContoursByArea(contours, reverse=True):
    
    '''
    This function takes in list of contours, sorts them based
    on contour area, computes the bounding rectangle for each
    contour, and outputs the sorted contours and their
    corresponding bounding rectangles.
    
    Parameters
    ----------
    contours : {list}
        The list of contours to sort.
        
    Returns
    -------
    sorted_contours : {list}
        The list of contours sorted by contour area in descending
        order.
    bounding_boxes : {list}
        The list of bounding boxes ordered corresponding to the
        contours in `sorted_contours`.
    '''
    
    # Sort contours based on contour area.
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Construct the list of corresponding bounding boxes.
    bounding_boxes = [cv2.boundingRect(c) for c in sorted_contours]
    
    return sorted_contours, bounding_boxes

def XLargestBlobs(mask, top_X=None):
    
    '''
    This function finds contours in the given image and
    keeps only the top X largest ones.
    
    Parameters
    ----------
    mask : {numpy.ndarray, dtype=np.uint8}
        The mask to get the top X largest blobs.
    top_X : {int}
        The top X contours to keep based on contour area
        ranked in decesnding order.
        
        
    Returns
    -------
    n_contours : {int}
        The number of contours found in the given `mask`.
    X_largest_blobs : {numpy.ndarray}
        The corresponding mask of the image containing only
        the top X largest contours in white.
    '''
        
    # Find all contours from binarised image.
    # Note: parts of the image that you want to get should be white.
    contours, hierarchy = cv2.findContours(image=mask,
                                           mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_NONE)
    
    n_contours = len(contours)
    
    # Only get largest blob if there is at least 1 contour.
    if n_contours > 0:
        
        # Make sure that the number of contours to keep is at most equal 
        # to the number of contours present in the mask.
        if n_contours < top_X or top_X == None:
            top_X = n_contours
        
        # Sort contours based on contour area.
        sorted_contours, bounding_boxes = SortContoursByArea(contours=contours,
                                                             reverse=True)
        
        # Get the top X largest contours.
        X_largest_contours = sorted_contours[0:top_X]
        
        # Create black canvas to draw contours on.
        to_draw_on = np.zeros(mask.shape, np.uint8)
        
        # Draw contours in X_largest_contours.
        X_largest_blobs = cv2.drawContours(image=to_draw_on, # Draw the contours on `to_draw_on`.
                                           contours=X_largest_contours, # List of contours to draw.
                                           contourIdx=-1, # Draw all contours in `contours`.
                                           color=1, # Draw the contours in white.
                                           thickness=-1) # Thickness of the contour lines.
        
    return n_contours, X_largest_blobs


def InvertMask(mask):
    
    '''
    This function inverts a given mask (i.e. 0 -> 1
    and 1 -> 0).
    
    Parameters
    ----------
    mask : {numpy.ndarray, dtype=np.uint8}
        The mask to invert.
        
    Returns
    -------
    inverted_mask: {numpy.ndarray}
        The inverted mask.
    '''
    
    inverted_mask = np.zeros(mask.shape, np.uint8)
    inverted_mask[mask == 0] = 1
    

    return inverted_mask



def InPaint(img, mask, flags="ns", inpaintRadius=30):

    '''
    This function restores an input image in areas indicated
    by the given mask (elements with 1 are restored). 
    
    Parameters
    ----------
    img : {numpy.ndarray}
        The image to restore.
    mask : {numpy.ndarray, dtype=np.uint8}
        The mask that indicates where (elements == 1) in the
        `img` the damage is.
    inpaintRadius : {int}
        Radius of a circular neighborhood of each point
        inpainted that is considered by the algorithm.
        
    Returns
    -------
    inpainted_img: {numpy.ndarray}
        The restored image.
    '''
    
    # First convert to `img` from float64 to uint8.
    # img = 255 * img
    # img = img.astype(np.uint8)
    # img = (255*(img/float(np.max(img)))).astype('uint8')
    #print(img.shape)
    #print(mask.shape)
    
    # Then inpaint based on flags.
    if flags == "telea":
        #print('teleando')
        inpainted_img = cv2.inpaint(src=img,
                                    inpaintMask=mask,
                                    inpaintRadius=inpaintRadius,
                                    flags=cv2.INPAINT_TELEA)
    elif flags == "ns":
        inpainted_img = cv2.inpaint(src=img,
                                    inpaintMask=mask,
                                    inpaintRadius=inpaintRadius,
                                    flags=cv2.INPAINT_NS)
    
    return inpainted_img





def ApplyMask(img, mask):
    
    '''
    This function applies a mask to a given image. White
    areas of the mask are kept, while black areas are
    removed.
    
    Parameters
    ----------
    img : {numpy.ndarray}
        The image to mask.
    mask : {numpy.ndarray, dtype=np.uint8}
        The mask to apply.
        
    Returns
    -------
    masked_img: {numpy.ndarray}
        The masked image.
    '''
    
    masked_img = img.copy()
    masked_img[mask == 0] = 0
    
    return masked_img

def img_resize(img, shapeImgOut=(1024, 787), resizeMethod='span', resampleMethod='NEAREST', position='left'):
    # It resizes the input image to the size sizeImgOut with:
    # resizeMethod in {padding, span}
    # The padding option resizes the longest side of the pixel array and rescale the image 
    # maintaining the original ratio. 
    # Then creates a black square and pastes the rescaled image on it, 
    # on the left handside of the square (commented code to change this behaviour).
    # resampleMethod should be a method accepted by Image.resize:
    # (0) PIL.Image.NEAREST
    # (4) PIL.Image.BOX
    # (2) PIL.Image.BILINEAR
    # (5) PIL.Image.HAMMING
    # (3) PIL.Image.BICUBIC
    # (1) PIL.Image.LANCZOS
    # position can be 'centre' (padding on both sides) or 'left'(padding on right side)
    # returns (numpy array)


    if isinstance(shapeImgOut, int):
        # Per output quadrato:
        # shapeImgOut = (shapeImgOut, shapeImgOut)

        # the following keeps ratio if we set by default, 
        # shapeImgOut=1024 in the def.
        # But we have different ratios in optimam
        # we end up with images with same height and different width.
        # shapeImgOut = (shapeImgOut,ma.floor((img.shape[1]*shapeImgOut)/img.shape[0]))
        shapeImgOut = (shapeImgOut[0], shapeImgOut[1])

    sizeImgOut = shapeImgOut[::-1]

    if resizeMethod == 'padding':
        divMax = max(img.shape[0] / shapeImgOut[0], img.shape[1] / shapeImgOut[1])
        print(divMax)
        sizeResize = (int(img.shape[1] / divMax), int(img.shape[0] / divMax))
    else:
        sizeResize = sizeImgOut


    typeImgIn = img.dtype
    img = Image.fromarray(img.astype(np.uint8))
    img = img.resize(sizeResize, resample=getattr(Image, resampleMethod))

    if resizeMethod == 'padding':
        # PIL.Image.new(mode, size, color) creates a new image with the given mode and size  
        # L is (8-bit pixels, black and white)
        # Size is given as a (width, height)-tuple
        # default colour is 0
        out_img = Image.new('L', sizeImgOut)   
        # Image.paste(im, box=None, mask=None)
        
        if position == 'centre':
            out_img.paste(img, ((sizeImgOut[0] - img.size[0]) // 2, (sizeImgOut[1] - img.size[1]) // 2))
        else:
            out_img.paste(img, (0, 0))
    else:
        out_img = img

    return np.array(out_img).astype(typeImgIn)





def HorizontalFlip(mask):
    
    '''
    This function figures out how to flip (also entails whether
    or not to flip) a given mammogram and its mask. The correct
    orientation is the breast being on the left (i.e. facing
    right) and it being the right side up. i.e. When the
    mammogram is oriented correctly, the breast is expected to
    be found in the bottom left quadrant of the frame.
    
    Parameters
    ----------
    mask : {numpy.ndarray, dtype=np.uint8}
        The corresponding mask of the CC image to flip.

    Returns
    -------
    horizontal_flip : {boolean}
        True means need to flip horizontally,
        False means otherwise.
    '''
    
    # Get number of rows and columns in the image.
    nrows, ncols = mask.shape
    x_center = ncols // 2
    y_center = nrows // 2
    
    # Sum down each column.
    col_sum = mask.sum(axis=0)
    # Sum across each row.
    row_sum = mask.sum(axis=1)
    
    left_sum = sum(col_sum[0:x_center])
    right_sum = sum(col_sum[x_center:-1])
    top_sum = sum(row_sum[0:y_center])
    bottom_sum = sum(row_sum[y_center:-1])
    
    if left_sum < right_sum:
        horizontal_flip = True
    else:
        horizontal_flip = False
        
    return horizontal_flip