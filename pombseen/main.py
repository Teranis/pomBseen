from skimage import filters, measure
import numpy as np
from skimage.segmentation import clear_border as sk_clear_border
from skimage.morphology import binary_dilation, remove_small_objects, binary_closing
from skimage.util import invert
from skimage.filters import unsharp_mask

def norm_img(img): # norms it to 0 to 1
    img_min = np.min(img)
    img = img-img_min
    img_max = np.max(img)  
    img = img/img_max
    print(f"""
          Minumum original pixel: {img_min} \n
          Maximum original pixel: {img_max}
          """)
    return img
    
def import_image(image, radius, amount):# this performs the operations which are in this function other than importing
    image = unsharp_mask(image, radius=radius, amount=amount, preserve_range=True) # sharpen image using unsharp mask
    return image

def thresh_binarize(image, block_size, offset, footprint, inverse_bw_max_pix, connectivity): # threshold and binarize image, also includes halo improvements
    adaptive_threshold = filters.threshold_local(image, block_size=block_size, offset=offset)  # make adaptive threshold
    image = image > adaptive_threshold

    image = binary_dilation(image, footprint=footprint) # these all all build in skimage functions. For details see skimage documentation online
    image = remove_small_objects(image, min_size=inverse_bw_max_pix, connectivity=connectivity)
    image = binary_closing(image, footprint=footprint)

    return image

def inverse_bw(image, inverse_bw_max_pix, connectivity): # inverses pictures, cleans pic
    image = remove_small_objects(image, min_size=inverse_bw_max_pix, connectivity=connectivity) # these all all build in skimage functions. For details see skimage documentation online
    imageInverse1 = invert(image)
    imageInverse2 = remove_small_objects(imageInverse1, min_size=inverse_bw_max_pix, connectivity=connectivity)
    return imageInverse1, imageInverse2

def clear_border(image, clear_border_conn, clear_border_max_pix):
    image = sk_clear_border(image, clear_border_conn) # clearing all pixels touching the edge
    image = remove_small_objects(image, clear_border_max_pix) # Remove small objects, the ones which might have been created from removing borders
    return image

def convex_filter(image, connectivity, ConvexFilterSlope, ConvexFilterIntercept, min_size, max_size):

    CC = measure.label(image, connectivity=connectivity) # label objects

    unique_labels, label_counts = np.unique(CC, return_counts=True) # get labels and with how many pixels they are associated with
    valid_labels = unique_labels[(label_counts >= min_size) & (label_counts <= max_size)] # filter out labels which have too many/ too little pixels

    CC_filtered1 = np.zeros_like(CC) # create empty picture with dimensions as original
    valid_mask = np.isin(CC, valid_labels) # create mask from valid labels
    CC_filtered1[valid_mask] = CC[valid_mask] # copy only pieces which where in the mask

    properties = measure.regionprops(CC_filtered1) # get properties

    num_regions = len(properties) # get length of properties (in effect how many labels are there)

    label = np.empty(num_regions) # set up empty arrays to carry relevant poperies
    area = np.empty(num_regions)
    convex_area = np.empty(num_regions)
    major_axis_length = np.empty(num_regions)
    minor_axis_length = np.empty(num_regions)
    
    for i, prop in enumerate(properties): #read out relevant poperies
        label[i] = prop.label
        area[i] = prop.area
        convex_area[i] = prop.convex_area
        major_axis_length[i] = prop.major_axis_length
        minor_axis_length[i] = prop.minor_axis_length

    area_ratio = convex_area / area # perform calculations (This should be the same as was done in matlab)
    aspect_ratio = major_axis_length / minor_axis_length
    y_values = (ConvexFilterSlope * area_ratio) - ConvexFilterIntercept
    label_mask = aspect_ratio > y_values
    valid_labels = label[label_mask]

    CC_filtered2 = np.zeros_like(CC_filtered1) # same as above (only copy desired objects)
    valid_mask = np.isin(CC_filtered1, valid_labels)
    CC_filtered2[valid_mask] = CC_filtered1[valid_mask]
    
    return CC_filtered1, CC_filtered2

def pomBseg(image, sharpen_image, radius, amount, block_size, offset, footprint, inverse_bw_max_pix, connectivity, clear_border_conn, clear_border_max_pix, ConvexFilterSlope, ConvexFilterIntercept, min_size, max_size):
    image = norm_img(image)
    if sharpen_image == True:
        sharpened_image = import_image(image, radius, amount) # sharpen image
    else:
        sharpened_image = image
    offset = offset / 100
    binary_image = thresh_binarize(sharpened_image, block_size, offset, footprint, inverse_bw_max_pix, connectivity) # threshold image
    imageInverse1, imageInverse2 = inverse_bw(binary_image, inverse_bw_max_pix, connectivity) # invert image and remove small areas from inside cells and from background
    imageFiltered = clear_border(imageInverse2, clear_border_conn, clear_border_max_pix) # clearing edge bordering cells
    CC_filtered1, CC_filtered2 = convex_filter(imageFiltered, connectivity, ConvexFilterSlope, ConvexFilterIntercept, min_size, max_size) # convex and size filter

    return CC_filtered2

def show_quick(imgs):
    num = len(imgs)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, num, 1)
    ax.axis('off')
    ax.imshow(imgs[0], cmap='viridis')
 
    for i, img in enumerate(imgs[1:]):
        i+=2
        fig.add_subplot(1, num, i, sharex=ax, sharey=ax)
        plt.axis('off')
        plt.imshow(img, cmap='viridis')
    
    plt.show()

def pomBsegNuc(image, seg, connectivity, offset, min_size, max_size, max_nuclei, rel_size_max):

    id = 1
    background_label = 0

    image = norm_img(image)
    CC_filtered = np.zeros_like(seg) # initiate new img
    unique_labels, counts = np.unique(seg, return_counts=True) # get cell ids 
    for label, count in zip(unique_labels, counts): # iterate over cells 
        if label == background_label: # skip background
            continue

        cell = np.where(seg == label, 1, 0) # create mask for a cell
        masked_image = np.multiply(image, cell) # apply mask
        unique_pix_num = len(np.unique(masked_image)) # get number of unique pixel number for bins
        hist, bins = np.histogram(masked_image, bins=np.linspace(masked_image.min(), masked_image.max(), unique_pix_num)) # bin img
        hist = hist[1:] # cut out 0
        bins = bins[1:]
        thresh = filters.threshold_otsu(image, hist=(hist, bins)) # get threshold
        offset = offset / 100
        thresh += offset # apply offset
        binary_img = masked_image > thresh # apply thresh

        CC = measure.label(binary_img, connectivity=connectivity) # segment
        unique_labels_cell, label_counts_cell = np.unique(CC[CC != background_label], return_counts=True) # get segmentation areas and sizes
        max_vals = np.sort(label_counts_cell)[-max_nuclei:] # get the max_nuclei-largest ones
        unique_labels_cell_max = unique_labels_cell[
            (label_counts_cell >= min_size) 
            & (label_counts_cell <= max_size) 
            & (np.isin(label_counts_cell, max_vals))
            & (label_counts_cell <= rel_size_max * count)
            ] # filter for the largest ones, which are in the size bracket
        if unique_labels_cell_max.size == 0: # ignore cells which dont have any applicable seg areas and show user warning
            print(f'Found no nucleus in cell {label}')
        else:
            try:
                for nuc in unique_labels_cell_max: # iterate over seg areas ids and assign them a unique ID in new img
                    CC_filtered[CC == nuc] = id
                    id += 1
            except:
                print(f'Error: failed to apply seg for cell {label}') # this gave a few errors, so here is some debug help
                print(CC.shape, CC_filtered.shape, unique_labels_cell_max, (np.isin(CC, unique_labels_cell_max)).shape)
    return CC_filtered