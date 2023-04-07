from OCT.formats.OCTVol import OCTVol
import numpy as np
import skimage.transform as skt
from glob import glob
import os
import warnings


class NoRingScanFoundError(Exception): pass


def pre_process_and_save_batch(data_path, stack_save_name):
    """
    Pre-processes all the vol files with one B scan in a folder and saves the results

    Parameters
    ----------
    data_path : str
        Path to the folder
    stack_save_name : str
        Name of the file for the pre-processed vol files stack to be saved
    """
    # Pre-process bacth
    pre_processed_stack = pre_process_batch(data_path)

    # Save the stack
    np.save(os.path.join(data_path, stack_save_name+".npy"), pre_processed_stack)


def pre_process_batch(data_path):
    """
    Pre-processes all the vol files with one B scan in a folder and returns the results

    Parameters
    ----------
    data_path : str
        Path to the folder

    Returns
    -------
    numpy.ndarray
        Pre-processed OCT scans stacked, with the shape of N*C*H*W, where N is batch, C is input channel (0
        pre-processed B scan and 1 label image), and H and W are the height and width of the image

    Raises
    ------
    NoRingScanFoundError
        If the folder does not exist or the folder does not contain .vol ring scans

    Notes
    -----
    This function does not check if an input vol file is a peripapillary ring scan, so this needs to be checked
    separately, as this is out of the scope of this function and its task
    """
    # Find .vol files in the directory
    vol_files_list = glob(os.path.join(data_path, "*.vol"))

    # Go through each volume and for a stack of B Scans
    pre_processed_stack = []
    for vol_file_path in vol_files_list:
        try:
            # Read the vol file
            oct_vol = OCTVol(vol_file_path)

            # Pre-process only if the file has one BScan
            if oct_vol.header['num_b_scans'] == 1:
                # Pre-process the .vol file
                b_scan, label_image = pre_process(oct_vol)

                # Combine the b scan and the label image and add it to the stack
                pre_processed_stack.append(np.concatenate((b_scan[np.newaxis], label_image[np.newaxis]), axis=0))
        except Exception as e:
            warnings.warn("{} This vol file resulted in an error, we are moving on {}".format(e, vol_file_path))

    # Check if the pre-processed batch contains anything, if not raise an error
    if not pre_processed_stack:
        raise NoRingScanFoundError("Either the folder doesn't exist or the folder doesn't contain .vol ring scans!")

    # Return the pre-processed stack as numpy
    return np.array(pre_processed_stack)


def pre_process(oct_vol):
    """
    Pre-process a peripapillary ring scan

    Pre-processing by forming a label image based on the segmentation of ILM, BM, and pRNFL, normalizing the B Scan,
    and performing flattening with BM as the reference, cropping vertically to 0.45 mm and resizing to 512x512 on both
    the B scan and label image

    Parameters
    ----------
    oct_vol : OCTVol
        An OCT scan in the OCTVol format

    Returns
    -------
    tuple
        Pre-processed B scan and label image in the numpy.ndarray format

    Notes
    -----
    I understand that it makes sense to make the label image as two separate channels with one coding the retina
    tissue (between ILM and BM as 1 (foreground) and the rest as 0 (background)) and another similarly coding the
    pRNFL layer, since assigning numbers other than 0 and 1 to categorical entities is kinda wrong. But I still made
    the label image as a single channel since pRNFL is part of the retina tissue so assigning one to pRNFL and 1/2 to
    the rest of the tissue will highlight the pRNFL region, which is in line with our findings that pRNFL is the
    most import part of peripapillary ring scans.

    The images are cropped and centered vertically to make sure all images cover similar region (0.45mm) and are in
    the similar location on the image, in order to make sure that the model will not be learning the image
    translation. This way we can eliminate the augmentation step (only augmentation with vertical shift seemed to be
    relevant, so we reduced/eliminated the variance only in that direction). 0.45mm was chosen because I couldn't find
    any ring scan with retina tissue thickness of more than 0.45mm at any point. In case the thickness is more than
    0.45mm or the retina tissue is too close to the upper or lower border, this will cause an error.
    """
    # Transfer the image with the formula provided by HE (pixel intensity is the 4th root of the read values)
    b_scan = oct_vol.b_scans.squeeze()
    b_scan = b_scan ** 0.25
    b_scan[b_scan > 1] = 0

    # Extract segmentation lines
    # The segmentation is in subpixel so I round them to pixels
    ilm_seg = np.floor(oct_vol.b_scan_header['boundary_1'].squeeze()).astype(int)
    bm_seg = np.floor(oct_vol.b_scan_header['boundary_2'].squeeze()).astype(int)
    rnfl_seg = np.floor(oct_vol.b_scan_header['boundary_3'].squeeze()).astype(int)

    # Normalize only the tissue part of the B scan (foreground) and set the rest (background) to zero
    for i_a_scan in range(oct_vol.header['size_x']):
        b_scan[:ilm_seg[i_a_scan], i_a_scan] = np.nan
        b_scan[bm_seg[i_a_scan]+1:, i_a_scan] = np.nan
    b_scan = (b_scan - np.nanmin(b_scan)) / (np.nanmax(b_scan) - np.nanmin(b_scan))
    b_scan = np.nan_to_num(b_scan)

    # Flatten B Scan and segmentation lines based on the BM segmentation
    bm_lowest_point = np.max(bm_seg)
    for i_a_scan in range(oct_vol.header['size_x']):
        shift = bm_lowest_point - bm_seg[i_a_scan]
        b_scan[:, i_a_scan] = np.roll(b_scan[:, i_a_scan], shift)
        ilm_seg[i_a_scan] = ilm_seg[i_a_scan] + shift
        bm_seg[i_a_scan] = bm_seg[i_a_scan] + shift
        rnfl_seg[i_a_scan] = rnfl_seg[i_a_scan] + shift

    # Form the label image based on the segmentation
    label_image = np.zeros_like(b_scan)
    for i_a_scan in range(oct_vol.header['size_x']):
        label_image[ilm_seg[i_a_scan]:rnfl_seg[i_a_scan] + 1, i_a_scan] = 1
        label_image[rnfl_seg[i_a_scan] + 1:bm_seg[i_a_scan] + 1, i_a_scan] = 1 / 2

    # Crop the B scan and label image vertically to 0.45 mm in a way that the foreground is in the center
    maximum_thickness = np.max(bm_seg - ilm_seg) + 1
    crop_thickness = np.ceil(0.45 / oct_vol.header['scale_z']).astype(int)
    crop_upper_boundary = np.min(ilm_seg) - ((crop_thickness - maximum_thickness) / 2).astype(int)
    b_scan = b_scan[crop_upper_boundary:crop_upper_boundary + crop_thickness]
    label_image = label_image[crop_upper_boundary:crop_upper_boundary + crop_thickness]

    # Resize
    b_scan = skt.resize(b_scan, (512, 512))
    label_image = skt.resize(label_image, (512, 512))

    return b_scan, label_image
