import os
import re
import glob
import json
import numpy as np
import pandas as pd
import SimpleITK as sitk

from loguru import logger


def extract_study_id(filename):
    pattern = r"[\da-zA-Z]+-[\da-zA-Z]+-[\da-zA-Z]+-[\da-zA-Z]+-[\da-zA-Z]+"
    match = re.search(pattern, filename)
    return match.group(0) if match else None


def crop_image_around_centroid(img, centroid_physical, distance):
    # Convert distance to half distances for cropping
    half_distance = [d / 2 for d in distance]

    # Calculate the physical bounds
    min_bounds_physical = [c - hd for c, hd in zip(centroid_physical, half_distance)]
    max_bounds_physical = [c + hd for c, hd in zip(centroid_physical, half_distance)]

    # Convert physical bounds to index bounds
    min_bounds_index = img.TransformPhysicalPointToIndex(min_bounds_physical)
    max_bounds_index = img.TransformPhysicalPointToIndex(max_bounds_physical)

    # Calculate the region to crop
    start = min_bounds_index
    size = [max_idx - min_idx for max_idx, min_idx in zip(max_bounds_index, min_bounds_index)]

    # Crop the image
    cropped_img = sitk.RegionOfInterest(img, size=size, index=start)
    return cropped_img


def crop_image_around_centroid_with_padding(img, centroid_physical, distance):
    # Convert distance to half distances for cropping
    half_distance = [d / 2 for d in distance]

    # Calculate the physical bounds
    min_bounds_physical = [c - hd for c, hd in zip(centroid_physical, half_distance)]
    max_bounds_physical = [c + hd for c, hd in zip(centroid_physical, half_distance)]

    # Convert physical bounds to index bounds
    min_bounds_index = img.TransformPhysicalPointToIndex(min_bounds_physical)
    max_bounds_index = img.TransformPhysicalPointToIndex(max_bounds_physical)

    # Calculate padding
    img_size = img.GetSize()
    padding = [(abs(min(0, min_idx)), max(0, max_idx - sz)) for min_idx, max_idx, sz in zip(min_bounds_index, max_bounds_index, img_size)]
    pad_lower = [p[0] for p in padding]
    pad_upper = [p[1] for p in padding]

    # Apply padding
    padded_img = sitk.ConstantPad(img, pad_lower, pad_upper, constant=0)

    # Update centroid position after padding
    new_centroid_physical = np.array(centroid_physical) + np.array(pad_lower) * np.array(img.GetSpacing())

    # Convert new centroid to index bounds
    min_bounds_physical = [c - hd for c, hd in zip(new_centroid_physical, half_distance)]
    max_bounds_physical = [c + hd for c, hd in zip(new_centroid_physical, half_distance)]
    min_bounds_index = padded_img.TransformPhysicalPointToIndex(min_bounds_physical)
    max_bounds_index = padded_img.TransformPhysicalPointToIndex(max_bounds_physical)

    # Calculate the region to crop
    start = min_bounds_index
    size = [max_idx - min_idx for max_idx, min_idx in zip(max_bounds_index, min_bounds_index)]

    # Crop the image
    cropped_img = sitk.RegionOfInterest(padded_img, size=size, index=start)
    return cropped_img


def resample_image(img, new_spacing, new_size=None, interpolator=sitk.sitkBSpline):
    img = sitk.Cast(img, sitk.sitkFloat32)
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    new_size = new_size if new_size else [int(round(osz*ospc/nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]

    size_img = img.GetSize()
    size = [x // 2 for x in size_img]
    orientation = np.array(img.GetDirection()).reshape(3, 3)
    orientation = np.round(orientation).astype(int)
    temp = orientation * new_spacing
    vol_center = temp.dot(new_size) / 2
    new_origin = img.TransformIndexToPhysicalPoint(size) - vol_center
    
    resampled_image = sitk.Resample(img, new_size, sitk.Transform(), interpolator,
                            new_origin, new_spacing, img.GetDirection(), 0,
                            img.GetPixelID())
    
    return resampled_image


def extract_centroid(msk, target_label=3):
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(msk)

    centroid_physical = label_shape_filter.GetCentroid(target_label)  # Assuming the heart is labeled with 3
    centroid_index = msk.TransformPhysicalPointToIndex(centroid_physical)  # Convert the physical centroid to index space

    return centroid_physical, centroid_index


def extract_volume_from_directory(mask_dir, target_label=1):
    mask = sitk.ReadImage(mask_dir)
    mask_arr = sitk.GetArrayFromImage(mask)
    spacing = mask.GetSpacing()
    voxel_volume = np.prod(spacing)
    total_volume = np.sum(mask_arr == target_label) * voxel_volume / 1000  # Convert to cm^3
    return total_volume


if __name__ == "__main__":
    heart_label = 1  # label for the heart in the segmentation mask
    spacing = (1.25, 1.25, 5.0)
    size = (128, 128, 48)
    data_source = "../data/dummy_data/"  # path to raw data
    dataset_id = "processed_dummy_1"  # preprocessed dataset id

    save_dir = f"../data/{dataset_id}/"
    target_img_dir = os.path.join(save_dir, "images")
    target_msk_dir = os.path.join(save_dir, "masks")
    target_info_dir = os.path.join(save_dir, "info")

    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_msk_dir, exist_ok=True)
    os.makedirs(target_info_dir, exist_ok=True)

    errors = {}
    info = {
        "Study ID": [],
        "image": [],
        "mask": [],
    }

    # chnage this patterns if you have different data structure
    images = sorted(glob.glob(f"{data_source}/images/*.nii.gz"))
    masks = sorted(glob.glob(f"{data_source}/masks/*.nii.gz"))

    beg = 0
    limit = None
    for ii, (image, mask) in enumerate(zip(images[beg:limit], masks[beg:limit])):
        study_id = extract_study_id(image)
        logger.info(f"Number {ii}: {study_id}")
        try:
            img = sitk.ReadImage(image)
            msk = sitk.ReadImage(mask)

            centroid_physical, _ = extract_centroid(msk, target_label=heart_label)
            distances = [sp * sz for sp, sz in zip(spacing, size)]  # 16 cm in seg/cor; 24 cm in axial

            img = crop_image_around_centroid_with_padding(img, centroid_physical, distances)
            msk = crop_image_around_centroid_with_padding(msk, centroid_physical, distances)

            img = resample_image(img, spacing, size)
            msk = resample_image(msk, spacing, size, interpolator=sitk.sitkNearestNeighbor)

            img_arr = sitk.GetArrayFromImage(img)
            msk_arr = sitk.GetArrayFromImage(msk)

            img_arr = img_arr * (msk_arr == heart_label)
            msk_arr = (msk_arr == heart_label).astype(np.uint8)

            img_2 = sitk.GetImageFromArray(img_arr)
            img_2.CopyInformation(img)
            msk_2 = sitk.GetImageFromArray(msk_arr)
            msk_2.CopyInformation(msk)

            logger.success(f"Processed {study_id} successfully.")


        except Exception as e:
            errors[study_id] = str(e)
            logger.error(f"Error processing {study_id}: {e}")
            continue
        
        img_path = os.path.join(target_img_dir, study_id + ".nii.gz")
        msk_path = os.path.join(target_msk_dir, study_id + ".nii.gz")
        sitk.WriteImage(img_2, img_path)
        sitk.WriteImage(msk_2, msk_path)

        info["Study ID"].append(study_id)
        info["image"].append(img_path)
        info["mask"].append(msk_path)


    # Save errors to a JSON file
    with open(os.path.join(target_info_dir, "errors.json"), "w") as f:
        json.dump(errors, f, indent=4)

    # Add demographic information
    info_df = pd.DataFrame(info)
    all_patient_info = pd.read_csv("../data/metadata/patient_information_113528_19-9-2024.csv")
    info_df = info_df.merge(all_patient_info, on="Study ID", how="left")

    # add volume information
    info_df["volume"] = info_df["mask"].apply(extract_volume_from_directory)

    # add BMI
    info_df["BMI"] = info_df["Weight"] / info_df["Height"] ** 2

    # change sex to float
    info_df.replace("female", 0.0, inplace=True)
    info_df.replace("male", 1.0, inplace=True)

    # save information
    info_df.to_csv(os.path.join(target_info_dir, "info.csv"), index=False)