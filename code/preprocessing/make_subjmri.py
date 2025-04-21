import argparse
import json
import os, sys
import numpy as np
import pandas as pd
import scipy.io
import concurrent.futures

from nsda import NSDAccess


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subj",
        dest="subject",
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )

    opt = parser.parse_args()
    subject = opt.subject

    root_dir = '../../data'
    output_dir = f'{root_dir}/processed/fmris/{subject}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/whole', exist_ok=True)
    os.makedirs(f'{output_dir}/area', exist_ok=True)

    atlasname = 'HCP_MMP1'

    nsda = NSDAccess(root_dir)
    nsd_expdesign = scipy.io.loadmat(f'{root_dir}/nsddata/experiments/nsd/nsd_expdesign.mat')

    sharedix = nsd_expdesign['sharedix'] - 1

    atlas = nsda.read_atlas_results(subject=subject, atlas=atlasname, data_format='func1pt8mm')

    atlas_to_json = [
        atlas[0].tolist(),
        atlas[1]
    ]
    json.dump(atlas_to_json, open(f'{output_dir}/atlas.json', 'w'), indent=4)

    atlas_to_json = [
        atlas[0].tolist(),
        atlas[1]
    ]
    json.dump(atlas_to_json, open(f'{output_dir}/atlas_{atlasname}.json', 'w'), indent=4)


    behs = pd.DataFrame()
    for i in range(1, 38):
        beh = nsda.read_behavior(subject=subject,
                                 session_index=i)
        behs = pd.concat((behs, beh))

    # Caution: 73KID is 1-based! https://cvnlab.slite.page/p/fRv4lz5V2F/Behavioral-data
    stims_unique = behs['73KID'].unique() - 1  # 9841 unique images
    stims_all = behs['73KID'] - 1  # 27750 trials

    mask = np.isin(stims_all, sharedix[0])
    train_index = stims_all[~mask]
    val_index = stims_all[mask]

    print("[Train/Val Split (INDEX)]")
    print(train_index.shape, val_index.shape)

    with open(f'{output_dir}/nsd_fmri2image.json', 'w') as f:
        index = {
            'train': train_index.tolist(),
            'val': val_index.tolist(),
        }
        json.dump(index, f, indent=4)

    def read_beta_trial(i, subject):
        return nsda.read_betas(subject=subject,
                               session_index=i,
                               trial_index=[],  # empty list as index means get all for this session
                               data_type='betas_fithrf_GLMdenoise_RR',
                               data_format='func1pt8mm')

    with concurrent.futures.ProcessPoolExecutor(max_workers=64) as executor:
        futures = [executor.submit(read_beta_trial, i, subject) for i in range(1, 38)]


        beta_trials = []
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                beta_trial = future.result()
                beta_trials.append(beta_trial)
                print(f"Task {i} completed successfully.")
            except Exception as e:
                print(f"Task {i} failed with exception: {e}")

    betas_all = np.concatenate(beta_trials, axis=0)

    for roi, val in atlas[1].items():
        print(roi, val)
        if val == 0:
            continue
            roi = 'whole'
            betas_roi = betas_all
            print(betas_all.shape)
        else:
            betas_roi = betas_all[:, atlas[0].transpose([2, 1, 0]) == val]
            print(betas_roi.shape)  # (session \times trail, x, y, z)

        if val == 0:
            np.save(f'{output_dir}/whole/nsd_{roi}_betas_mean.npy', betas_roi.mean(axis=0))
            np.save(f'{output_dir}/whole/nsd_{roi}_betas_std.npy', betas_roi.std(axis=0))

        else:
            np.save(f'{output_dir}/area/nsd_{roi}_betas_mean.npy', betas_roi.mean(axis=0))
            np.save(f'{output_dir}/area/nsd_{roi}_betas_std.npy', betas_roi.std(axis=0))

        # Averaging for each stimulus
        betas_roi_ave = []
        for stim in stims_unique:
            # select stim related beats data, (session \times trail, x, y, z) --> (xxx, x, y, z)  --> (x, y, z)
            stim_mean = np.mean(betas_roi[stims_all == stim, :], axis=0)
            betas_roi_ave.append(stim_mean)
        betas_roi_ave = np.stack(betas_roi_ave)
        print(betas_roi_ave.shape)  # (type of stims (images), xyz)

        # Train/Test Split
        # ALLDATA
        betas_tr = []
        betas_te = []

        for idx, stim in enumerate(stims_all):
            if stim in sharedix:  # total 1000 images
                betas_te.append(betas_roi[idx, :])
            else:  # total 9841 - 1000 images
                betas_tr.append(betas_roi[idx, :])

        betas_tr = np.stack(betas_tr)
        betas_te = np.stack(betas_te)

        # AVERAGED DATA
        betas_ave_tr = []
        betas_ave_te = []
        for idx, stim in enumerate(stims_unique):
            if stim in sharedix:
                betas_ave_te.append(betas_roi_ave[idx, :])
            else:
                betas_ave_tr.append(betas_roi_ave[idx, :])
        betas_ave_tr = np.stack(betas_ave_tr)
        betas_ave_te = np.stack(betas_ave_te)

        print("[Train/Test Split]")
        print(betas_tr.shape, betas_te.shape)
        if val == 0:
            for i in range(betas_tr.shape[0]):
                np.save(f'{output_dir}/whole/nsd_betas_tr_{i:06}.npy', betas_tr[i])
            for i in range(betas_te.shape[0]):
                np.save(f'{output_dir}/whole/nsd_betas_te_{i:06}.npy', betas_te[i])

        # Save
        np.save(f'{output_dir}/area/nsd_{roi}_betas_tr.npy', betas_tr)
        np.save(f'{output_dir}/area/nsd_{roi}_betas_te.npy', betas_te)
        np.save(f'{output_dir}/area/nsd_{roi}_betas_ave_tr.npy', betas_ave_tr)
        np.save(f'{output_dir}/area/nsd_{roi}_betas_ave_te.npy', betas_ave_te)
