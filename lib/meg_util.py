
import mne
import os

from lib import sps_info
from mne.minimum_norm import read_inverse_operator

cond_map = sps_info.condition_map


def load_subject_data(subject_name, data_dir, subject_dir, labels,
                      epochs_fmt, sps_dir, tmin, tmax, cond):

    epochs_fname = epochs_fmt % subject_name
    epochs_dir = os.path.join(sps_dir, subject_name, 'epochs')
    epochs = mne.read_epochs(os.path.join(epochs_dir, epochs_fname))
    epochs = epochs[cond_map[cond]].crop(tmin=tmin, tmax=tmax)

    subject_id = "AKCLEE_1%02d" % (int(subject_name.split('_')[-1]),)

    labels_morphed = list()
    for lab in labels:
        if isinstance(lab, mne.Label):
            labels_morphed.append(lab.copy())
        elif isinstance(lab, mne.BiHemiLabel):
            labels_morphed.append(lab.lh.copy() + lab.rh.copy())

    for i, l in enumerate(labels_morphed):
        if l.subject == subject_name:
            continue
        elif l.subject == 'unknown':
            print("uknown subject for label %s" % l.name,
                  "assuming if is 'fsaverage' and morphing")
            l.subject = 'fsaverage'

        if isinstance(l, mne.Label):
            l.values.fill(1.0)
            labels_morphed[i] = l.morph(subject_to=subject_id,
                                        subjects_dir=subject_dir)
        elif isinstance(l, mne.BiHemiLabel):
            l.lh.values.fill(1.0)
            l.rh.values.fill(1.0)
            labels_morphed[i].lh = l.lh.morph(subject_to=subject_id,
                                              subjects_dir=subject_dir)
            labels_morphed[i].rh = l.rh.morph(subject_to=subject_id,
                                              subjects_dir=subject_dir)

    inv_dir = os.path.join(sps_dir, subject_name, 'inverse')
    inv_fname = "%s-55-sss-meg-eeg-fixed-inv.fif" % subject_name
    inv = read_inverse_operator(os.path.join(inv_dir, inv_fname))

    return epochs, inv, labels_morphed
