from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import mne
import numpy as np
import os

from mne.utils import verbose, logger

subjects_dir = mne.utils.get_subjects_dir()


@verbose
def combine_medial_labels(labels, subject='fsaverage', surf='white',
                          dist_limit=0.02):
    from scipy.spatial.distance import cdist
    subjects_dir = mne.get_config('SUBJECTS_DIR')
    rrs = dict((hemi, mne.read_surface(os.path.join(subjects_dir, subject, 'surf',
                                       '%s.%s' % (hemi, surf)))[0] / 1000.)
               for hemi in ('lh', 'rh'))
    use_labels = list()
    used = np.zeros(len(labels), bool)
    logger.info('Matching medial regions for %s labels on %s %s, d=%0.1f mm'
                % (len(labels), subject, surf, 1000 * dist_limit))
    for li1, l1 in enumerate(labels):
        if used[li1]:
            continue
        used[li1] = True
        use_label = l1.copy()
        rr1 = rrs[l1.hemi][l1.vertices]
        for li2 in np.where(~used)[0]:
            l2 = labels[li2]
            same_name = (l2.name.replace(l2.hemi, '') ==
                         l1.name.replace(l1.hemi, ''))
            if l2.hemi != l1.hemi and same_name:
                rr2 = rrs[l2.hemi][l2.vertices]
                mean_min = np.mean(mne.surface._compute_nearest(
                    rr1, rr2, return_dists=True)[1])
                if mean_min <= dist_limit:
                    use_label += l2
                    used[li2] = True
                    logger.info('  Matched: ' + l1.name)
        use_labels.append(use_label)
    logger.info('Total %d labels' % (len(use_labels),))
    return use_labels


def load_labsn_7_labels():
    label_str = os.path.join(subjects_dir, "fsaverage/label/*labsn*")
    rtpj_str = os.path.join(subjects_dir, 'fsaverage/label/rh.RTPJ.label')
    label_fnames = glob.glob(label_str)
    label_fnames.insert(0, rtpj_str)
    labels = [mne.read_label(fn, subject='fsaverage') for fn in label_fnames]
    labels = sorted(labels, key=lambda x: x.name)

    return labels


def load_hcpmmp1_combined():

    labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1_combined')
    labels = sorted(labels, key=lambda x: x.name)
    labels = combine_medial_labels(labels)

    return labels

def load_hcpmmp1():

    labels = mne.read_labels_from_annot('fsaverage', parc='HCPMMP1_combined')
    labels = sorted(labels, key=lambda x: x.name)

    return labels


def load_labsn_hcpmmp1_7_labels(include_visual=False):

    hcp_mmp1_labels = mne.read_labels_from_annot('fsaverage',
                                                 parc='HCPMMP1_combined')
    hcp_mmp1_labels = combine_medial_labels(hcp_mmp1_labels)
    label_names = [l.name for l in hcp_mmp1_labels]

    ips_str = os.path.join(subjects_dir, "fsaverage/label/*.IPS-labsn.label")
    ips_fnames = glob.glob(ips_str)
    ips_labels = [mne.read_label(fn, subject='fsaverage') for fn in ips_fnames]

    labels = list()
    pmc_labs = [l for l in hcp_mmp1_labels if 'Premotor Cortex' in l.name]
    eac_labs = [l for l in hcp_mmp1_labels if 'Early Auditory Cortex' in l.name]

    # this is in place of original rtpj
    ipc_labs = [l for l in hcp_mmp1_labels if 'Inferior Parietal Cortex' in l.name]

    labels = list()
    labels.extend(pmc_labs)
    labels.extend(eac_labs)
    labels.extend(ips_labels)
    labels.extend(ipc_labs)

    # optionally include early visual regions as controls
    if include_visual:
        prim_visual = [l for l in hcp_mmp1_labels if 'Primary Visual Cortex' in l.name]

        # there should be only one b/c of medial merge
        prim_visual = prim_visual[0]

        early_visual_lh = label_names.index("Early Visual Cortex-lh")
        early_visual_rh = label_names.index("Early Visual Cortex-rh")
        early_visual_lh = hcp_mmp1_labels[early_visual_lh]
        early_visual_rh = hcp_mmp1_labels[early_visual_rh]
        visual = prim_visual + early_visual_lh + early_visual_rh

        labels.append(visual)

    return labels


def load_labsn_hcpmmp1_7_plus_vision_labels():
    return load_labsn_hcpmmp1_7_labels(include_visual=True)


def load_labsn_hcpmmp1_av_rois_small():

    hcp_mmp1_labels = mne.read_labels_from_annot('fsaverage',
                                                 parc='HCPMMP1_combined')
    hcp_mmp1_labels = combine_medial_labels(hcp_mmp1_labels)
    label_names = [l.name for l in hcp_mmp1_labels]

    #prim_visual_lh = label_names.index("Primary Visual Cortex (V1)-lh")
    #prim_visual_rh = label_names.index("Primary Visual Cortex (V1)-rh")
    #prim_visual_lh = hcp_mmp1_labels[prim_visual_lh]
    #prim_visual_rh = hcp_mmp1_labels[prim_visual_rh]
    prim_visual = [l for l in hcp_mmp1_labels if 'Primary Visual Cortex' in l.name]

    # there should be only one b/c of medial merge
    prim_visual = prim_visual[0]

    early_visual_lh = label_names.index("Early Visual Cortex-lh")
    early_visual_rh = label_names.index("Early Visual Cortex-rh")
    early_visual_lh = hcp_mmp1_labels[early_visual_lh]
    early_visual_rh = hcp_mmp1_labels[early_visual_rh]

    #visual_lh = prim_visual_lh + early_visual_lh
    #visual_rh = prim_visual_rh + early_visual_rh

    visual = prim_visual + early_visual_lh + early_visual_rh
    labels = [visual]

    #labels = [visual_lh, visual_rh]

    eac_labs = [l for l in hcp_mmp1_labels if 'Early Auditory Cortex' in l.name]
    labels.extend(eac_labs)

    tpo_labs = [l for l in hcp_mmp1_labels if 'Temporo-Parieto-Occipital Junction' in l.name]
    labels.extend(tpo_labs)

    dpc_labs = [l for l in hcp_mmp1_labels if 'DorsoLateral Prefrontal Cortex' in l.name]
    labels.extend(dpc_labs)

    ## extra labels KC wanted
    #pmc_labs = [l for l in hcp_mmp1_labels if 'Premotor Cortex' in l.name]
    #labels.extend(pmc_labs)

    #ips_str = glob.glob(os.path.join(subjects_dir, "fsaverage/label/*IPS*labsn*"))
    #ips_labs = [mne.read_label(fn, subject='fsaverage') for fn in ips_str]
    #labels.extend(ips_labs)

    #rtpj_labs = [l for l in hcp_mmp1_labels if 'Inferior Parietal Cortex-rh' in l.name]
    #labels.extend(rtpj_labs)

    return labels
