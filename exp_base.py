"""exp_base.py - extracted from SAM_CapsNet_Ablation v17.ipynb.
Imports + UCF11VideoDataset + make_stratified_splits, reused verbatim
so the gating experiment inherits the notebook tested data pipeline.
"""
import os, cv2, json, math, random, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold
import re

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


import hashlib, pickle


def _group_of(video_path, cls_name):
    """Map a clip path to its source-video group id (leakage-free splitting).

    UCF-11 (nested): <class>/v_<class>_<NN>/v_<class>_<NN>_<clip>.mpg
      -> the parent directory (v_<class>_<NN>) IS the group.
    UCF101 (flat):   <class>/v_<Class>_g<NN>_c<NN>.avi
      -> all clips sharing the g<NN> token come from one source video; the
         group is the v_<Class>_g<NN> prefix (this is the split key UCF101's
         own official train/test lists use to prevent actor/background leak).
    Generic fallback: strip a trailing _<clip> index so siblings coalesce.
    Prefixing with the class name keeps ids globally unique across classes.
    """
    p = Path(video_path)
    parent = p.parent.name
    if parent and parent != cls_name and parent.lower() != 'annotation':
        return "{}/{}".format(cls_name, parent)          # UCF-11 nested dir
    stem = p.stem
    m = re.match(r'(.+_g\d+)_c?\d+$', stem)               # UCF101 g<NN> token
    if m:
        return "{}/{}".format(cls_name, m.group(1))
    m = re.match(r'(v_.+?_\d+)_\d+$', stem)               # generic clip index
    return "{}/{}".format(cls_name, m.group(1) if m else stem)


class UCF11VideoDataset(Dataset):
    """
    UCF-11 dataset with fixed-rate sampling at 5fps (from Paper 2).

    Sampling strategy:
      - Fixed-rate anchors: one frame every (native_fps / sample_fps) frames.
        This gives physically comparable motion intervals across clips
        regardless of original recording frame rate.
      - NO context-window averaging: clean single frames are passed to the
        model so attention branches receive sharp, differentiable inputs.

    Why NOT context-window averaging for SAM-CapsNet:
      Context-window averaging works in Paper 2 because the CoM trajectory
      is extracted before averaging -- two separate signals feeding separate
      branches. In SAM-CapsNet, the attention branches (temporal, spatial,
      channel) all need sharp per-frame differences to compute meaningful
      attention maps. Averaging 11 frames homogenises all frame
      representations, removing exactly the signal the attention branches
      need to differentiate themselves. This was confirmed empirically:
      context-window averaging reduced all 8 ablation configs to within
      2.41pp of each other, with the full model scoring LOWEST.

    Disk cache: processed tensors saved on first run, reloaded on
    subsequent epochs/seeds for 10-50x speedup.
    """
    VIDEO_EXTS = ('*.avi', '*.mp4', '*.mov', '*.mpg', '*.mpeg')

    def __init__(self, root_dir, sequence_length=16,
                 target_size=(224, 224), sample_fps=5.0,
                 augment=False, cache_dir='.cache', clip_start=0.0):
        self.root_dir        = Path(root_dir)
        self.sequence_length = sequence_length
        self.target_size     = target_size
        self.sample_fps      = sample_fps
        self.augment         = augment
        # clip_start (2026-08-17): fractional temporal position of the sampling
        # window. 0.0 = start of video (the historical, and default, behaviour);
        # 1.0 = the last window that still fits. At 5 fps a 16-frame clip spans
        # ~3.2 s while UCF101 clips average ~7 s, so a single window at
        # clip_start=0 SEES LESS THAN HALF OF A TYPICAL VIDEO. Published UCF101
        # numbers are multi-clip (usually x multi-crop) and ours were
        # single-view, which the run ledger flags as a caveat on every
        # comparison table. Build views with `view(frac)` and average them at
        # EVAL time only -- training stays single-view, so no trained model and
        # no existing result changes.
        self.clip_start      = float(clip_start)
        # CACHE_DIR env overrides the caller's cache_dir so the fp16 .npy cache
        # can live on native ext4 instead of the /mnt/d 9p bridge. On this 7GB
        # host the >5GB working set never fits in page cache, so every epoch
        # re-reads cold; 9p RPC latency then dominates (GPU idles). Same md5
        # keying + same bytes -> identical training, just local-disk fast.
        env_cache = os.environ.get('CACHE_DIR')
        cache_dir = env_cache if env_cache else cache_dir
        self.cache_dir       = Path(cache_dir) if cache_dir else None

        aug_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomRotation(degrees=8),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        clean_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        self.transform = aug_tf if augment else clean_tf

        self.samples      = []
        self.groups       = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self._setup()

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _setup(self):
        if not self.root_dir.exists():
            raise FileNotFoundError("Dataset not found: {}".format(self.root_dir))
        classes = sorted(d.name for d in self.root_dir.iterdir()
                         if d.is_dir() and not d.name.startswith(('.','__')))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for i, c in enumerate(classes)}
        print("Found {} classes: {}".format(len(classes), classes))
        for cls in classes:
            p = self.root_dir / cls
            vids = []
            for ext in self.VIDEO_EXTS:
                vids.extend(p.glob(ext))
            for sub in p.iterdir():
                if sub.is_dir() and sub.name != 'annotation':
                    for ext in self.VIDEO_EXTS:
                        vids.extend(sub.glob(ext))
            for vp in vids:
                self.samples.append((str(vp), self.class_to_idx[cls]))
                self.groups.append(_group_of(vp, cls))
            print("  {}: {} videos".format(cls, len(vids)))
        print("Total: {} videos ({} source groups)".format(
            len(self.samples), len(set(self.groups))))

    def view(self, clip_start):
        """A sibling dataset reading a different temporal window, same samples.

        Shallow copy on purpose: `samples`/`groups`/`class_to_idx` are shared by
        reference, so (a) building N views costs no directory re-scan, and
        (b) every view indexes IDENTICALLY -- view k's item i is the same source
        clip as view 0's item i. Multi-clip averaging depends on that alignment,
        and re-running `_setup()` per view would risk a different glob order.
        """
        import copy
        v = copy.copy(self)
        v.clip_start = float(clip_start)
        return v

    def _cache_key(self, path):
        sig = "{}|{}|{}|{}".format(
            path, self.sequence_length, self.sample_fps, str(self.target_size))
        # The offset joins the key ONLY when non-zero. clip_start=0.0 must keep
        # producing the historical key or the existing 67 GB fp16 cache is
        # invalidated wholesale and every clip re-decodes -- while a key that
        # ignored the offset would be far worse, silently serving view 0's
        # frames for every view and making multi-clip eval a no-op that merely
        # averages a tensor with itself.
        if self.clip_start:
            sig += "|cs{:.4f}".format(self.clip_start)
        return hashlib.md5(sig.encode()).hexdigest()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # fp16 .npy cache: np.load is leak-free, unlike pickle.loads of a torch
        # tensor (which leaks its full storage and OOMs this ~8GB host).
        if self.cache_dir and not self.augment:
            npy = self.cache_dir / (self._cache_key(path) + '.npy')
            if npy.exists():
                try:
                    return torch.from_numpy(np.load(npy).astype(np.float32)), label
                except Exception:
                    pass

        frames = self._load_video(path)
        if frames is None:
            frames = torch.zeros(self.sequence_length, 3, *self.target_size)

        if self.cache_dir and not self.augment:
            try:
                npy = self.cache_dir / (self._cache_key(path) + '.npy')
                np.save(npy, frames.numpy().astype(np.float16))
            except Exception:
                pass

        return frames, label

    def _load_video(self, path):
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                return None
            native_fps = cap.get(cv2.CAP_PROP_FPS)
            if native_fps <= 0 or native_fps > 240:
                native_fps = 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return None

            # Fixed-rate anchors at sample_fps, offset by clip_start.
            # `span` is what the window covers; `slack` is how much of the video
            # is left over, so clip_start=1.0 lands the window flush against the
            # end. Short videos have slack 0 and every view collapses onto the
            # same frames -- correct (there is only one window), and the reason
            # multi-clip gains are dataset-dependent rather than free.
            step    = max(1, int(round(native_fps / self.sample_fps)))
            span    = (self.sequence_length - 1) * step
            slack   = max(0, total - 1 - span)
            base    = int(round(self.clip_start * slack))
            anchors = [min(base + i * step, total - 1)
                       for i in range(self.sequence_length)]

            # Sequential read: one forward pass, no random seeking
            cap.set(cv2.CAP_PROP_POS_FRAMES, anchors[0])
            raw   = {}
            cur   = anchors[0]
            last  = anchors[-1]
            while cur <= last:
                ret, frame = cap.read()
                if ret:
                    raw[cur] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cur += 1
            cap.release()

            if not raw:
                # Seek-hostile container: on some UCF-11 .mpg files ANY
                # CAP_PROP_POS_FRAMES seek makes every subsequent read() fail,
                # while a plain sequential read returns every frame. Without this
                # retry _load_video returns None, __getitem__ substitutes
                # torch.zeros(...) AND CACHES IT, so the clip stays a black video
                # for every later run and seed. Full scan 2026-08-07: 1 of 1600
                # UCF-11 clips affected (basketball/v_shooting_25/v_shooting_25_06,
                # 12 frames).
                # Retry only on failure, never unconditionally: seeking perturbs
                # MPEG-1 decode slightly (mean |delta| ~4/255), so dropping the
                # seek outright would change the pixels of all 1599 healthy clips
                # and silently invalidate the .cache/*.npy frame cache.
                cap = cv2.VideoCapture(path)
                cur = 0
                while cur <= last:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    raw[cur] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cur += 1
                cap.release()

            if not raw:
                return None

            frames = []
            for anchor in anchors[:self.sequence_length]:
                # Single clean frame — no averaging
                fi = min(raw.keys(), key=lambda k: abs(k - anchor))
                frames.append(self.transform(raw[fi]))

            while len(frames) < self.sequence_length:
                frames.append(frames[-1])
            return torch.stack(frames[:self.sequence_length])

        except Exception as e:
            print("Error loading {}: {}".format(path, e))
            return None

    def get_labels(self):
        return [s[1] for s in self.samples]

    def get_groups(self):
        """Source-video group id per sample, for leakage-free splitting.

        UCF-11 packs ~4-6 clips per source video into one directory
        (e.g. class/v_jumping_01/{v_jumping_01_01.mpg, ..._02.mpg}).
        Clips in a group share actor/background/camera, so they must
        never straddle the train/test boundary. See [[make_group_splits]].
        """
        return list(self.groups)

    def clear_cache(self):
        if self.cache_dir and self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir()
            print("Cache cleared.")


class FrozenFeatureDataset(Dataset):
    """Serves POOLED frozen-stem features in place of decoded frames.

    Motivation: with a frozen stem, 12 epochs re-derive the same features 12
    times, and on UCF101 that streams ~540 GB off disk per run (67 GB of fp16
    frames x ~8 passes) to compute something that never changes. Caching the
    stem's output instead makes a run head-bound: 221 KB/clip, ~2.9 GB total,
    small enough to sit in page cache.

    WHY POOLING BEFORE THE PROJECTION IS EXACT. The model computes
        pre_caps_norm(adaptive_avg_pool2d(proj(body(x)), 3)).
    `proj` is a 1x1 conv -- a per-POSITION linear map over channels -- and
    `adaptive_avg_pool2d` is a per-CHANNEL linear map over positions, so the two
    commute exactly:  pool(proj(f)) == proj(pool(f)).  We therefore cache
    pool(body(x), 3) at [T, C, 3, 3] and let the trainable projection run on the
    pooled tensor. `adaptive_avg_pool2d(., 3)` then sees a 3x3 input and is the
    identity, so the model's forward needs no branch. This shortcut is valid
    ONLY while the last stem layer is 1x1; a 3x3 projection would break it.

    Mirrors UCF11VideoDataset's interface (samples/groups/get_labels/get_groups/
    view) so the splitters and multi-clip eval work unchanged.
    """

    def __init__(self, base, stem, feat_dir, clip_start=0.0):
        self.base = base
        self.stem = stem
        self.feat_dir = Path(feat_dir)
        self.clip_start = float(clip_start)
        # shared BY REFERENCE: identical indexing to the frame dataset, so a
        # split computed on one is valid on the other
        self.samples = base.samples
        self.groups = base.groups
        self.class_to_idx = base.class_to_idx
        self.idx_to_class = base.idx_to_class

    def feat_key(self, path):
        # The stem is IN the key: two stems produce different features at the
        # same [C,3,3] shape, so a key without it would silently serve CLIP
        # features to a MobileNet run (or vice versa) with no shape error to
        # catch it.
        sig = "{}|{}|{}|{}|{}|cs{:.4f}".format(
            path, self.base.sequence_length, self.base.sample_fps,
            str(self.base.target_size), self.stem, self.clip_start)
        return hashlib.md5(sig.encode()).hexdigest()

    def feat_path(self, path):
        return self.feat_dir / (self.feat_key(path) + '.npy')

    def view(self, clip_start):
        import copy
        v = copy.copy(self)
        v.clip_start = float(clip_start)
        v.base = self.base.view(clip_start)
        return v

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        p = self.feat_path(path)
        if not p.exists():
            raise FileNotFoundError(
                'no cached {} feature for {} (clip_start={}). Run '
                'synccaps_precompute_stem.py first.'.format(
                    self.stem, path, self.clip_start))
        return torch.from_numpy(np.load(p).astype(np.float32)), label

    def get_labels(self):
        return [s[1] for s in self.samples]

    def get_groups(self):
        return list(self.groups)


def make_group_splits(dataset, train_r=0.70, val_r=0.15, seed=42):
    """Leakage-free train/val/test split: group-disjoint AND class-stratified.

    Every UCF-11 source video contributes several near-duplicate clips (same
    actor/background/camera). A clip-level split scatters those across the
    train/test boundary, letting the model memorise appearance and inflating
    test accuracy. StratifiedGroupKFold keeps each source group wholly on one
    side while balancing classes. Test is carved as one fold (~1/k_test of the
    data), then val is carved group-disjointly from the remainder.

    Returns (train_idx, val_idx, test_idx). Asserts group-disjointness so a
    regression can never silently re-introduce the leak.
    """
    labels = np.array(dataset.get_labels())
    groups = np.array(dataset.get_groups())
    n = len(labels)

    test_r = 1.0 - train_r - val_r
    k_test = max(2, round(1.0 / test_r))
    sgkf1 = StratifiedGroupKFold(n_splits=k_test, shuffle=True, random_state=seed)
    tv_idx, test_idx = next(sgkf1.split(np.zeros(n), labels, groups))

    k_val = max(2, round((train_r + val_r) / val_r))
    sgkf2 = StratifiedGroupKFold(n_splits=k_val, shuffle=True, random_state=seed)
    rel_tr, rel_val = next(sgkf2.split(
        np.zeros(len(tv_idx)), labels[tv_idx], groups[tv_idx]))
    train_idx, val_idx = tv_idx[rel_tr], tv_idx[rel_val]

    g_tr, g_va, g_te = set(groups[train_idx]), set(groups[val_idx]), set(groups[test_idx])
    assert not (g_tr & g_te), "group leak: {} groups in BOTH train and test".format(len(g_tr & g_te))
    assert not (g_tr & g_va), "group leak: {} groups in BOTH train and val".format(len(g_tr & g_va))
    assert not (g_va & g_te), "group leak: {} groups in BOTH val and test".format(len(g_va & g_te))
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def make_official_split1(dataset, val_groups=(8, 9, 10), test_groups=tuple(range(1, 8))):
    """UCF101's OFFICIAL split-1, reconstructed from the g<NN> token.

    Published UCF101 numbers use the official train/test lists, which hold out
    groups 1-7 for test and train on 8-25. Our seeded StratifiedGroupKFold is
    equally leak-free but is a DIFFERENT partition, so its accuracies are not
    directly comparable to the literature. This reproduces the official
    partition exactly: holding out g01-g07 yields 9537 train / 3783 test on
    UCF101-full, matching the published counts (verified 2026-08-16).

    The official protocol defines no validation set, but the training loop
    selects a checkpoint on val. We therefore carve val from TRAIN groups
    (default g08-g10, 3 of the 18 train groups); test is never touched. This
    trains on ~83% of the official train set, a small and clearly-stated
    deviation. Pass val_groups=() to train on all 9537 instead.

    Returns (train_idx, val_idx, test_idx).
    """
    groups = np.array(dataset.get_groups())
    gid = []
    for g in groups:
        m = re.search(r'_g(\d+)', str(g))
        gid.append(int(m.group(1)) if m else -1)
    gid = np.array(gid)
    if (gid < 0).any():
        raise ValueError(
            'make_official_split1 needs UCF101-style _g<NN> group ids; {} clips '
            'had none (UCF-11 has no official splits — use '
            'make_stratified_splits)'.format(int((gid < 0).sum())))
    test_set, val_set = set(test_groups), set(val_groups)
    test_idx  = np.where(np.isin(gid, list(test_set)))[0]
    val_idx   = np.where(np.isin(gid, list(val_set)))[0]
    train_idx = np.where(~np.isin(gid, list(test_set | val_set)))[0]

    g_tr, g_va, g_te = set(gid[train_idx]), set(gid[val_idx]), set(gid[test_idx])
    assert not (g_tr & g_te), 'group leak: train/test share {}'.format(g_tr & g_te)
    assert not (g_tr & g_va), 'group leak: train/val share {}'.format(g_tr & g_va)
    assert not (g_va & g_te), 'group leak: val/test share {}'.format(g_va & g_te)
    print('[official split-1] train {} / val {} / test {}  (train+val {} vs '
          'official 9537; test vs official 3783)'.format(
              len(train_idx), len(val_idx), len(test_idx),
              len(train_idx) + len(val_idx)), flush=True)
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


# All 19 experiment scripts call make_stratified_splits(); aliasing here makes
# every re-run group-disjoint with no per-caller edits. The old clip-level
# StratifiedShuffleSplit body (the leak the TCSVT reviewer flagged) is retired.
make_stratified_splits = make_group_splits


print("Dataset class defined (5fps fixed-rate, clean frames, disk cache).")

