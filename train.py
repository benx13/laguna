import pandas as pd, numpy as np
import sys,os,shutil,gc,re,json,glob,math,time,random,warnings,logging

from tqdm import tqdm
from sklearn.model_selection import KFold,StratifiedKFold,GroupKFold,StratifiedGroupKFold
import sklearn.metrics as skm
from sklearn import preprocessing
import torch
from torch import nn
import torch.nn.functional as F
import cv2

from mmengine import Config
from mmengine.runner import Runner
import mmdet

N_SPLITS = 5
RANDOM_STATE = 41
FOLD=0

def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
fix_seed(RANDOM_STATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DIR_DATA = 'data'
CLASSES = {
    'PNN': 0, 'MO': 1, 'MM': 2, 'LyB': 3, 'LGL': 4, 'Thromb': 5, 'LLC': 6, 
    'LAM3': 7, 'EO': 8, 'LY': 9, 'BA': 10, 'MoB': 11, 'LM': 12, 'LH_lyAct': 13,
    'Lysee': 14, 'Er': 15, 'LF': 16, 'LZMG': 17, 'MBL': 18, 'SS': 19, 'PM': 20,
    'B': 21, 'M': 22
}
df = pd.read_csv(f'{DIR_DATA}/train.csv')
df['path'] = f'{DIR_DATA}/images/'+df['NAME']
df['category_id'] = df['class'].map(CLASSES)

def split_data(df):
    print("\nPerforming StratifiedGroupKFold split...")
    gkf = StratifiedGroupKFold(n_splits=N_SPLITS,shuffle=True,random_state=RANDOM_STATE)
    df['fold'] = -1
    for fold_id, (train_index, test_index) in enumerate(gkf.split(df,y=df['class'],groups=df.Image_ID)):
        df.loc[test_index,'fold'] = fold_id
        print(f"Fold {fold_id}: {len(test_index)} validation samples")
    return df

def get_splits(df,fold):
    df_trn = df[df.fold!=fold].copy()
    df_val = df[df.fold==fold].copy()

    return df_trn,df_val

df = split_data(df)

df['split'] = 'train'
df.loc[df.fold==FOLD,'split'] = 'val'
print(f"\nSplit distribution:")
print(df.split.value_counts())
print("-"*50)



def gen_anno(meta,split):
    print(f"\nGenerating COCO annotations for {split} split...")
    annotations = []
    images = []
    anno_id = 0
    image_id = 0

    meta=meta.copy()
    n_skipped = 0
    for file_name,d in tqdm(meta.groupby('NAME')):
        path = d.path.values[0]
        im = cv2.imread(path)
        height,width,_ = im.shape
        for _,row in d.iterrows():
            x0, y0, x1, y1 = row.x1, row.y1, row.x2, row.y2
            w = x1 - x0
            h = y1 - y0
            bbox = np.array([x0, y0, w, h]).tolist()
            area = w * h
            if area<1:
                n_skipped += 1
                continue

            anno = dict(
                        image_id = image_id,
                        id = anno_id,
                        category_id = row.category_id,
                        bbox = bbox,
                        area = area,
                        iscrowd = 0
                    )
            anno_id += 1
            annotations.append(anno)

        images.append(dict(id=image_id, file_name=file_name,height=height,width=width))
        image_id += 1

    print(f"Skipped {n_skipped} annotations with area < 1")
    print(f"Total annotations: {len(annotations)}")
    print(f"Total images: {len(images)}")
    print("-"*50)

    categories = [dict(id=id, name=name) for name, id in CLASSES.items()]
    coco_json = dict(images=images, annotations=annotations, categories=categories)
    path = f'data/{split}.json'
    print(f'written {len(annotations)} annotations for {len(images)} images to {path}')
    with open(path,'w', encoding='utf-8') as f:
        json.dump(coco_json,f,ensure_ascii=False)


gen_anno(df[df.split=='val'],'val')
### gen_anno(df[df.split=='train'],'train')
# training on complete dataset
gen_anno(df,'train')


cfg_path = 'configs/ddq/ddq-detr-4scale_swinl_8xb2-30e_coco.py'
cfg = Config.fromfile(cfg_path)

load_from = 'checkpoints/ddq_detr_swinl_30e.pth'

data_dir = 'data'
img_prefix = 'images'
max_epochs = 30
val_interval = 1

bs = 2
cfg.train_num_workers = num_workers = 8
cfg.train_dataloader.batch_size = bs

cfg.data_root = data_dir
cfg.work_dir = './output'
metainfo = {
    'classes': tuple(CLASSES.keys()),
    'palette': [
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
        for _ in range(len(CLASSES))
    ]
}
cfg.dataset_type = 'CocoDataset'

num_classes = 23

cfg.num_classes = num_classes
cfg.model.bbox_head.num_classes = num_classes

cfg.train_pipeline =[
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(prob=0.5, type='RandomFlip'),

    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (384, 384),
                        (416, 416),
                        (448, 448),
                        (480, 480),
                        (512, 512),
                    ],
                    type='RandomChoiceResize'),
            ],
            [
                dict(
                    keep_ratio=True,
                    scales=[(352, 352), (384, 384), (416, 416)],
                    type='RandomChoiceResize'),
                dict(
                    allow_negative_crop=True,
                    crop_size=(320, 320),
                    crop_type='absolute_range',
                    type='RandomCrop'),
                dict(
                    keep_ratio=True,
                    scales=[(384, 384), (416, 416), (448, 448)],
                    type='RandomChoiceResize'),
            ],
        ],
        type='RandomChoice'),

    dict(type='YOLOXHSVRandomAug'),
    dict(type='Sharpness',prob=0.5),
    dict(type='AutoContrast',prob=0.5,min_mag=0.1,max_mag=1.9,level=10),
    dict(type='Rotate', level=10, min_mag=180.,max_mag=180.,prob=0.5),

    dict(type='PackDetInputs'),
    ]


cfg.train_dataloader.dataset=dict(
        data_root=cfg.data_root,
        metainfo=metainfo,
        ann_file='train.json',
        backend_args=None,
        data_prefix=dict(img=img_prefix),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=cfg.train_pipeline,

        type='CocoDataset')

cfg.train_dataloader.num_workers = cfg.val_dataloader.num_workers = num_workers
cfg.val_dataloader.dataset=dict(
        data_root=cfg.data_root,
        metainfo=metainfo,
        data_prefix=dict(img=img_prefix),
        ann_file='val.json',
        pipeline = cfg.test_pipeline,
        test_mode=True,
    type='CocoDataset')

cfg.test_dataloader = cfg.val_dataloader

cfg.load_from = load_from
cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=val_interval)
cfg.visualizer = dict( name='visualizer',type='DetLocalVisualizer',vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])
cfg.val_evaluator = cfg.test_evaluator = dict(
    ann_file='data/val.json',
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
    ],
    type='CocoMetric')


fix_seed(RANDOM_STATE)
runner = Runner.from_cfg(cfg)
runner.train()
