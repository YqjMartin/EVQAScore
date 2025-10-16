# Usage

## Prepare

Download VATEX-EVAL dataset in the following link

```
https://drive.google.com/drive/folders/1jAfZZKEgkMEYFF2x1mhYo39nH-TNeGm6?usp=sharing
```

Download YOLO model checkpoint yolo11x-seg in the following link

```
https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt
```

Download PAC-S++ clip model checkpoint PAC++_clip_ViT-L-14 in the following link

```
https://ailb-web.ing.unimore.it/publicfiles/pac++/PAC++_clip_ViT-L-14.pth
```

Download corresponding clip model code following instructions under

```
https://github.com/aimagelab/pacscore
```

## Compute EVQAScore

First, run `extract.py` to get keywords of all candidates.

```
python extract.py
```

Then, extract video features in parallel by running

```
python evqascore.py --preprocess --interval 30 --num-chunks 8 --chunk-idx 0 --run-name xxx
```

Finally, get EVQAScore of vatex-eval dataset by running

```
python evqascore.py --interval 30 --num-chunks 8 --run-name xxx
```

The results will store in the result folder as a json file.
