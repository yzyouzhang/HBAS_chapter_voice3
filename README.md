# HBAS_chapter_voice3


```angular2html
python3 train.py -o /data3/neil/hbas/models1101/resnet_ocsoftmax_8888 -l ocsoftmax --gpu 3 -m resnet --seed 8888
```

```angular2html
python3 test.py -t ASVspoof2019LA -m /data3/neil/hbas/models1101/resnet_ocsoftmax_8888 -l ocsoftmax --gpu 3
```