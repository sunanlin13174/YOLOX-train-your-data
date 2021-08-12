# YOLOX train your data
you need generate data.txt like follow format **(per line-> one image)**.
## prepare data.txt like this:<br>
<br>img_path1 x1,y1,x2,y2,class_id x1,y1,x2,y2,class_id2 ........**(per line-> one image)** <br>
<br>img_path2 x1,y1,x2,y2,class_id <br>
<br>************************************************************<br>
### note:<br>
**<br>x1,y1,x2,y2 is int type and it belong to 0~img__w ,0~img__h, not 0~1 !!!<br>
<br>img_path is abs path ;must be careful the sign " " and "," in data.txt, there was an example: <br>
<br>/home/sal/images/000010.jpg 0,190,466,516,1<br>
<br>/home/sal/images/000011.jpg 284,548,458,851,7 256,393,369,608,1<br>**
 ## Train
 **i.step1** , before train,you need change yolox/exp/yolox_base.py follow you need, i add some explain in it. **such as change data.txt path in it.** <br>
**ii.step2** , change train.py params, just as https://github.com/Megvii-BaseDetection/YOLOX.git ,when you have changed , just run : **python train.py**

 **iii. star**
