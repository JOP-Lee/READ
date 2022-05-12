# READ-Large-Scale-Neural-Scene-Rendering-for-Autonomous-Driving
"READ:  Large-Scale Neural Scene Rendering for Autonomous Driving".  

A large-scale neural rendering method is proposed to synthesize the autonomous driving scene~(READ), which makes it possible to synthesize large-scale driving scenarios on a PC. Our model can not only synthesize realistic driving scenes but also stitch and edit driving scenes.

Paper: https://arxiv.org/abs/2205.05509

Video: https://www.youtube.com/watch?v=73zcrqwNFRk (Compressed)

https://youtu.be/W3h5nmmM5BM (HD)

Demo: (Use only one camera view for training)
<p float="center">
<img src="https://user-images.githubusercontent.com/24960306/168012420-468478de-1db5-430d-bdd2-b52755477cd3.gif" width="32%">
<img src="https://user-images.githubusercontent.com/24960306/168014170-b964a639-25de-4290-8e91-dc3d3f66ab7c.gif" width="32%">
<img src="https://user-images.githubusercontent.com/24960306/168012387-ff471fcf-f617-4844-a4d6-bfbf52753d03.gif" width="32%">
</p>
<p float="center">
  <img src="https://user-images.githubusercontent.com/24960306/168002213-c7c49209-d2bf-489d-9f84-aac2fe6b757b.gif" width="96%">
</p>

## Citation

```

@misc{li2022read,
    title={READ: Large-Scale Neural Scene Rendering for Autonomous Driving},
    author={Zhuopeng Li and Lu Li and Zeyu Ma and Ping Zhang and Junbo Chen and Jianke Zhu},
    year={2022},
    eprint={2205.05509},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Overview: 

![contents](./image/main.jpg)




<!--- 
# [![Watch the video](https://i.ytimg.com/an_webp/kC-bwky4e7Q/mqdefault_6s.webp?du=3000&sqp=CIDh7JMG&rs=AOn4CLAE5KzsOlrQzpZVB2DYJbC4UMOhGQ)](https://youtu.be/kC-bwky4e7Q)
[<img src="https://i.ytimg.com/an_webp/kC-bwky4e7Q/mqdefault_6s.webp?du=3000&sqp=CIDh7JMG&rs=AOn4CLAE5KzsOlrQzpZVB2DYJbC4UMOhGQ" width="60%">](https://youtu.be/73zcrqwNFRk)
--> 



## Novel View(Click to view the video):

[![Watch the video](./image/video.png)](https://youtu.be/73zcrqwNFRk)

##  Scene Editing:

READ can move and remove the cars in different views. A panorama with larger view can be synthesized by changing the camera parameters.
![contents](./image/Scene_Editing.jpg)
 

## Scene Stitching:

READ is able to synthesize the larger driving scenes and update local areas with obvious changes in road conditions. 
![contents](./image/Scene_Stitching.jpg)

## Novel View Synthesis:

![contents](./image/NovelView.jpg)

