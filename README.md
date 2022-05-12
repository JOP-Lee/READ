# READ-Large-Scale-Neural-Scene-Rendering-for-Autonomous-Driving
implementation of "READ:  Large-Scale Neural Scene Rendering for Autonomous Driving"

Paper: https://arxiv.org/abs/2205.05509

Compressed video: https://www.youtube.com/watch?v=73zcrqwNFRk

## Citation

```
@misc{https://doi.org/10.48550/arxiv.2205.05509,
  doi = {10.48550/ARXIV.2205.05509},
  url = {https://arxiv.org/abs/2205.05509},  
  author = {Li, Zhuopeng and Li, Lu and Ma, Zeyu and Zhang, Ping and Chen, Junbo and Zhu, Jianke},  
  title = {READ: Large-Scale Neural Scene Rendering for Autonomous Driving},  
  publisher = {arXiv},  
  year = {2022},  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

![Turn](https://user-images.githubusercontent.com/24960306/168002213-c7c49209-d2bf-489d-9f84-aac2fe6b757b.gif)


## Overview: 

![contents](./image/main.jpg)




<!--- 
# [![Watch the video](https://i.ytimg.com/an_webp/kC-bwky4e7Q/mqdefault_6s.webp?du=3000&sqp=CIDh7JMG&rs=AOn4CLAE5KzsOlrQzpZVB2DYJbC4UMOhGQ)](https://youtu.be/kC-bwky4e7Q)
[<img src="https://i.ytimg.com/an_webp/kC-bwky4e7Q/mqdefault_6s.webp?du=3000&sqp=CIDh7JMG&rs=AOn4CLAE5KzsOlrQzpZVB2DYJbC4UMOhGQ" width="60%">](https://youtu.be/73zcrqwNFRk)
--> 



## Novel View(Click to view the video):

[![Watch the video](./image/video.png)](https://youtu.be/73zcrqwNFRk)

##  Scene Editing:

We can move and remove the cars in different views. A panorama with larger view can be synthesized by changing the camera parameters.
![contents](./image/SceneEdit.jpg)


## Scene Stitching:

Our model is able to synthesize the larger driving scenes and update local areas with obvious changes in road conditions. 
![contents](./image/Scene_Stitching.jpg)

## Novel View Synthesis:

![contents](./image/NovelView.jpg)

