# READ: Large-Scale Neural Scene Rendering for Autonomous Driving

This is the code release for our AAAI2023 paper.

A large-scale neural rendering method is proposed to synthesize the autonomous driving scene~(READ), which makes it possible to synthesize large-scale driving scenarios on a PC. Our model can not only synthesize realistic driving scenes but also stitch and edit driving scenes.

Paper: https://arxiv.org/abs/2205.05509

Video: 
 [Bilibili](https://www.bilibili.com/video/BV1KY411w7qh/)  [Youtube](https://youtu.be/W3h5nmmM5BM)  [(Compressed)](https://youtu.be/73zcrqwNFRk)



Demo: (Use only one camera view for training)


<p float="left">
<img src="https://user-images.githubusercontent.com/24960306/168012420-468478de-1db5-430d-bdd2-b52755477cd3.gif" width="270"/>
<img src="https://user-images.githubusercontent.com/24960306/168014170-b964a639-25de-4290-8e91-dc3d3f66ab7c.gif" width="270"/>
<img src="https://user-images.githubusercontent.com/24960306/168012387-ff471fcf-f617-4844-a4d6-bfbf52753d03.gif" width="270"/>
</p>

<p float="center">
  <img src="https://user-images.githubusercontent.com/24960306/168002213-c7c49209-d2bf-489d-9f84-aac2fe6b757b.gif" width="410">
   <img src="https://user-images.githubusercontent.com/24960306/169205523-2d4e051a-2c56-461d-b16a-bb022e2596f2.gif" width="410">
</p>


## Citation

```

@article{li2022read,
  title={READ: Large-Scale Neural Scene Rendering for Autonomous Driving},
  author={Li, Zhuopeng and Li, Lu and Ma, Zeyu and Zhang, Ping and Chen, Junbo and Zhu, Jianke},
  journal={arXiv preprint arXiv:2205.05509},
  year={2022}
}
```

## Overview: 

![contents](./image/main.jpg)




<!--- 
# [![Watch the video](https://i.ytimg.com/an_webp/kC-bwky4e7Q/mqdefault_6s.webp?du=3000&sqp=CIDh7JMG&rs=AOn4CLAE5KzsOlrQzpZVB2DYJbC4UMOhGQ)](https://youtu.be/kC-bwky4e7Q)
[<img src="https://i.ytimg.com/an_webp/kC-bwky4e7Q/mqdefault_6s.webp?du=3000&sqp=CIDh7JMG&rs=AOn4CLAE5KzsOlrQzpZVB2DYJbC4UMOhGQ" width="60%">](https://youtu.be/73zcrqwNFRk)
--> 



## Novel View(Click to view the video):

[![Watch the video](./image/Video.png)](https://youtu.be/W3h5nmmM5BM )

##  Scene Editing:

READ can move and remove the cars in different views. A panorama with larger view can be synthesized by changing the camera parameters.
![contents](./image/Scene_Editing.jpg)
 

## Scene Stitching:

READ is able to synthesize the larger driving scenes and update local areas with obvious changes in road conditions. 
![contents](./image/Scene_Stitching.jpg)

## Novel View Synthesis:

![contents](./image/NovelView.jpg)

