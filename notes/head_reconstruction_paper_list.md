# Hair
- [2017 - HairNet: Single-View Hair Reconstruction using Convolutional Neural Networks](https://arxiv.org/pdf/1806.07467.pdf)
- [2018 - Two-phase Hair Image Synthesis by Self-Enhancing Generative Model](https://arxiv.org/pdf/1902.11203.pdf)

# Hair and Head
- [2015 - High-Quality Hair Modeling from A Single Portrait Photo](http://www.eecs.harvard.edu/~kalyans/research/portraitrelief/PortraitRelief_SIGA15.pdf)
- [2016 - Semantic 3D Reconstruction of Heads](http://www.eccv2016.org/files/posters/P-4A-20.pdf)
- [2016 - Real-time Facial Animation with Image-based Dynamic Avatars](http://kunzhou.net/2016/imageAvatar.pdf)
- [2016 - Head Reconstruction from Internet Photos](http://grail.cs.washington.edu/projects/liangshu/0351.pdf)
- [2017 - Video to Fully Automatic 3D Hair Model](https://grail.cs.washington.edu/projects/liangshu/hair.pdf)
- [2017 - Avatar Digitization From a Single Image For Real-Time Rendering](http://www.hao-li.com/publications/papers/siggraphAsia2017ADFSIFRTR.pdf)
    - [This method was patented](https://patentscope.wipo.int/search/docs2/pct/WO2019050808/pic/UrFdLTcUtz3JKcyJ46QtvLMkXaNmvfyBorXD2PQDCXitYw2wreFZ3oBRhwWlnf0Fds2z1772vbV7MdXOJAwiYx9LF1gRcYx7dN_YyjJe0ZTw7oAA66RF0TSUPbivsnxNLHjYUzz1Bx1SL5RL38uYAHE48-A0zFm3iSuZRHCiHJU;jsessionid=E4A9C91C2023083168462BC32BC7283C.wapp2nA?docId=id00000047123213&psAuth=ONPpCO2UiiwHU6t0y2DituB3qciMk-BpxUW5k0KmLzA)
- [2019 - Combining 3D Morphable Models: A Large scale Face-and-Head Model](https://arxiv.org/pdf/1903.03785.pdf)

# Face
- [face reconstruction list](https://github.com/YadiraF/face3d/blob/master/3D%20Face%20Papers.md)

# Summary
- 2019 - Combining 3D Morphable Models: A Large scale Face-and-Head Model
    - combine a morphable face model with a morphable head model
        - face model: LSFM
        - head model: LYHM
        - use nonrigid ICP for face-head alignment
    - regression
        - a single image -> face PCA parameters -> head PCA parameters
        - use [3D Face Morphable Models “In-the-Wild”](https://arxiv.org/pdf/1701.05360.pdf) for inference
        from image to face PCA parameters
