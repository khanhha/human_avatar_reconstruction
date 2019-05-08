# table of content
- [experiment purpose](#experiment-purpose)
- [experiment procedure](#experiment-procedure)
- [effect of camera distance on silhouette](#effect-of-camera-distance-on-silhouette)
- [effect of camera height on silhouette](#effect-of-camera-height-on-silhouette)
- [effect of camera focal length on silhouette](#effect-of-camera-focal-length-on-silhouette)

# experiment purpose
find out how three camera properties: camera distance, focal length and camera height affect the front and side Silhouettes

# experiment procedure
- use Blender change camera properties (distance, height, focal length) and project the corresponding Silhouettes

- normalize silhouettes to the same height

- draw all silhouettes over each other and calculate the heat map  

# effect of camera distance on silhouette
configuration
 - mesh height: 165
 - camera distance range: [300, 700]
 - camera positions at two boundary distance values
 ![](./images/cam_dif_distance.jpg)
 - original silhouette taken at two boundary distance values
 ![](./images/cam_dif_distance_sil.jpg)
 - normalized silhouette heat map
 ![](./images/cam_dif_distance_sil_normalized.jpg)

# effect of camera height on silhouette
configuration
  - mesh height: 165
  - camera height range: [70, 150]

  - camera positions at two boundary height values
  ![](./images/cam_dif_height.jpg)

  - original silhouettes at two boundary height values
  ![](./images/cam_dif_height_sil.jpg)

  - normalized silhouette heat map
  ![](./images/cam_dif_height_sil_normalized.jpg)

# effect of camera focal length on silhouette
configuration
  - mesh height: 165
  - focal length range: [2.5mm, 5.0mm]
  - original silhouettes at two boundary focal length values
  ![](./images/cam_dif_focal_len_sil.jpg)
  - normalized silhouette heat map
  ![](./images/cam_dif_focal_len_sil_normalized.jpg)
