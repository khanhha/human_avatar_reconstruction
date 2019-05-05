# purpose
find out how three camera properties: camera distance, focal length and camere height affect the front and side Silhouettes

# observation
- focal lengths don't have affect on normalized silhouette

# different camera distances
configuration
 - mesh height: 165
 - camera distance range: [300, 700]
 - camera
 ![](./images/cam_dif_distance.jpg)
 - real silhouettes
 ![](./images/cam_dif_distance_sil.jpg)
 - normalized silhouette heat map
 ![](./images/cam_dif_distance_sil_normalized.jpg)

# different camera heights
configuration
  - mesh height: 165
  - camera height range: [70, 150]
  - camera
  ![](./images/cam_dif_height.jpg)
  - real silhouettes
  ![](./images/cam_dif_height_sil.jpg)
  - normalized silhouette heatmap
  ![](./images/cam_dif_height_sil_normalized.jpg)

# different focal lengths
configuration
  - mesh height: 165
  - focal length range: [2.5mm, 5.0mm]
  - real silhouettes
  ![](./images/cam_dif_focal_len_sil.jpg)
  - normalized silhouette heat map
  ![](./images/cam_dif_focal_len_sil_normalized.jpg)
