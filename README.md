# pixelnodes

 This package converts an image into Graph-Representation.
You can check the graph using superpixel or node representation.

## Installation

```bash

poetry add git+https://github.com/kevin-tofu/pixelnodes.git

```

## Example

```python

import pixelnodes
image = skimage.io.imread('./lena.png')
    
cluster = pixelnodes.clustering(image, 0.01, 2000, weight_pixelvalue=0.2)

image_superpixel = pixelnodes.create_superpixel_image(cluster)
image_nodes = pixelnodes.create_nodes_image(cluster)

skimage.io.imsave('./lena-superpixel.png', image_superpixel)

```
