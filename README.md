# pixelnodes
 This package converts an image into Graph-Representation.
You can check the graph using superpixel or node representation.

## Installation
```bash

poetry add 

```

## Example
```python

image = skimage.io.imread('./lena.png')
    
pixelset = get_pixelset(image).reshape((-1, 5))
cluster = clustering(pixelset, 0.01, 2000, weight_pixelvalue=0.2)


pixelset_clustered = create_image(
    cluster,
    pixelset.shape,
    image.shape
)

skimage.io.imsave('./lena-out.png', pixelset_clustered)

```
