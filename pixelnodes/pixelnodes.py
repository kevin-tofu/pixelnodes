
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import skimage

from typing import NamedTuple

class Nodes(NamedTuple):
    labels: list[int]
    centres: list[list[float]]
    members_size: list[int]
    weight_pixelvalue: float
    image_shape: tuple
    pixelset_shape: tuple


def get_pixelset(image: np.ndarray):
    
    # image = np.zeros((512, 512, 3))
    width = image.shape[1]
    height = image.shape[0]

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    img_data = np.stack((x, y), axis=2)

    ret = np.concatenate([image, img_data], axis = 2) # RGBXY
    return ret


def clustering(
    # pixelset: np.ndarray,
    image: np.ndarray,
    quantile: float,
    n_samples: int,
    weight_pixelvalue: float = 1.0
):
    # pixelset = get_pixelset(image)
    pixelset = get_pixelset(image).reshape((-1, 5))

    # (N, 5 (RGBXY)) or (N, 3 (GXY))
    range_weight = 3 if pixelset.shape[1] == 5 else 1
    pixelset[:, 0:range_weight] = pixelset[:, 0:range_weight] * weight_pixelvalue
    bandwidth = estimate_bandwidth(pixelset, quantile=quantile, n_samples=n_samples)
    # print('bandwidth:', bandwidth)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(pixelset)
    
    members_size = [ len(ms.labels_ == k) for k in range(ms.cluster_centers_.shape[0])]

    nodes = Nodes(
        ms.labels_,
        ms.cluster_centers_.tolist(),
        members_size,
        weight_pixelvalue,
        image.shape,
        pixelset.shape
    )
    # print(nodes.members_size)
    return nodes


def create_superpixel_image(nodes: Nodes):
    labels = nodes.labels
    cluster_centers = np.array(nodes.centres)
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    pixelset_clustered = np.zeros(nodes.pixelset_shape)
    for cloop, k in enumerate(range(cluster_centers.shape[0])):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        pixelset_clustered[my_members] = np.expand_dims(cluster_center, axis=0)

    pixelset_clustered = pixelset_clustered[:, 0:3].reshape(nodes.image_shape)
    pixelset_clustered = (pixelset_clustered / nodes.weight_pixelvalue)
    return pixelset_clustered.astype(np.uint8)


def create_nodes_image(nodes: Nodes, nodes_radius: int=10):
    
    from skimage.draw import disk

    ret = np.zeros(nodes.image_shape)
    channels = len(nodes.centres[0]) - 2
    maxsize = max(nodes.members_size)
    max_radius = int(ret.shape[0] / 20)
    # get_radius = lambda a, b=max_radius, c=maxsize: int(b * a / c)

    for (c, mem_size) in zip(nodes.centres, nodes.members_size):
        # print(get_radius(mem_size))
        # rr, cc = disk((int(c[-1]), int(c[-2])), get_radius(mem_size))
        rr, cc = disk((int(c[-1]), int(c[-2])), nodes_radius)
        rr = np.clip(rr, 0, ret.shape[0] - 1)
        cc = np.clip(cc, 0, ret.shape[1] - 1)

        ret[rr, cc] = np.array(c[0:channels], dtype=np.uint8) / nodes.weight_pixelvalue

    return ret.astype(np.uint8)


if __name__ == '__main__':

    from skimage.transform import resize
    from skimage.util import img_as_ubyte

    image = skimage.io.imread('./lena.png')
    image_resize = img_as_ubyte(resize(image, (256, 256)))
    skimage.io.imsave('./lena-resize.png', image_resize)
    
    
    # nodes = clustering(image, 0.01, 2000, weight_pixelvalue=10)
    nodes = clustering(image_resize, 0.01, 200, weight_pixelvalue=0.03)
    image_superpixel = create_superpixel_image(nodes)
    image_nodes = create_nodes_image(nodes)

    skimage.io.imsave('./lena-superpixel.png', image_superpixel)
    skimage.io.imsave('./lena-nodes.png', image_nodes)

    # print(np.array(nodes.centres))
    # print(image_superpixel)
    print(np.array(image_resize).flatten().shape)
    print(np.array(nodes.centres).flatten().shape)
    
