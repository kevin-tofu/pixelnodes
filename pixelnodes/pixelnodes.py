
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import skimage

from typing import NamedTuple

class Cluster(NamedTuple):
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

    cluster = Cluster(
        ms.labels_,
        ms.cluster_centers_.tolist(),
        members_size,
        weight_pixelvalue,
        image.shape,
        pixelset.shape
    )
    # print(cluster.members_size)
    return cluster


def create_superpixel_image(cluster: Cluster):
    labels = cluster.labels
    cluster_centers = np.array(cluster.centres)
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    pixelset_clustered = np.zeros(cluster.pixelset_shape)
    for cloop, k in enumerate(range(cluster_centers.shape[0])):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        pixelset_clustered[my_members] = np.expand_dims(cluster_center, axis=0)

    pixelset_clustered = pixelset_clustered[:, 0:3].reshape(cluster.image_shape)
    pixelset_clustered = (pixelset_clustered / cluster.weight_pixelvalue).astype(np.uint8)

    return pixelset_clustered


def create_nodes_image(cluster: Cluster):
    pass


if __name__ == '__main__':

    image = skimage.io.imread('./lena.png')
    
    cluster = clustering(image, 0.01, 2000, weight_pixelvalue=0.2)

    image_superpixel = create_superpixel_image(cluster)
    image_nodes = create_superpixel_image(cluster)
    print(pixelset_clustered.shape, pixelset_clustered.dtype)

    skimage.io.imsave('./lena-superpixel.png', image_superpixel)
    # skimage.io.imsave('./lena-nodes.png', image_nodes)

