import numpy as np
from numba import njit, prange

from enum import Enum, auto


class Manipulation(Enum):
    """
    .CROP - Crop an image to a specified rectangular region.
    -
    Parameters
    img : np.ndarray
        Input image.
    coords : tuple[int, int, int, int]
        Cropping coordinates in the form (x_min, x_max, y_min, y_max).


    .ROTATE - Rotate an image by a given angle.
    -
    Parameters
    img : np.ndarray
        Input image.
    angle : float
        Rotation angle in degrees.

    Notes
    The algorithm uses inverse mapping with bilinear interpolation.


    .BLUR - Apply box blur to an image.
    -
    Parameters
    img : np.ndarray
        Input image.

    Notes
    Box blur applies a uniform averaging filter over a square neighborhood.


    .MIRROR - Mirror an image along a specified axis.
    -
    Parameters
    img : np.ndarray
        Input image.
    axis : int
        Axis along which the image is mirrored
        (e.g., 0 for vertical, 1 for horizontal).


    .COMPRESSION - Resize an image using area-based downsampling.
    -
    Parameters
    img : np.ndarray
        Input image.
    size : int
        Target size of the output image.

    Notes
    The algorithm uses area-based downsampling (box averaging).

    .PERMUTATION Permute square blocks of an image.
    -
    Parameters
    img : np.ndarray
        Input image.
    size : int
        Edge length (in pixels) of square blocks used for permutation.


    .INVERT -Invert image intensities.
    -
    Parameters
    img : np.ndarray
        Input image.
    channel : int
        Specify channel to invert (
            0 - RED
            1 - GREEN
            2 - BLUE
            )
    """
    CROP = auto()
    ROTATE = auto()
    BLUR = auto()
    MIRROR = auto()
    COMPRESSION = auto()
    PERMUTATION = auto()
    INVERT = auto()


@njit(fastmath=True)
def _rotate_numba(src, angle):
    h, w, c = src.shape
    theta = np.deg2rad(angle)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    nh = int(np.ceil(abs(h * cos_t) + abs(w * sin_t)))
    nw = int(np.ceil(abs(h * sin_t) + abs(w * cos_t)))

    dst = np.zeros((nh, nw, c), dtype=src.dtype)

    cx = w / 2.0
    cy = h / 2.0
    ncx = nw / 2.0
    ncy = nh / 2.0

    for y in range(nh):
        for x in range(nw):

            # inverse mapping
            xd = x - ncx
            yd = y - ncy

            xs = xd * cos_t + yd * sin_t
            ys = -xd * sin_t + yd * cos_t

            x_src = xs + cx
            y_src = ys + cy

            x0 = int(np.floor(x_src))
            y0 = int(np.floor(y_src))
            x1 = x0 + 1
            y1 = y0 + 1

            if x0 < 0 or x1 >= w or y0 < 0 or y1 >= h:
                for ch in range(c):
                    dst[y, x, ch] = 128
                continue

            dx = x_src - x0
            dy = y_src - y0

            w00 = (1 - dx) * (1 - dy)
            w01 = dx * (1 - dy)
            w10 = (1 - dx) * dy
            w11 = dx * dy

            for ch in range(c):
                dst[y, x, ch] = (
                    w00 * src[y0, x0, ch] +
                    w01 * src[y0, x1, ch] +
                    w10 * src[y1, x0, ch] +
                    w11 * src[y1, x1, ch]
                )

    return dst


@njit(parallel=True, fastmath=True)
def _blur_numba(img, w, count):
    pad = len(w) - 1
    H, W, C = img.shape
    out = np.zeros_like(img, dtype=np.float32)

    for y in prange(pad, H - pad):
        for x in range(pad, W - pad):

            for c in range(C):
                acc = 0.0

                for r in range(pad + 1):

                    k = w[r] / count[r]

                    if r == 0:
                        acc += img[y, x, c] * k
                    else:
                        # верх и низ
                        for j in range(-r, r + 1):
                            acc += img[y - r, x + j, c] * k
                            acc += img[y + r, x + j, c] * k

                        # лево и право (без углов)
                        for i in range(-r + 1, r):
                            acc += img[y + i, x - r, c] * k
                            acc += img[y + i, x + r, c] * k

                out[y, x, c] = acc

    return out


class ImageManipulation:
    """
    Collection of fundamental image manipulation operations.

    This class provides a unified interface for common image processing
    transformations such as cropping, rotation, blurring, resizing,
    padding, permutation, and intensity inversion.

    All operations are designed to work on NumPy arrays representing
    decoded images in memory (H × W × C).

    Notes
    -----
    - Images are assumed to be fully decoded and stored in RAM.
    - Local operations (e.g., blur, interpolation) require explicit
      boundary conditions, which should be defined via padding.
    - Padding is treated as a separate, explicit step and is not
      implicitly applied by other methods.
    - Unless stated otherwise, operations do not modify the input
      image in-place and return a new array.

    The `apply` method provides a dispatch mechanism for invoking
    operations using a predefined manipulation enum.
    """

    def crop(self, img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """
        Crop an image to a specified rectangular region.

        Parameters
        ----------
        img : np.ndarray
            Input image.
        coords : tuple[int, int, int, int]
            Cropping coordinates in the form (y_min, x_min, y_max, x_max).
        """
        h, w, _ = img.shape
        y0, x0, y1, x1 = bbox
        if not (0 <= y0 < y1 <= h) or not (0 <= x0 < x1 <= w):
            raise ValueError("Невалидные координаты bbox. См. документацию.")

        return img[y0: y1, x0: x1]

    def rotate(self, src: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate an image by a given angle.

        Parameters
        ----------
        img : np.ndarray
            Input image.
        angle : float
            Rotation angle in degrees.

        Notes
        -----
        The algorithm uses inverse mapping with bilinear interpolation.
        """
        return _rotate_numba(src, angle).astype(np.uint8)

    def blur(self, img: np.ndarray) -> np.ndarray:
        """
        Apply box blur to an image.

        Parameters
        ----------
        img : np.ndarray
            Input image.

        Notes
        -----
        Box blur applies a uniform averaging filter over a square neighborhood.
        """
        pad = 21
        sigma = 8.0   # можно менять для силы blur

        # -----------------------
        # 2. Gaussian веса по радиусу
        # -----------------------

        r = np.arange(0, pad + 1, dtype=np.float32)
        w = np.exp(-(r**2) / (2 * sigma**2)).astype(np.float32)
        w /= w.sum()   # нормализация

        # размер каждого кольца L∞
        count = np.zeros(pad + 1, dtype=np.float32)
        count[0] = 1.0
        for i in range(1, pad + 1):
            count[i] = 8.0 * i

        # -----------------------
        # 3. Padding
        # -----------------------

        padded = np.pad(
            img.astype(np.float32),
            pad_width=((pad, pad), (pad, pad), (0, 0)),
            mode="reflect"
        )

        # -----------------------
        # 4. Запуск
        # -----------------------

        nout = _blur_numba(padded, w, count)

        # обрезаем padding
        nout = nout[pad:-pad, pad:-pad]

        return np.clip(nout, 0, 255).astype(np.uint8)

    def mirror(self, img: np.ndarray, axis: int) -> np.ndarray:
        """
        Mirror an image along a specified axis.

        Parameters
        ----------
        img : np.ndarray
            Input image.
        axis : int
            Axis along which the image is mirrored
            (e.g., 0 for vertical, 1 for horizontal).
        """
        if axis != 0 and axis != 1:
            raise ValueError("Неверное значение axis. См документацию.")

        return img[:, ::-1] if axis else img[::-1, :]

    def compression(self, img: np.ndarray, size: int) -> np.ndarray:
        """
        Resize an image using area-based downsampling.

        Parameters
        ----------
        img : np.ndarray
            Input image.
        size : int
            Target size of the output image.

        Notes
        -----
        The algorithm uses area-based downsampling (box averaging).
        """
        h, w, c = img.shape

        pad_h = int(np.ceil(h / size) * size - h)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_w = int(np.ceil(w / size) * size - w)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padded = np.pad(img, pad_width=((pad_top, pad_bottom),
                                        (pad_left, pad_right),
                                        (0,0)), mode="edge")

        kh, kw = padded.shape[0] // size, padded.shape[1] // size

        padded = (
            np.reshape(padded, (size, kh, size, kw, c))
            .transpose(0, 2, 1, 3, 4)
            .mean(axis=(2, 3))
        )

        return np.clip(padded, 0, 255).astype(np.uint8)

    def permutation(self, img: np.ndarray, size: int) -> np.ndarray:
        """
        Permute square blocks of an image.

        Parameters
        ----------
        img : np.ndarray
            Input image.
        size : int
            Edge length (in pixels) of square blocks used for permutation.
        """
        h, w, c = img.shape

        pad_h = int(np.ceil(h / size) * size - h)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_w = int(np.ceil(w / size) * size - w)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padded = np.pad(img, pad_width=((pad_top, pad_bottom),
                                        (pad_left, pad_right),
                                        (0,0)), mode="edge")

        nh, nw = padded.shape[0] // size, padded.shape[1] // size

        perm = np.random.permutation(nh * nw)

        shuffled = (
            padded
            .reshape(nh, size, nw, size, c)
            .transpose(0, 2, 1, 3, 4)
            .reshape(-1, size, size, c)
            [perm]
            .reshape(nh, nw, size, size, c)
            .transpose(0, 2, 1, 3, 4)
            .reshape(nh * size, nw * size, c)
        )

        return np.clip(shuffled, 0, 255).astype(np.uint8)

    def invert(self, img: np.ndarray) -> np.ndarray:
        """
        Invert image intensities.

        Parameters
        ----------
        img : np.ndarray
            Input image.
        """
        return np.array([255, 255, 255]) - img[:, :]

    def apply(self, op: Manipulation, img, **kwargs):
        dispatch = {
            Manipulation.CROP: self.crop,
            Manipulation.ROTATE: self.rotate,
            Manipulation.BLUR: self.blur,
            Manipulation.MIRROR: self.mirror,
            Manipulation.COMPRESSION: self.compression,
            Manipulation.PERMUTATION: self.permutation,
            Manipulation.INVERT: self.invert
        }
        return dispatch[op](img, **kwargs)
