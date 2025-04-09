import numpy as np
from PIL import Image, ImageFilter
import cv2
from skimage.exposure import match_histograms
from skimage.color import rgb2lab, deltaE_ciede2000


def blur_if_too_sharp(generated, source, threshold=1.4):
    """Blur generated image only if noticeably sharper than source."""
    def sharpness(img):
        return cv2.Laplacian(np.array(img), cv2.CV_64F).var()

    if sharpness(generated) > sharpness(source) * threshold:
        return generated.filter(ImageFilter.GaussianBlur(radius=1.0))
    return generated


def histogram_blend(generated, original, mask, blend_factor=0.7):
    """Histogram match only inside the mask and blend it with original."""
    gen_np = np.array(generated).astype(np.float32)
    orig_np = np.array(original).astype(np.float32)
    mask_np = np.array(mask.convert("L")) > 127

    # Match each channel inside the mask
    for c in range(3):
        matched = match_histograms(
            gen_np[..., c][mask_np],
            orig_np[..., c][mask_np]
        )
        gen_np[..., c][mask_np] = (
            blend_factor * matched +
            (1 - blend_factor) * gen_np[..., c][mask_np]
        )

    # Smooth the edge of the mask for better transition
    blurred_mask = cv2.GaussianBlur(mask_np.astype(np.float32), (25, 25), 0)
    blended = (gen_np * blurred_mask[..., None] +
               orig_np * (1 - blurred_mask[..., None]))

    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def color_consistency_check(img1, img2, mask):
    """Returns the average CIEDE2000 color distance inside the mask."""
    img1 = np.array(img1)
    img2 = np.array(img2)
    mask_np = np.array(mask.convert("L")) > 127
    lab1 = rgb2lab(img1)
    lab2 = rgb2lab(img2)
    diff = deltaE_ciede2000(lab1[mask_np], lab2[mask_np])
    return diff.mean()


def jpeg_simulation(img, quality=30):
    """Simulates compression artifacts using JPEG round-trip."""
    from io import BytesIO
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return Image.open(buffer)


def auto_postprocess(source, generated, mask=None,
                     apply_blur=True,
                     apply_histogram=True,
                     apply_blend=True,
                     apply_jpeg=False):
    """
    Full pipeline: adaptively matches tone, sharpness, and blends.
    """
    out = generated

    if mask is not None and apply_histogram:
        out = histogram_blend(out, source, mask)

    if apply_blur:
        out = blur_if_too_sharp(out, source)

    if apply_jpeg:
        out = jpeg_simulation(out, quality=35)

    if apply_blend and mask is not None:
        out = histogram_blend(out, source, mask, blend_factor=0.5)

    return out
