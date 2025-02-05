import torch
from transformers import CLIPImageProcessor, CLIPVisionModel

from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import load_image
from .src.diffusers.models.referencenet.referencenet_unet_2d_condition import (
    ReferenceNetModel,
)
from .src.diffusers.models.referencenet.unet_2d_condition import UNet2DConditionModel
from .src.diffusers.pipelines.referencenet.pipeline_referencenet import (
    StableDiffusionReferenceNetPipeline,
)

import face_alignment
from .anonymize_faces_in_image import anonymize_faces_in_image




def anonymize(img_path, save_path):
    
    face_model_id = "hkung/face-anon-simple"
    clip_model_id = "openai/clip-vit-large-patch14"
    sd_model_id = "stabilityai/stable-diffusion-2-1"

    unet = UNet2DConditionModel.from_pretrained(
        face_model_id, subfolder="unet", use_safetensors=True
    )
    referencenet = ReferenceNetModel.from_pretrained(
        face_model_id, subfolder="referencenet", use_safetensors=True
    )
    conditioning_referencenet = ReferenceNetModel.from_pretrained(
        face_model_id, subfolder="conditioning_referencenet", use_safetensors=True
    )
    vae = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae", use_safetensors=True)
    scheduler = DDPMScheduler.from_pretrained(
        sd_model_id, subfolder="scheduler", use_safetensors=True
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(
        clip_model_id, use_safetensors=True
    )
    image_encoder = CLIPVisionModel.from_pretrained(clip_model_id, use_safetensors=True)

    pipe = StableDiffusionReferenceNetPipeline(
        unet=unet,
        referencenet=referencenet,
        conditioning_referencenet=conditioning_referencenet,
        vae=vae,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda")

    generator = torch.manual_seed(1)

    # get an input image for anonymization
    original_image = load_image(img_path)

    # SFD (likely best results, but slower)
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, face_detector="sfd"
    )

    # generate an image that anonymizes faces
    anon_image = anonymize_faces_in_image(
        image=original_image,
        face_alignment=fa,
        pipe=pipe,
        generator=generator,
        face_image_size=512,
        num_inference_steps=25,
        guidance_scale=4.0,
        anonymization_degree=1.25,
    )
    
    anon_image.save(save_path, format="PNG")

    return anon_image 