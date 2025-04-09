import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, AutoencoderKL
from controlnet_aux import LineartDetector, OpenposeDetector
from compel import Compel
import argparse

from deepface import DeepFace

MODEL_NAME = "SG161222/Realistic_Vision_V5.0_noVAE"
# MODEL_NAME = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
MODEL_CACHE = "cache"
VAE_CACHE = "vae-cache"

from PIL import ImageFilter

################################################################################
# Anonimization
################################################################################


class Predictor:
    def __init__(self, load= True):
        """Initialize and load models."""
        if load:
            self.setup()

    def setup(self):

        # # 1) Load multiple ControlNets in float16
        # controlnet_inpaint = ControlNetModel.from_pretrained(
        #     "lllyasviel/control_v11p_sd15_inpaint",
        #     torch_dtype=torch.float16
        # )
        # lineart_controlnet = ControlNetModel.from_pretrained(
        #     "ControlNet-1-1-preview/control_v11p_sd15_lineart",
        #     torch_dtype=torch.float16
        # )
        # openpose_controlnet = ControlNetModel.from_pretrained(
        #     "lllyasviel/control_v11p_sd15_openpose",
        #     torch_dtype=torch.float16
        # )
        
        controlnet_inpaint = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint",
            torch_dtype=torch.float16
        )
        lineart_controlnet = ControlNetModel.from_pretrained(
            "ControlNet-1-1-preview/control_v11p_sd15_lineart",
            torch_dtype=torch.float16
        )
        openpose_controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose",
            torch_dtype=torch.float16
        )

        self.lineart_processor = LineartDetector.from_pretrained("lllyasviel/Annotators")
        self.openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")

        controlnets = [controlnet_inpaint, lineart_controlnet, openpose_controlnet]

        # 2) Load VAE in float16
        # vae = AutoencoderKL.from_single_file(
        #     "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors",
        #     dtype=torch.float16,
        #     cache_dir=VAE_CACHE
        # )
        
        # Replace the VAE loading code in the setup() method:

        # 2) Load VAE in float16
        vae = AutoencoderKL.from_single_file(
            "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors",
            cache_dir=VAE_CACHE
        )
        vae = vae.to(dtype=torch.float16)  # Manually cast to float16 after loading
        # print(vae.encoder.conv_in.weight.dtype)  # Should be torch.float16
        # print(vae.encoder.conv_in.bias.dtype)    # Should be torch.float16

        # 3) Create the pipeline, float16
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            MODEL_NAME,
            controlnet=controlnets,
            torch_dtype=torch.float16,
            vae=vae
        )
        
        
        #will be deleted
        from diffusers import DPMSolverMultistepScheduler
        #pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config) #trying for the first time
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,  # Better for low-step counts
            algorithm_type="sde-dpmsolver++"  # Improved texture
        )

        self.pipe = pipe.to("cuda")
        self.pipe.enable_model_cpu_offload()

        # 5) (Optional) enable xFormers memory-efficient attention
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[Predictor.setup()] xFormers memory-efficient attention enabled.")
        except Exception as e:
            print(f"[Predictor.setup()] xFormers not available or error enabling it: {e}")

        # NO CPU offload here:
        # pipe.enable_model_cpu_offload()  # <- REMOVED for multi-GPU usage

        # For prompt processing
        self.compel_proc = Compel(
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder
        )
        self.pipe = pipe


    def resize_image(self, image, max_width, max_height):
        """
        Resize an image to a specific height/width while maintaining aspect ratio.
        """
        original_width, original_height = image.size
        width_ratio = max_width / original_width
        height_ratio = max_height / original_height
        resize_ratio = min(width_ratio, height_ratio)

        new_width = int(original_width * resize_ratio)
        new_height = int(original_height * resize_ratio)
        return image.resize((new_width, new_height), Image.LANCZOS)

    def make_inpaint_condition(self, image, image_mask):
        """
        Convert image + mask into inpainting condition (pixels replaced with -1 where masked).
        """
        image = np.array(image.convert("RGB")).astype(np.float16) / 255.0
        image_mask = np.array(image_mask.convert("L")).astype(np.float16) / 255.0

        assert image.shape[:2] == image_mask.shape[:2], "Image and mask must have the same size."
        image[image_mask > 0.5] = -1.0  # Mark masked pixels
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image

    def closest_multiple_of_8(self, width, height):
        """
        Rounds width/height up to the closest multiple of 8.
        """
        w = ((width + 7) // 8) * 8
        h = ((height + 7) // 8) * 8
        return w, h
    
    def age_mapping(self, age: int) -> str:
        """
        Converts a numeric age to a rough age-group label.
        Customize the thresholds and labels as needed.
        """
        
            # Convert age to int if it's a string
        if isinstance(age, str):
            try:
                age = int(float(age))  # Handles cases like "32.5"
            except (ValueError, TypeError):
                age = 30  # Default fallback
            
        if age < 1:
            return "newborn baby"
        elif 1 <= age < 4:
            return "toddler"  
        elif 4 <= age < 10:
            return "young child"
        elif 10 <= age < 14:
            return "pre-teen"
        elif 14 <= age < 18:
            return "teenager"
        elif 18 <= age < 25:
            return "young adult" 
        elif 25 <= age < 35:
            return "adult"
        elif 35 <= age < 50:
            return "middle-aged adult"
        elif 50 <= age < 65:
            return "mature adult"
        elif 65 <= age < 80:
            return "senior"
        else:
            return "elderly" 
        
    def parse_prompt(self, prompt: str):
        # Remove the prefix and split out the emotion
        description, emotion_part = prompt.replace("A photorealistic portrait of a ", "").split(", with a ")
        emotion = emotion_part.replace(" expression", "").strip()
        
        # Possible two-word age groups
        multiword_ages = {"young adult", "middle-aged adult", "elderly adult"}

        # Split the description and see if first two words form a known age group
        words = description.split()
        maybe_two_words = " ".join(words[:2])
        if maybe_two_words in multiword_ages:
            age_group = maybe_two_words
            race = words[2]
            gender = words[3].rstrip(",")
        else:
            age_group = words[0]
            race = words[1]
            gender = words[2].rstrip(",")

        return age_group, race, gender, emotion


    def create_prompt(self, im_path: str, deepface_result: dict = None) -> str:
        """
        Use DeepFace to analyze the image at im_path,
        build a prompt, and return [age_group, race, gender, emotion].
        """
        if deepface_result == None:
            try:
                deepface_result = DeepFace.analyze(
                    img_path=im_path,
                    actions=['age','gender','race','emotion'],
                    detector_backend="skip",
                    enforce_detection=False
                )
                deepface_result = deepface_result[0]
            except:
                # fallback if face not detected
                deepface_result = {"age":'',"dominant_gender":"","dominant_race":"","emotion":{"neutral":1.0}}
        
        #deepface_result = deepface_result[0]
        
        age = deepface_result.get('age', 30)  # default 30 if missing
        dominant_gender = deepface_result.get('dominant_gender', '').lower()
        dominant_race = deepface_result.get('dominant_race', '').lower()
        # dominant_emotion = deepface_result.get('dominant_emotion', 'neutral').lower()
        emotions = deepface_result.get('emotion', {})
        # dominant_emotion = 'neutral'
        highest_confidence = 0

        # Find emotion with highest confidence
        for emotion, confidence in emotions.items():
            if emotion.lower() == 'neutral' and confidence >= 0.5:
                dominant_emotion = 'neutral'
                break
            
            if confidence > highest_confidence:
                highest_confidence = confidence
                if confidence >= 0.80:
                    dominant_emotion = emotion.lower()
                elif confidence > 0.50:
                    dominant_emotion = f"mildly {emotion.lower()}"
                else:
                    dominant_emotion = 'neutral'

        age_group = self.age_mapping(age)
        race_descriptor = dominant_race.title()
        gender_descriptor = dominant_gender.lower()
        emotion_descriptor = dominant_emotion.lower()

        prompt = (
            f"A photorealistic portrait of a {age_group} {race_descriptor} {gender_descriptor}, "
            f"with a {emotion_descriptor} expression"
        )
        
        prompt = (
        f"A {age_group} {race_descriptor} {gender_descriptor} face, "
        f"showing {emotion_descriptor} expression, "
        "natural skin texture with exact tone matching <0.8>, "
        "consistent with original lighting conditions, "
        "low-resolution security camera capture, "
        "slightly blurry, grainy texture, minor compression artifacts, "
        "unposed candid moment, realistic human features"
        )
        
        attrs = [age_group, race_descriptor, gender_descriptor, emotion_descriptor]
        print(prompt)
        return prompt, attrs

    def predict(
        self,
        image: str,
        prompt: str = "",
        mask: str = None,
        negative_prompt: str = "",
        strength: float = 0.8,
        max_height: int = 612,
        max_width: int = 612,
        steps: int = 20,
        seed: int = None,
        guidance_scale: float = 10.0,
        im_path: str = None
    ):
        """
        image (str): Path to input image.
        mask (str): Path to mask image.
        prompt (str): Positive text prompt.
        negative_prompt (str): Negative text prompt.
        strength (float): Control strength/weight.
        max_height (int): Maximum allowable height.
        max_width (int): Maximum allowable width.
        steps (int): Number of denoising steps.
        seed (int): Random seed (if None, random).
        guidance_scale (float): Guidance scale.

        Returns:
            Image.Image: Output image
        """

        # Handle random seed
        if not seed or seed == 0:
            seed = int.from_bytes(os.urandom(2), byteorder='big')
        generator = torch.Generator('cuda').manual_seed(seed)
        print("Using seed:", seed)
        
        #create prompts
        
        if negative_prompt == "":
            negative_prompt = (
                "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, "
                "anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, "
                "bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, "
                "mutation, mutated, ugly, disgusting, amputation"
            )
            # For 224x224 images:
            negative_prompt = (
                                "cartoon, anime, 3d render, painting, drawing, "
                                "perfect skin, studio lighting, sharp focus, "
                                "8k, DSLR, professional photo, "
                                "makeup, plastic texture, "
                                "disproportionate features, deformed iris"
                            )
        
        if prompt == "" or None:
            attrs = []

            if im_path is not None:
                prompt, attrs = self.create_prompt(im_path)
                # prompt += ", soft facial features, grainy texture, vintage surveillance photo"
                #prompt = "surveillance camera footage, CCTV quality, " + prompt + ", natural skin tones, color-matched to original, " + "consistent lighting with surroundings, [exact skin tone match:0.7]"
            else:
                prompt = "A photorealistic portrait of a person"
        
        # age_group, race_descriptor, gender_descriptor, emotion_descriptor = self.parse_prompt(prompt)
             
        # attrs = [age_group, race_descriptor, gender_descriptor, emotion_descriptor]
        attrs = []

        # Load and resize images
        # init_image = Image.open(image)
        init_image = image
        init_image = self.resize_image(init_image, max_width, max_height)
        width, height = init_image.size
        width, height = self.closest_multiple_of_8(width, height)
        init_image = init_image.resize((width, height))

        # Before passing mask to predict():
        mask_image = mask.convert("L").resize((width, height))
        mask_image = mask_image.filter(ImageFilter.MaxFilter(size=3))

        # Create inpainting condition
        inpainting_control_image = self.make_inpaint_condition(init_image, mask_image)
        inpainting_control_image = inpainting_control_image.to(dtype=torch.float16)
        
        # Create lineart condition
        lineart_control_image = self.lineart_processor(init_image)
        lineart_control_image = lineart_control_image.resize((width, height))
        
        # Create lineart condition
        openpose_control_image = self.openpose_detector(init_image)
        openpose_control_image = openpose_control_image.resize((width, height))

        images = [inpainting_control_image, lineart_control_image, openpose_control_image]
        # images = [inpainting_control_image, lineart_control_image] 

        # Run the pipeline
        with torch.no_grad():
            result = self.pipe(
                prompt_embeds=self.compel_proc(prompt),
                negative_prompt_embeds=self.compel_proc(negative_prompt),
                num_inference_steps=steps,
                generator=generator,
                eta=1,
                image=init_image,
                mask_image=mask_image,
                control_image=images,
                controlnet_conditioning_scale=strength,
                guidance_scale=guidance_scale,
            )


        out_image = result.images[0]
        return out_image, init_image, attrs
    
    def anonymize(
        self,
        image: str,
        prompt: str = "",
        mask: str = None,
        negative_prompt: str = "",
        strength: float = 0.8,
        max_height: int = 612,
        max_width: int = 612,
        steps: int = 20,
        seed: int = None,
        guidance_scale: float = 10.0,
        out_path: str  = "./res",
        im_path: str = None,
        max_try: int = 3,
        threshold: float = 0.3
    ):
        
        params = {
            "image": image,
            "prompt": prompt,
            "mask": mask,
            "negative_prompt": negative_prompt,
            "strength": strength,
            "max_height": max_height,
            "max_width": max_width,
            "steps": steps,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "im_path": im_path
        }
        
        print('Anonymization Started!')
        
        out_image, init_image, attrs = self.predict(**params)
        
        try:
            similarity = DeepFace.verify(
                img1_path=np.array(out_image)[..., ::-1],
                img2_path=np.array(init_image)[..., ::-1],
                detector_backend = "skip",
                threshold=threshold
            )
        except ValueError as e:
            print(f"Verification failed (ValueError)")
            similarity = {}
            similarity['verified'] = False
            print(f"Face could not found. Anonymization is unknown.")
            return out_image
        except Exception as e:
            raise e 
        
        count = 0 
        
        while similarity['verified'] == True:
            print(f"Image has not been properly anonymized with {similarity['distance']} distance between images, >={threshold} required.",
                "Process starts again with different configurations.")
            
            #out_image.save(f"./.cache/{similarity['distance']}.png")
            out_temp = out_image
            
            if isinstance(params["strength"], list):
                params["strength"] = [min(1.0, x + 0.05) for x in params["strength"]]
            else:
                params['strength'] = min(1.0, params['strength'] + 0.05)
            params["guidance_scale"] = min(20, params["guidance_scale"] + 1)
            params["steps"] = min(70, params["steps"] + 5)
            # params["seed"] = 0

            
            out_image, init_image, attrs = self.predict(**params)
            
            
            try:
                similarity = DeepFace.verify(
                    img1_path=np.array(out_image)[..., ::-1],
                    img2_path=np.array(init_image)[..., ::-1],
                    detector_backend = "skip",
                    threshold=threshold
                )
            except ValueError as e:
                print(f"Verification failed (ValueError):")
                similarity = {}
                similarity['verified'] = True
                similarity['distance'] = "Unknown"
                print(f"Face could not found. Anonymization is unknown. Trying again!")
                out_image = out_temp
            except Exception as e:
                continue
            
            count += 1
            if count > max_try-1: 
                print(f'Image can not anonymized in {max_try} trials. Returning last modified face')
                break
            torch.cuda.empty_cache() #might be removed
            torch.cuda.ipc_collect()

        # out_image.save(out_path)
        print(f"Image has been anonymized with {similarity['distance']}.")
        # print(f"Saved output image to: {out_path}")
        
        # After generation:
        out_image = out_image.filter(ImageFilter.GaussianBlur(radius=0.5))
        return out_image, round(similarity['distance'], 3), attrs
    

