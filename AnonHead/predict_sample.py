import os
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, AutoencoderKL
from controlnet_aux import LineartDetector, OpenposeDetector
from compel import Compel
from deepface import DeepFace


MODEL_NAME = "SG161222/Realistic_Vision_V5.0_noVAE"
VAE_CACHE = "vae-cache"


class Predictor:
    def __init__(self, device_id=0, load=True):
        """
        Args:
            device_id (int): which GPU to place the pipeline on
            load (bool): if True, call self.setup()
        """
        self.device_id = device_id
        if load:
            self.setup()

    def setup(self):
        """
        Create the pipeline in float32, move it to GPU device_id, 
        and optionally enable xFormers memory-efficient attention.
        """
        print(f"[Predictor.setup()] Loading all modules in float32 on GPU {self.device_id} ...")

        # 1) Load multiple ControlNets in float32
        controlnet_inpaint = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint",
            torch_dtype=torch.float32
        )
        lineart_controlnet = ControlNetModel.from_pretrained(
            "ControlNet-1-1-preview/control_v11p_sd15_lineart",
            torch_dtype=torch.float32
        )
        openpose_controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose",
            torch_dtype=torch.float32
        )

        self.lineart_processor = LineartDetector.from_pretrained("lllyasviel/Annotators")
        self.openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")

        controlnets = [controlnet_inpaint, lineart_controlnet, openpose_controlnet]

        # 2) Load VAE in float32
        vae = AutoencoderKL.from_single_file(
            "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors",
            dtype=torch.float32,
            cache_dir=VAE_CACHE
        )

        # 3) Create the pipeline, float32
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            MODEL_NAME,
            controlnet=controlnets,
            torch_dtype=torch.float32,
            vae=vae
        )

        # 4) Move pipeline to GPU device
        device_str = f"cuda:{self.device_id}"
        pipe.to(device_str)

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

    def closest_multiple_of_8(self, width, height):
        """
        Round width/height up to the closest multiple of 8.
        """
        w = ((width + 7) // 8) * 8
        h = ((height + 7) // 8) * 8
        return w, h

    def make_inpaint_condition(self, image, mask):
        """
        Convert image + mask into inpainting condition (pixels replaced with -1 where masked).
        """
        import torch
        import numpy as np
        img_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        mask_np = np.array(mask.convert("L")).astype(np.float32) / 255.0

        assert img_np.shape[:2] == mask_np.shape[:2], "Image and mask must have same size."
        img_np[mask_np > 0.5] = -1.0  # Mark masked region as -1
        img_np = np.expand_dims(img_np, 0).transpose(0, 3, 1, 2)
        return torch.from_numpy(img_np).to(device=f"cuda:{self.device_id}")

    def age_mapping(self, age: int) -> str:
        if age < 13:
            return "child"
        elif 13 <= age < 20:
            return "teen"
        elif 20 <= age < 30:
            return "young adult"
        elif 30 <= age < 50:
            return "middle-aged adult"
        else:
            return "elderly adult"

    def create_prompt(self, im_path: str):
        """
        Use DeepFace to analyze the image at im_path,
        build a prompt, and return [age_group, race, gender, emotion].
        """
        try:
            analyze_result = DeepFace.analyze(
                img_path=im_path,
                actions=['age','gender','race','emotion'],
                detector_backend="skip",
                enforce_detection=False
            )
            analyze_result = analyze_result[0]
        except:
            # fallback if face not detected
            analyze_result = {"age":30,"dominant_gender":"female","dominant_race":"white","emotion":{"neutral":1.0}}

        age = analyze_result.get("age", 30)
        gender = analyze_result.get("dominant_gender","").lower()
        race = analyze_result.get("dominant_race","").lower()
        emotions = analyze_result.get("emotion",{})
        dom_emotion = 'neutral'
        for emo, conf in emotions.items():
            if conf > 0.5:
                dom_emotion = emo.lower()
                break

        age_group = self.age_mapping(age)
        race_descriptor = race.title()
        gender_descriptor = gender.lower()
        emotion_descriptor = dom_emotion.lower()

        prompt = (
            f"A photorealistic portrait of a {age_group} {race_descriptor} {gender_descriptor}, "
            f"with a {emotion_descriptor} expression"
        )
        attrs = [age_group, race_descriptor, gender_descriptor, emotion_descriptor]
        return prompt, attrs

    def predict(
        self,
        image,
        mask=None,
        im_path=None,
        prompt="",
        negative_prompt="",
        strength=0.8,
        max_height=612,
        max_width=612,
        steps=20,
        seed=None,
        guidance_scale=10.0
    ):
        """
        Inpaint + ControlNet on a single image + mask, building a prompt automatically if prompt is empty.
        Returns (out_image, init_image, attrs).
        """

        # 1) Make a random seed if needed
        if not seed or seed == 0:
            seed = int.from_bytes(os.urandom(2), byteorder='big')
        generator = torch.Generator(device=f"cuda:{self.device_id}").manual_seed(seed)

        # 2) Provide a default negative prompt if none
        if not negative_prompt:
            negative_prompt = (
                "(deformed iris, deformed pupils, cgi, 3d, render, sketch, cartoon, drawing, "
                "anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, "
                "bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, "
                "mutation, mutated, ugly, disgusting, amputation"
            )

        # 3) If user gave no prompt, generate from DeepFace
        attrs = []
        if not prompt:
            if im_path is not None:
                prompt, attrs = self.create_prompt(im_path)
            else:
                prompt = "A photorealistic portrait of a person"

        # 4) Preprocess the input images
        init_image = image.convert("RGB")
        init_image = self.resize_image(init_image, max_width, max_height)
        w, h = init_image.size
        w, h = self.closest_multiple_of_8(w, h)
        init_image = init_image.resize((w,h))

        if mask is None:
            # If no mask given, create an all-black mask => no inpainting
            mask = Image.new("L", (w,h), 0)
        else:
            mask = mask.convert("L").resize((w,h))

        # 5) Create control images
        inpainting_control_image = self.make_inpaint_condition(init_image, mask)
        lineart_control = self.lineart_processor(init_image).resize((w,h))
        openpose_control = self.openpose_detector(init_image).resize((w,h))


        # 6) Actually run the pipeline
        result = self.pipe(
            prompt_embeds=self.compel_proc(prompt),
            negative_prompt_embeds=self.compel_proc(negative_prompt),
            image=init_image,
            mask_image=mask,
            control_image=[inpainting_control_image, lineart_control, openpose_control],
            controlnet_conditioning_scale=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            eta=1,
            generator=generator
        )

        out_image = result.images[0]
        return out_image, init_image, attrs

    def anonymize(self, image, mask=None, **kwargs):
        """
        Original anonymize method that also verifies face difference with DeepFace.
        ...
        """
        # [Your existing code, unchanged, using self.predict(...)]
        # Or do a partial example for brevity
        pass

    ########################################################################
    # NEW METHOD: anonymize_batch (or predict_batch)
    ########################################################################
    def anonymize_batch(
        self,
        image_paths,
        mask_paths=None,
        output_dir="./batch_results",
        **kwargs
    ):
        """
        Process a list of images (and optional masks) in a batch using the same loaded pipeline.
        Avoids re-initializing the model for each image.
        
        Args:
            image_paths (list of str): Paths to input images (or list of PIL Images).
            mask_paths (list of str], optional): Paths to mask images. Must match image_paths in length if provided.
            output_dir (str): Directory to save anonymized images.
            **kwargs: Additional arguments to pass to self.predict(...) or self.anonymize(...).
        """
        from tqdm import tqdm
        import os
        from PIL import Image
        
        os.makedirs(output_dir, exist_ok=True)
        
        # If masks are provided, they must match in length
        if mask_paths and len(mask_paths) != len(image_paths):
            raise ValueError("mask_paths length must match image_paths length.")

        for i, img_item in enumerate(tqdm(image_paths, desc="Batch anonymization")):
            # If image_paths are file paths, open them
            if isinstance(img_item, str):
                image = Image.open(img_item).convert("RGB")
                img_path = img_item
            else:
                # Already a PIL Image
                image = img_item
                img_path = f"image_{i}.png"  # fallback name

            # If we have mask paths, open or use PIL
            if mask_paths:
                m_item = mask_paths[i]
                if isinstance(m_item, str):
                    mask = Image.open(m_item).convert("RGB")
                else:
                    mask = m_item
            else:
                mask = None

            # You can call self.predict(...) or self.anonymize(...).
            # If you want face verification, call anonymize. If not, call predict.
            out_image, init_image, attrs = self.predict(
                image=image,
                mask=mask,
                im_path=img_path,
                **kwargs
            )

            # Save
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_name = f"{base}_anon.png"
            out_path = os.path.join(output_dir, out_name)
            out_image.save(out_path)
