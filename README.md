# Image-generation
!pip install diffusers transformers accelerate torch torchvision

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Check for CUDA availability and set device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")


def generate_images(prompt, num_images=1, guidance_scale=7.5, num_inference_steps=50, image_size=512):
    """
    Generates images using Stable Diffusion.

    Args:
        prompt: The text prompt to guide image generation.
        num_images: The number of images to generate.
        guidance_scale: How strongly the generated image reflects the prompt.
        num_inference_steps: The number of denoising steps. Higher values mean better quality but longer generation.
        image_size: The size of the generated images (e.g., 512, 768).  Stable Diffusion often works best with multiples of 64.

    Returns:
        A list of PIL Image objects.
    """

    # Load the Stable Diffusion pipeline (only once)
    if not hasattr(generate_images, "pipeline"):  # Check if pipeline exists
        generate_images.pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device) #Use float16 if using CUDA

    images = []
    for _ in range(num_images):
        image = generate_images.pipeline(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=image_size, width=image_size).images[0]
        images.append(image)

    return images



# Example usage:
prompts = [
    "A photorealistic image of a cat wearing a top hat",
    "A vibrant abstract painting with swirling colors",
    "A futuristic cityscape at night, with neon lights"
]

for prompt in prompts:
    generated_images = generate_images(prompt, num_images=2) # Generate 2 images per prompt

    for i, image in enumerate(generated_images):
        image.save(f"generated_image_{prompt.replace(' ', '_')}_{i}.png") # Save with descriptive filenames
        print(f"Image saved as generated_image_{prompt.replace(' ', '_')}_{i}.png")

# To display an image (optional, if you're in an environment that supports it like a notebook):
# generated_images[0].show()
