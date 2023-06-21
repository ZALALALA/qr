import subprocess

# Install required packages
subprocess.check_call(['pip', 'install', '-r', 'requirements.txt'])

import torch
import streamlit as st
from PIL import Image
import qrcode

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    HeunDiscreteScheduler,
    EulerDiscreteScheduler,
)

qrcode_generator = qrcode.QRCode(
    version=1,
    error_correction=qrcode.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)

controlnet = ControlNetModel.from_pretrained(
    "DionTimmer/controlnet_qrcode-control_v1p_sd15", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16,
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()

def resize_for_condition_image(input_image: Image.Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img

SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(
        config, use_karras=True, algorithm_type="sde-dpmsolver++"
    ),
    "DPM++ Karras": lambda config: DPMSolverMultistepScheduler.from_config(
        config, use_karras=True
    ),
    "Heun": lambda config: HeunDiscreteScheduler.from_config(config),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
    "DDIM": lambda config: DDIMScheduler.from_config(config),
    "DEIS": lambda config: DEISMultistepScheduler.from_config(config),
}

def generate_artistic_qr_code(qr_content, qr_image, prompt, scaling_factor, strength, seed, sampler, diffusion_steps):
    if qr_content:
        qrcode_generator.clear()
        qrcode_generator.add_data(qr_content)
        qrcode_generator.make(fit=True)
        qr_image = qrcode_generator.make_image(fill_color="black", back_color="white")
    else:
        qr_image = Image.open(qr_image).convert("1")

    with torch.no_grad():
        image = torch.tensor([resize_for_condition_image(qr_image, 256)], device="cuda").permute(0, 3, 1, 2)
        prompt_text = prompt
        prompt_text = prompt_text.strip().replace("\n", " \\n ")
        prompt_text = f" {prompt_text} "

        result_image = pipe.run(
            image,
            prompt=prompt_text,
            strength=strength,
            scale=scaling_factor,
            seed=seed,
            sampler=SAMPLER_MAP[sampler],
            num_diffusion_steps=diffusion_steps,
        )[0].cpu()

        result_image = Image.fromarray(result_image)

    return result_image

def main():
    st.title("QR Code AI Art Generator")
    st.markdown(
        """
        ## 💡 How to Use
        This app generates artistic QR codes using AI. You can provide a QR code content or upload a QR code image, and specify a prompt that guides the generation process. The generated QR code will be an artistic representation based on the given input.

        The app uses the Stable Diffusion pipeline along with a ControlNet model to generate the QR code art.

        - **QR Code Content**: Enter the content or URL that the QR code should represent.
        - **QR Code Image (Optional)**: You can upload a QR code image instead of providing the content directly. If both content and image are provided, the image takes precedence.
        - **Prompt**: Enter a prompt that guides the generation. This can be a short description of the desired artistic style or any other guiding instructions.
        - **Negative Prompt**: Words that describe what the generated QR code should not look like.
        - **Controlnet Conditioning Scale**: Controls the strength of the conditioning on the control image. Higher values result in more influence from the control image.
        - **Strength**: Controls the amount of noise added to the QR code image. Higher values result in more artistic distortion.
        - **Seed**: Controls the random seed used for generation. Changing the seed will produce a different result.
        - **Sampler**: Select the diffusion sampler to use. Different samplers may produce different results.
        - **Number of Diffusion Steps**: Controls the number of diffusion steps taken during the generation process. Higher values may result in more refined images but also increase computation time.

        Once you have provided the necessary inputs, click the **Generate QR Code Art** button to see the result.
        """
    )

    qr_code_content = st.text_input("QR Code Content", help="QR Code Content or URL")
    qr_code_image = st.file_uploader("QR Code Image (Optional)", type=["png", "jpg", "jpeg"])
    prompt = st.text_input("Prompt", help="Prompt that guides the generation")
    negative_prompt = st.text_input(
        "Negative Prompt",
        value="ugly, disfigured, low quality, blurry, nsfw",
        help="Words that describe what the generated QR code should not look like",
    )
    scaling_factor = st.slider("Controlnet Conditioning Scale", min_value=0.0, max_value=5.0, step=0.01, value=1.1)
    strength = st.slider("Strength", min_value=0.0, max_value=1.0, step=0.01, value=0.9)
    seed = st.slider("Seed", min_value=-1, max_value=9999999999, step=1, value=2313123, format="%d")
    sampler = st.selectbox("Sampler", list(SAMPLER_MAP.keys()), index=0)
    diffusion_steps = st.number_input("Number of Diffusion Steps", min_value=1, value=40, step=1, format="%d")

    if st.button("Generate QR Code Art"):
        result = generate_artistic_qr_code(
            qr_code_content,
            qr_code_image,
            prompt,
            scaling_factor,
            strength,
            seed,
            sampler,
            diffusion_steps,
        )
        st.image(result, caption="Result Image", use_column_width=True)


if __name__ == "__main__":
    main()
