# Image Generation

AI-driven image generation harnesses artificial intelligence to craft visual content, encompassing images and artwork. These systems possess the capability to generate images from textual descriptions, enhance existing photos, or craft entirely new visuals by leveraging diverse inputs and preferences. Additionally, they can generate images from other images, further extending their creative potential.

<img src="/assets/images/aiArtwork.png" alt="AI music meme" />

### Articles & Papers

- [Image GPT](https://openai.com/research/image-gpt) (2020)

- [High-resolution image synthesis with latent diffusion models](https://arxiv.org/abs/2112.10752) (2021): A research paper introduces a method for high-resolution image synthesis through latent diffusion models, breaking down the process into sequential denoising autoencoders. It attains top-tier results on image synthesis benchmarks.

- [DALL·E: Creating images from text ](https://openai.com/research/dall-e) (2021) is an AI system that crafts images from text descriptions. It's a transformer model trained to generate images in high resolution and offers creative applications. [(paper)](https://arxiv.org/abs/2102.12092)

- [CLIP: Connecting text and images ](https://openai.com/research/clip) (2021) by OpenAI connects text and images, bridging computer vision and natural language processing. Motivated by addressing challenges in deep learning and enabling zero-shot transfer, CLIP's evaluation is robust. However, it requires careful prompt engineering, and concerns regarding biases persist. [(paper)](https://arxiv.org/abs/2103.00020)

- [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487) (2022): by Goggle, unveils Imagen, a text-to-image diffusion model known for its remarkable photorealism and deep language comprehension. It employs a diffusion process, iteratively adding noise to a latent representation to generate images. Imagen is trained on a substantial text-image dataset, delivering top-tier performance on various image synthesis benchmarks. [(website)](https://imagen.research.google/)

- [DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) (2022): by Google, The paper suggests a technique to fine-tune subject-driven text-to-image diffusion models by introducing new tokens in the embedding space of a frozen model. [(website)](https://dreambooth.github.io/)

- [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) (2023): The paper introduces a technique for incorporating conditional control into text-to-image diffusion models, enabling image generation based on specific conditions like object presence or styles.

- [DALL·E 3](https://openai.com/dall-e-3) (2023): OpenAI's latest AI system, creates lifelike images from text descriptions. An evolution of DALL-E 2, it's more potent with improved editing, vivid concept visualization, and enhanced realism. This exemplifies AI's swift advancement in creative tasks, supporting multi-modal AI research and applications. [(paper)](https://cdn.openai.com/papers/DALL_E_3_System_Card.pdf)

### Courses

- [Diffusers](https://huggingface.co/docs/diffusers/index): Hugging Face's Diffusers library is versatile, simplifying work with state-of-the-art diffusion models. Key features include pre-trained diffusion pipelines for image, audio, and 3D structure generation, customizable noise schedulers, practical guides, support for both inference and training, and a focus on usability. The library supports various tasks and offers resources on ethical guidelines and safety implementations.

### Explainers

- [How AI Image Generators Work (Stable Diffusion / Dall-E) - Computerphile](https://www.youtube.com/watch?v=1CIpzeNxIhU&t=92s) by Dr Mike Pound, elaborates on the core working principle of image generation, specifically focusing on Stable Diffusion and DALL-E. Stable Diffusion employs a diffusion process to create high-quality images.

- [How Stable Diffusion Works (AI Image Generation)](https://www.youtube.com/watch?v=sFztPP9qPRc&t=427s) by Gonkee, explains how Stable Diffusion works for generating high-quality images.

### Guides

- [Image Prompting](https://learnprompting.org/docs/category/%EF%B8%8F-image-prompting) by Learn Prompting, offers an open-source course on Image Prompting techniques for both beginners and professionals.

### Reference

- [Camenduru's 3D ML Papers](https://github.com/camenduru#-3d-ml-papers): GitHub page featuring repositories dedicated to 3D machine learning papers.

<br />
<br />
# Multi-dimensional Image Generation

### Papers

- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934) (2020): NeRF uses deep neural networks to synthesize new views of complex scenes, excelling in this task. Its 5D function maps 3D space and 2D viewing directions to radiance values. NeRF finds applications in robotics, urban mapping, virtual reality, and more. [(website)](https://www.matthewtancik.com/nerf)

- [DreamFusion: Text-to-3D using 2D diffusion](https://arxiv.org/pdf/2209.14988.pdf) (2022): presents an innovative approach for creating 3D models from text descriptions using deep learning. It employs a frozen diffusion model to generate plausible images from text and introduces "Sample-Driven Synthesis" (SDS) for 3D model generation from 2D images. The methods outperform existing approaches and are valuable for researchers in text-to-3D and image-to-3D synthesis. [(website)](https://dreamfusion3d.github.io/)
