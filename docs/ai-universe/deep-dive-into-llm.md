# Deep dive into LLMs

Embark on an in-depth exploration of LLMs with our comprehensive resources. This deep dive offers a thorough understanding of LLMs, equipping you with advanced knowledge and practical insights in the field.

  <img src="/assets/images/memes/llmsMeme.png" alt="LLMs meme" />

### Courses

- [LLMs: Foundation Models from the Ground Up](https://www.edx.org/learn/computer-science/databricks-large-language-models-foundation-models-from-the-ground-up) by Databricks, provides an in-depth exploration of foundational models in LLMs, highlighting key innovations that fueled the rise of transformer-based models like BERT, GPT, and T5. It also covers advanced techniques, such as Flash Attention, LoRa, AliBi, and PEFT, contributing to the ongoing enhancements of LLM capabilities, including applications like ChatGPT.

- [State of GPT](https://build.microsoft.com/en-US/sessions/db3f4859-cd30-4445-a0cd-553c3304f8e2) by Andrej Karpathy, provides an easily comprehensible breakdown of the inner workings of ChatGPT and GPT models, offering guidance on their utilization and potential R&D pathways.

- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) by Andrej Karpathy, where you'll embark on a journey to build neural networks from the ground up, all through code. Starting with the fundamentals of backpropagation, progress to crafting cutting-edge deep neural networks like GPT. Language models serve as an excellent entry point into deep learning, with skills transferable across domains, making them our primary focus.

### Explainers

- [Large Language Models Process Explained what makes them tick & how they work under the hood!](https://www.youtube.com/watch?v=_Pt-rGE4zEE&t=5s) by AemonAlgiz, covers foundational concepts like softmax, tokenization, embedding, and positional encoding. Dive into the magic of attention and multi-attention heads for enhanced comprehension, all presented with accessible clarity and depth, suitable for AI enthusiasts at all levels.

- [Prompt injection: What’s the worst that can happen?](https://simonwillison.net/2023/Apr/14/worst-that-can-happen/): Prompt injection represents a significant security concern within LLM applications, and while there is no flawless remedy, Simon Willison provides a comprehensive explanation of this issue in his post. Simon consistently produces exceptional content on AI-related topics.

### LLM benchmarks

- [Chatbot Arena](https://lmsys.org/blog/2023-05-03-arena/): an innovative benchmark platform designed for LLMs. Elo rating system-based leaderboard, inspired by competitive games like chess, encourage the entire community to participate by submitting new models, evaluating their performance, and engaging in the exciting world of LLM battles.

- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard): A ranking by Hugging Face, comparing open source LLMs across a collection of standard benchmarks and tasks.

### Articles

- [The Scaling Hypothesis](https://gwern.net/scaling-hypothesis): Explore the Scaling Hypothesis, a captivating theory that posits larger AI models outperform smaller ones with ample data and resources. Delve into its impact on language models like GPT-3, controversies, applications, and ongoing debates among researchers. Discover the potential for achieving human-level or superhuman AI, and how organizations like EleutherAI are actively testing its limits through open-source models.

- [Building LLM applications for production](https://huyenchip.com/2023/04/11/llm-engineering.html): Chip Huyen explores several significant hurdles encountered in developing LLM applications, offers solutions for tackling them, and highlights the most suitable use cases for these applications.

- [chinchilla's wild implications](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications): This post delves into language model scaling laws, particularly those from the DeepMind paper introducing Chinchilla. Chinchilla, with 33-B parameters, defies the Scaling Hypothesis, highlighting the multifaceted role of factors like model architecture and data curation in performance.

- [GPT-4](https://openai.com/research/gpt-4): OpenAI's latest milestone, is a versatile multimodal model accepting text and image inputs, excelling in creative and technical writing. It generates, edits, and collaborates with users. It handles over 25k words, making it suitable for long-form content, conversations, and document analysis. Although advanced, it may have occasional reasoning errors and gullibility.

- [State of AI Report 2023](https://www.stateof.ai/) provides an exhaustive overview of AI, encompassing technology breakthroughs, industry developments, politics, safety, economic impacts, and future predictions. It encourages contributions from the AI community, fostering informed discussions about AI's future.

### Reference

- [MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models](https://meta-math.github.io/): MetaMath is a project for enhancing mathematical questions for language models. It builds the [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) dataset and fine-tunes LLaMA-2 models, creating specialized mathematical reasoning models. Results show MetaMath's significant performance lead on GSM8K and MATH benchmarks, even surpassing models of the same size. The project offers [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) data, [pretrained models](https://huggingface.co/meta-math/MetaMath-7B-V1.0), and [code](https://github.com/meta-math/MetaMath) on GitHub for public use, aiding those interested in generating mathematical questions for LLMs. [(paper)](https://arxiv.org/abs/2309.12284)

- [AutoGPT: An experimental open-source attempt to make GPT-4 fully autonomous](https://github.com/Significant-Gravitas/AutoGPT): AutoGPT is an open-source demonstration of GPT-4's capabilities. It's a conversational AI programming system that can generate code in response to prompts. The AutoGPT GitHub repository provides plugins, templates, and benchmarks, including a vector memory revamp. [(website)](https://news.agpt.co/) [(docs)](https://docs.agpt.co/)

- [BabyAGI](https://github.com/yoheinakajima/babyagi): This Python script showcases an AI task management system, using OpenAI and vector databases to create, prioritize, and execute tasks based on previous results and a defined objective.

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://blog.research.google/2022/11/react-synergizing-reasoning-and-acting.html): A novel paradigm fuses reasoning and acting in language models. It excels in language reasoning tasks, producing verbal traces and text actions simultaneously, enhancing dynamic reasoning, and adapting to external input for improved performance. [(paper)](https://arxiv.org/abs/2210.03629)

- [MemGPT: Towards LLMs as Operating Systems](https://memgpt.ai/) is an AI project enhancing memory in AI models for improved conversation and document analysis. It's an OS managing multiple memory tiers, extending context for large language models. It uses interrupts for user control and enables long-term interaction with dynamic conversational agents. [(paper)](https://arxiv.org/abs/2310.08560) [(code)](https://github.com/cpacker/MemGPT)

### Papers

- [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html) (2023): Alpaca 7B, a model fine-tuned based on the LLaMA 7B model using 52k instruction-following demonstrations, exhibits qualitative similarity to OpenAI's text-davinci-003 in single-turn instruction following during our initial assessment. Remarkably, Alpaca maintains a compact size and is straightforward and cost-effective to replicate. [(code)](https://github.com/tatsu-lab/stanford_alpaca) [(paper)](https://arxiv.org/pdf/2303.16199.pdf)

- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) (2023) : Offers a comprehensive overview of the evolving landscape of large language models, exploring their capabilities, applications, and challenges in NLP.

- [Sparks of Artificial General Intelligence: Early experiments with GPT-4](https://arxiv.org/abs/2303.12712) (2023): Initial assessment conducted by Microsoft Research on GPT-4, the most sophisticated LLM currently available, in comparison to human cognitive abilities.
