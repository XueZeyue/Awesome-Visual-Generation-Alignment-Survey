# Awesome-Visual-Generation-Alignment-Survey
The collection of awesome papers on the aligement of visual generation models (including AR and diffusion models)

We also use this [repo](https://github.com/xie-lab-ml/awesome-alignment-of-diffusion-models) as a reference.

**We welcome community contributions!**

## Tutorial on Reinforcement Learning
- [CS285](https://rail.eecs.berkeley.edu/deeprlcourse/) from UC Berkeley. (This course is mainly for robotics control, but very important if you want to be an expert on RL.)
- [Introduction to Reinforcement Learning](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
- [动手学强化学习](https://github.com/boyu-ai/Hands-on-RL) (in Chinese)
- [强化学习知乎专栏](https://www.zhihu.com/column/reinforcementlearning) (in Chinese)

## First Two Works in Each Subfield:
**Traditional Policy Gradient**
- DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models, [[pdf]](https://arxiv.org/pdf/2305.16381)
- Training Diffusion Models with Reinforcement Learning, [[pdf]](https://arxiv.org/abs/2305.13301)

**GRPO**
- DanceGRPO: Unleashing GRPO on Visual Generation, [[pdf]](https://arxiv.org/pdf/2505.07818)
- Flow-GRPO: Training Flow Matching Models via Online RL, [[pdf]](https://arxiv.org/pdf/2505.05470)

**DPO**
- Diffusion Model Alignment Using Direct Preference Optimization, [[pdf]](https://arxiv.org/pdf/2311.12908)
- Using Human Feedback to Fine-tune Diffusion Models without Any Reward Model, [[pdf]](https://arxiv.org/pdf/2311.13231)

**Reward Feedback Learning**
- ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation, [[pdf]](https://arxiv.org/pdf/2304.05977)
- Directly Fine-Tuning Diffusion Models on Differentiable Rewards, [[pdf]](https://arxiv.org/pdf/2309.17400)

**Alignment on AR models**
- Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step [[pdf]](https://arxiv.org/abs/2501.13926)
- Autoregressive Image Generation Guided by Chains of Thought [[pdf]](https://arxiv.org/pdf/2502.16965)

**Reward Models**
- ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation, [[pdf]](https://arxiv.org/pdf/2304.05977)
- Human Preference Score: Better Aligning Text-to-Image Models with Human Preference, [[pdf]](https://arxiv.org/pdf/2303.14420)


## Alignment on Diffusion/Flow Models
### Reinforcement Learning-based (RLHF)
- DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models, [[pdf]](https://arxiv.org/pdf/2305.16381)
- Training Diffusion Models with Reinforcement Learning, [[pdf]](https://arxiv.org/abs/2305.13301
)
- Towards Better Alignment: Training Diffusion Models with Reinforcement Learning Against Sparse Rewards, [[pdf]](https://arxiv.org/pdf/2501.06655)
- Schedule On the Fly: Diffusion Time Prediction for Faster and Better Image Generation, [[pdf]](https://arxiv.org/pdf/2412.01243v2)
- DanceGRPO: Unleashing GRPO on Visual Generation, [[pdf]](https://arxiv.org/pdf/2505.07818)
- Flow-GRPO: Training Flow Matching Models via Online RL, [[pdf]](https://arxiv.org/pdf/2505.05470)
- MixGRPO: Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE, [[pdf]](https://arxiv.org/pdf/2507.21802)

### DPO-based (referred to [here](https://github.com/xie-lab-ml/awesome-alignment-of-diffusion-models))
- Diffusion Model Alignment Using Direct Preference Optimization, [[pdf]](https://arxiv.org/pdf/2311.12908)
- Using Human Feedback to Fine-tune Diffusion Models without Any Reward Model, [[pdf]](https://arxiv.org/pdf/2311.13231)
- A Dense Reward View on Aligning Text-to-Image Diffusion with Preference, [[pdf]](https://arxiv.org/pdf/2402.08265)
- Aligning Diffusion Models by Optimizing Human Utility, [[pdf]](https://arxiv.org/pdf/2404.04465)
- Tuning Timestep-Distilled Diffusion Model Using Pairwise Sample Optimization, [[pdf]](https://arxiv.org/pdf/2410.03190)
- Scalable Ranked Preference Optimization for Text-to-Image Generation, [[pdf]](https://arxiv.org/pdf/2410.18013)
- Prioritize Denoising Steps on Diffusion Model Preference Alignment via Explicit Denoised Distribution Estimation, [[pdf]](https://arxiv.org/pdf/2411.14871)
- PatchDPO: Patch-level DPO for Finetuning-free Personalized Image Generation. arXiv 2024, [[pdf]](https://arxiv.org/pdf/2412.03177)
- SafetyDPO: Scalable Safety Alignment for Text-to-Image Generation, [[pdf]](https://arxiv.org/pdf/2412.10493)
- Margin-aware Preference Optimization for Aligning Diffusion Models without Reference, [[pdf]](https://arxiv.org/pdf/2406.06424)
- DSPO: Direct Score Preference Optimization for Diffusion Model Alignment, [[pdf]](https://openreview.net/forum?id=xyfb9HHvMe)
- Direct Distributional Optimization for Provable Alignment of Diffusion Models, [[pdf]](https://arxiv.org/pdf/2502.02954)
- Boost Your Human Image Generation Model via Direct Preference Optimization, [[pdf]](https://arxiv.org/pdf/2405.20216)
- Curriculum Direct Preference Optimization for Diffusion and Consistency Models, [[pdf]](https://arxiv.org/pdf/2405.13637)
- Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization, [[pdf]](https://arxiv.org/pdf/2406.04314)
- Personalized Preference Fine-tuning of Diffusion Models, [[pdf]](https://arxiv.org/pdf/2501.06655)
- Calibrated Multi-Preference Optimization for Aligning Diffusion Models, [[pdf]](https://arxiv.org/pdf/2502.02588)
- InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment, [[pdf]](https://arxiv.org/pdf/2503.18454)
- CHATS: Combining Human-Aligned Optimization and Test-Time Sampling for Text-to-Image Generation, [[pdf]](https://arxiv.org/pdf/2502.12579)
- D-Fusion: Direct Preference Optimization for Aligning Diffusion Models with Visually Consistent Samples, [[pdf]](https://arxiv.org/abs/2505.22002)
- Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences, [[pdf]](https://arxiv.org/abs/2506.02698)
- Refining Alignment Framework for Diffusion Models with Intermediate-Step Preference Ranking, [[pdf]](https://arxiv.org/pdf/2502.01667)
- Aligning Text to Image in Diffusion Models is Easier Than You Think, [[pdf]](https://arxiv.org/pdf/2503.08250)
- OnlineVPO: Align Video Diffusion Model with Online Video-Centric Preference Optimization, [[pdf]](https://arxiv.org/pdf/2412.15159)
- VideoDPO: Omni-Preference Alignment for Video Diffusion Generation, [[pdf]](https://arxiv.org/pdf/2412.14167)
- DreamDPO: Aligning Text-to-3D Generation with Human Preferences via Direct Preference Optimization, [[pdf]](https://arxiv.org/abs/2502.04370)
- Flow-DPO: Improving Video Generation with Human Feedback, [[pdf]](https://arxiv.org/abs/2501.13918)
- HuViDPO: Enhancing Video Generation through Direct Preference Optimization for Human-Centric Alignment, [[pdf]](https://arxiv.org/pdf/2502.01690)
- Diffusion-NPO: Negative Preference Optimization for Better Preference Aligned Generation of Diffusion Models, [[pdf]](https://openreview.net/pdf?id=iJi7nz5Cxc)


### Reward Feedback Learning
- ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation, [[pdf]](https://arxiv.org/pdf/2304.05977)
- Aligning Text-to-Image Diffusion Models with Reward Backpropagation, [[pdf]](https://arxiv.org/pdf/2310.03739v2)
- Directly Fine-Tuning Diffusion Models on Differentiable Rewards, [[pdf]](https://arxiv.org/pdf/2309.17400)
- Feedback Efficient Online Fine-Tuning of Diffusion Models, [[pdf]](https://arxiv.org/pdf/2402.16359)
- Deep Reward Supervisions for Tuning Text-to-Image Diffusion Models, [[pdf]](https://arxiv.org/abs/2405.00760)
- Reward Fine-Tuning Two-Step Diffusion Models via Learning Differentiable Latent-Space Surrogate Reward, [[pdf]](https://arxiv.org/pdf/2411.15247)


### Technical Reports 
We only list the reports using alignment methods:
- Seedream 2.0: A native chinese-english bilingual image generation foundation model, [[pdf]](https://arxiv.org/pdf/2503.07703)
- Seedream 3.0 technical report, [[pdf]](https://arxiv.org/pdf/2504.11346)
- Seedance 1.0: Exploring the Boundaries of Video Generation Models, [[pdf]](https://arxiv.org/pdf/2506.09113)
- Step-Video-T2V Technical Report: The Practice, Challenges, and Future of Video Foundation Model, [[pdf]](https://arxiv.org/pdf/2502.10248)
- Qwen-Image Technical Report, [[pdf]](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/Qwen_Image.pdf)

## Alignment on AR models (referred to [here](https://github.com/ChaofanTao/Autoregressive-Models-in-Vision-Survey))
- Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step, [[pdf]](https://arxiv.org/abs/2501.13926)
- Autoregressive Image Generation Guided by Chains of Thought, [[pdf]](https://arxiv.org/pdf/2502.16965)
- LightGen: Efficient Image Generation through Knowledge Distillation and Direct Preference Optimization, [[pdf]](https://arxiv.org/abs/2503.08619)
- SimpleAR: Pushing the Frontier of Autoregressive Visual Generation through Pretraining, SFT, and RL, [[pdf]](https://arxiv.org/pdf/2504.11455)
- UniGen: Enhanced Training & Test-Time Strategies for Unified Multimodal Understanding and Generation, [[pdf]](https://arxiv.org/pdf/2505.14682) 
- UniRL: Self-Improving Unified Multimodal Models via Supervised and Reinforcement Learning, [[pdf]](https://arxiv.org/pdf/2505.23380)
- ReasonGen-R1: CoT for Autoregressive Image Generation model through SFT and RL, [[pdf]](https://arxiv.org/pdf/2505.24875)
- Unlocking Aha Moments via Reinforcement Learning: Advancing Collaborative Visual Comprehension and Generation, [[pdf]](https://arxiv.org/pdf/2506.01480)
- Delving into RL for Image Generation with CoT: A Study on DPO vs. GRPO, [[pdf]](https://arxiv.org/abs/2505.17017)
- CoT-lized Diffusion: Let’s Reinforce T2I Generation Step-by-step, [[pdf]](https://arxiv.org/pdf/2507.04451) 
- X-Omni: Reinforcement Learning Makes Discrete Autoregressive Image Generative Models Great Again, [[pdf]](https://arxiv.org/pdf/2507.22058)
- T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT, [[pdf]](https://arxiv.org/pdf/2505.00703)

## Benchmarks & Reward Models
**Benchmarks**
- DALL-Eval: Probing the Reasoning Skills and Social Biases of Text-to-Image Generative Transformers, [[pdf]](https://arxiv.org/pdf/2202.04053)
- Human Evaluation of Text-to-Image Models on a Multi-Task Benchmark, [[pdf]](https://arxiv.org/pdf/2211.12112)
- LLMScore: Unveiling the Power of Large Language Models in Text-to-Image Synthesis Evaluation, [[pdf]](https://arxiv.org/pdf/2305.11116)
- VPGen & VPEval: Visual Programming for Text-to-Image Generation and Evaluation, [[pdf]](https://arxiv.org/pdf/2305.15328)
- GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment, [[pdf]](https://arxiv.org/pdf/2310.11513)
- Holistic Evaluation of Text-to-Image Models, [[pdf]](https://arxiv.org/pdf/2311.04287)
- Social Reward: Evaluating and Enhancing Generative AI through Million-User Feedback from an Online Creative Community, [[pdf]](https://arxiv.org/pdf/2402.09872)
- Rich Human Feedback for Text to Image Generation, [[pdf]](https://arxiv.org/pdf/2312.10240)
- Learning Multi-Dimensional Human Preference for Text-to-Image Generation, [[pdf]](https://arxiv.org/pdf/2405.14705)
- Evaluating Text-to-Visual Generation with Image-to-Text Generation, [[pdf]](https://arxiv.org/pdf/2404.01291)
- Multimodal Large Language Models Make Text-to-Image Generative Models Align Better, [[pdf]](https://openreview.net/pdf?id=IRXyPm9IPW)
- Measuring Style Similarity in Diffusion Models, [[pdf]](https://arxiv.org/pdf/2404.01292)
- T2I-CompBench++: An Enhanced and Comprehensive Benchmark for Compositional Text-to-Image Generation, [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10847875)
- DreamBench++: A Human-Aligned Benchmark for Personalized Image Generation, [[pdf]](https://arxiv.org/pdf/2406.16855)
- PAL: Sample-Efficient Personalized Reward Modeling for Pluralistic Alignment, [[pdf]](https://openreview.net/pdf?id=1kFDrYCuSu)
- Video-Bench: Human-Aligned Video Generation Benchmark, [[pdf]](https://arxiv.org/pdf/2504.04907)

**Reward Models**
- Human Preference Score: Better Aligning Text-to-Image Models with Human Preference, [[pdf]](https://arxiv.org/pdf/2303.14420)
- ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation, [[pdf]](https://arxiv.org/pdf/2304.05977)
- Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation, [[pdf]](https://arxiv.org/pdf/2305.01569)
- Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis, [[pdf]](https://arxiv.org/pdf/2306.09341)
- Improving Video Generation with Human Feedback, [[pdf]](https://arxiv.org/pdf/2501.13918)
- VideoScore: Building Automatic Metrics to Simulate Fine-grained Human Feedback for Video Generation, [[pdf]](https://arxiv.org/pdf/2406.15252)
- VisionReward: Fine-Grained Multi-Dimensional Human Preference Learning for Image and Video Generation, [[pdf]](https://arxiv.org/pdf/2412.21059)
- Unified Reward Model for Multimodal Understanding and Generation, [[pdf]](https://arxiv.org/pdf/2503.05236)
- LiFT: Leveraging Human Feedback for Text-to-Video Model Alignment, [[pdf]](https://arxiv.org/pdf/2412.04814)

## Alignment with Prompt Engineering
- Optimizing Prompts for Text-to-Image Generation, [[pdf]](https://arxiv.org/pdf/2212.09611)
- RePrompt: Automatic Prompt Editing to Refine AI-Generative Art Towards Precise Expressions, [[pdf]](https://arxiv.org/pdf/2302.09466)
- Improving Text-to-Image Consistency via Automatic Prompt Optimization, [[pdf]](https://arxiv.org/pdf/2403.17804)
- Dynamic Prompt Optimizing for Text-to-Image Generation, [[pdf]](https://arxiv.org/pdf/2404.04095)
- T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT, [[pdf]](https://arxiv.org/pdf/2505.00703)
- RePrompt: Reasoning-Augmented Reprompting for Text-to-Image Generation via Reinforcement Learning, [[pdf]](https://arxiv.org/pdf/2505.17540?)
