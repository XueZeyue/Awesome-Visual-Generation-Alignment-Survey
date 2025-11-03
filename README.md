# Awesome-Visual-Generation-Alignment-Survey
The collection of awesome papers on the alignment of visual generation models (including AR and diffusion models)

We also use this [repo](https://github.com/xie-lab-ml/awesome-alignment-of-diffusion-models) as a reference.

**We welcome community contributions!**

## Tutorial on Reinforcement Learning
- [CS285](https://rail.eecs.berkeley.edu/deeprlcourse/) from UC Berkeley. (This course is mainly for robotics control, but very important if you want to be an expert on RL.)
- [Introduction to Reinforcement Learning](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
- [动手学强化学习](https://github.com/boyu-ai/Hands-on-RL) (in Chinese)
- [强化学习知乎专栏](https://www.zhihu.com/column/reinforcementlearning) (in Chinese)

## First Two Works in Each Subfield:
**Traditional Policy Gradient**
- DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models, [[pdf]](https://arxiv.org/pdf/2305.16381), 2023.05
- Training Diffusion Models with Reinforcement Learning, [[pdf]](https://arxiv.org/abs/2305.13301), 2023.05

**GRPO**
- DanceGRPO: Unleashing GRPO on Visual Generation, [[pdf]](https://arxiv.org/pdf/2505.07818), 2025.05
- Flow-GRPO: Training Flow Matching Models via Online RL, [[pdf]](https://arxiv.org/pdf/2505.05470), 2025.05

**DPO**
- Diffusion Model Alignment Using Direct Preference Optimization, [[pdf]](https://arxiv.org/pdf/2311.12908), 2023.11
- Using Human Feedback to Fine-tune Diffusion Models without Any Reward Model, [[pdf]](https://arxiv.org/pdf/2311.13231), 2023.11

**Reward Feedback Learning**
- ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation, [[pdf]](https://arxiv.org/pdf/2304.05977), 2023.04
- Directly Fine-Tuning Diffusion Models on Differentiable Rewards, [[pdf]](https://arxiv.org/pdf/2309.17400), 2023.09

**Alignment on AR models**
- Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step [[pdf]](https://arxiv.org/abs/2501.13926), 2025.01
- Autoregressive Image Generation Guided by Chains of Thought [[pdf]](https://arxiv.org/pdf/2502.16965), 2025.02

**Reward Models**
- ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation, [[pdf]](https://arxiv.org/pdf/2304.05977), 2023.04
- Human Preference Score: Better Aligning Text-to-Image Models with Human Preference, [[pdf]](https://arxiv.org/pdf/2303.14420), 2023.03


## Alignment on Diffusion/Flow Models
### Reinforcement Learning-based (RLHF)
- DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models, [[pdf]](https://arxiv.org/pdf/2305.16381), 2023.05
- Training Diffusion Models with Reinforcement Learning, [[pdf]](https://arxiv.org/abs/2305.13301), 2023.05
- Score as Action: Fine-Tuning Diffusion Generative Models by Continuous-time Reinforcement Learning, [[pdf]](https://arxiv.org/pdf/2502.01819), 2025.02
- Towards Better Alignment: Training Diffusion Models with Reinforcement Learning Against Sparse Rewards, [[pdf]](https://arxiv.org/pdf/2501.06655), 2025.01
- Schedule On the Fly: Diffusion Time Prediction for Faster and Better Image Generation, [[pdf]](https://arxiv.org/pdf/2412.01243v2), 2024.12
- DanceGRPO: Unleashing GRPO on Visual Generation, [[pdf]](https://arxiv.org/pdf/2505.07818), 2025.05
- Flow-GRPO: Training Flow Matching Models via Online RL, [[pdf]](https://arxiv.org/pdf/2505.05470), 2025.05
- MixGRPO: Unlocking Flow-based GRPO Efficiency with Mixed ODE-SDE, [[pdf]](https://arxiv.org/pdf/2507.21802), 2025.07
- TempFlow-GRPO: When Timing Matters for GRPO in Flow Models, [[pdf]](https://arxiv.org/pdf/2508.04324), 2025.08
- Pref-GRPO: Pairwise Preference Reward-based GRPO for Stable Text-to-Image Reinforcement Learning, [[pdf]](https://arxiv.org/pdf/2508.20751), 2025.08
- DiffusionNFT: Online Diffusion Reinforcement with Forward Process, [[pdf]](https://arxiv.org/pdf/2509.16117), 2025.09
- BranchGRPO: Stable and Efficient GRPO with Structured Branching in Diffusion Models, [[pdf]](https://arxiv.org/pdf/2509.06040), 2025.09
- Advantage Weighted Matching: Aligning RL with Pretraining in Diffusion Models, [[pdf]](https://arxiv.org/pdf/2509.25050), 2025.09
- PCPO: Proportionate Credit Policy Optimization for Aligning Image Generation Models, [[pdf]](https://arxiv.org/pdf/2509.25774), 2025.09
- Coefficients-Preserving Sampling for Reinforcement Learning with Flow Matching, [[pdf]](https://arxiv.org/pdf/2509.05952), 2025.09
- Dynamic-TreeRPO: Breaking the Independent Trajectory Bottleneck with Structured Sampling, [[pdf]](https://arxiv.org/pdf/2509.23352), 2025.09
- G^2RPO: Granular GRPO for Precise Reward in Flow Models, [[pdf]](https://arxiv.org/pdf/2510.01982), 2025.10
- Reinforcing Diffusion Models by Direct Group Preference Optimization, [[pdf]](https://arxiv.org/pdf/2510.08425), 2025.10
- Smart-GRPO: Smartly Sampling Noise for Efficient RL of Flow-Matching Models, [[pdf]](https://arxiv.org/pdf/2510.02654), 2025.10
- Identity-GRPO: Optimizing Multi-Human Identity-preserving Video Generation via Reinforcement Learning, [[pdf]](https://arxiv.org/pdf/2510.14256), 2025.10
- Self-Forcing++: Towards Minute-Scale High-Quality Video Generation, [[pdf]](https://arxiv.org/pdf/2510.02283), 2025.10

### DPO-based (referred to [here](https://github.com/xie-lab-ml/awesome-alignment-of-diffusion-models))
- Diffusion Model Alignment Using Direct Preference Optimization, [[pdf]](https://arxiv.org/pdf/2311.12908), 2023.11
- Using Human Feedback to Fine-tune Diffusion Models without Any Reward Model, [[pdf]](https://arxiv.org/pdf/2311.13231), 2023.11
- A Dense Reward View on Aligning Text-to-Image Diffusion with Preference, [[pdf]](https://arxiv.org/pdf/2402.08265), 2024.02
- Aligning Diffusion Models by Optimizing Human Utility, [[pdf]](https://arxiv.org/pdf/2404.04465), 2024.04
- Boost Your Human Image Generation Model via Direct Preference Optimization, [[pdf]](https://arxiv.org/pdf/2405.20216), 2024.05
- Curriculum Direct Preference Optimization for Diffusion and Consistency Models, [[pdf]](https://arxiv.org/pdf/2405.13637), 2024.05
- Margin-aware Preference Optimization for Aligning Diffusion Models without Reference, [[pdf]](https://arxiv.org/pdf/2406.06424), 2024.06
- DSPO: Direct Score Preference Optimization for Diffusion Model Alignment, [[pdf]](https://openreview.net/forum?id=xyfb9HHvMe), 2024.09
- Diffusion-NPO: Negative Preference Optimization for Better Preference Aligned Generation of Diffusion Models, [[pdf]](https://openreview.net/pdf?id=iJi7nz5Cxc), 2024.09
- Tuning Timestep-Distilled Diffusion Model Using Pairwise Sample Optimization, [[pdf]](https://arxiv.org/pdf/2410.03190), 2024.10
- Scalable Ranked Preference Optimization for Text-to-Image Generation, [[pdf]](https://arxiv.org/pdf/2410.18013), 2024.10
- Prioritize Denoising Steps on Diffusion Model Preference Alignment via Explicit Denoised Distribution Estimation, [[pdf]](https://arxiv.org/pdf/2411.14871), 2024.11
- PatchDPO: Patch-level DPO for Finetuning-free Personalized Image Generation. arXiv 2024, [[pdf]](https://arxiv.org/pdf/2412.03177), 2024.12
- SafetyDPO: Scalable Safety Alignment for Text-to-Image Generation, [[pdf]](https://arxiv.org/pdf/2412.10493), 2024.12
- OnlineVPO: Align Video Diffusion Model with Online Video-Centric Preference Optimization, [[pdf]](https://arxiv.org/pdf/2412.15159), 2024.12
- VideoDPO: Omni-Preference Alignment for Video Diffusion Generation, [[pdf]](https://arxiv.org/pdf/2412.14167), 2024.12
- Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization, [[pdf]](https://arxiv.org/pdf/2406.04314), 2024.06
- Personalized Preference Fine-tuning of Diffusion Models, [[pdf]](https://arxiv.org/pdf/2501.06655), 2025.01
- Calibrated Multi-Preference Optimization for Aligning Diffusion Models, [[pdf]](https://arxiv.org/pdf/2502.02588), 2025.02
- Direct Distributional Optimization for Provable Alignment of Diffusion Models, [[pdf]](https://arxiv.org/pdf/2502.02954), 2025.02
- InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment, [[pdf]](https://arxiv.org/pdf/2503.18454), 2025.03
- CHATS: Combining Human-Aligned Optimization and Test-Time Sampling for Text-to-Image Generation, [[pdf]](https://arxiv.org/pdf/2502.12579), 2025.02
- Refining Alignment Framework for Diffusion Models with Intermediate-Step Preference Ranking, [[pdf]](https://arxiv.org/pdf/2502.01667), 2025.02
- Aligning Text to Image in Diffusion Models is Easier Than You Think, [[pdf]](https://arxiv.org/pdf/2503.08250), 2025.03
- DreamDPO: Aligning Text-to-3D Generation with Human Preferences via Direct Preference Optimization, [[pdf]](https://arxiv.org/abs/2502.04370), 2025.02
- Flow-DPO: Improving Video Generation with Human Feedback, [[pdf]](https://arxiv.org/abs/2501.13918), 2025.01
- HuViDPO: Enhancing Video Generation through Direct Preference Optimization for Human-Centric Alignment, [[pdf]](https://arxiv.org/pdf/2502.01690), 2025.02
- Fine-Tuning Diffusion Generative Models via Rich Preference Optimization, [[pdf]](https://arxiv.org/pdf/2503.11720), 2025.03
- D-Fusion: Direct Preference Optimization for Aligning Diffusion Models with Visually Consistent Samples, [[pdf]](https://arxiv.org/abs/2505.22002), 2025.05
- Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences, [[pdf]](https://arxiv.org/abs/2506.02698), 2025.06
- Follow-Your-Preference: Towards Preference-Aligned Image Inpainting, [[pdf]](https://arxiv.org/pdf/2509.23082), 2025.09
- Towards Better Optimization For Listwise Preference in Diffusion Models, [[pdf]](https://arxiv.org/pdf/2510.01540), 2025.10

### Reward Feedback Learning
- ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation, [[pdf]](https://arxiv.org/pdf/2304.05977), 2023.04
- Aligning Text-to-Image Diffusion Models with Reward Backpropagation, [[pdf]](https://arxiv.org/pdf/2310.03739v2), 2023.10
- Directly Fine-Tuning Diffusion Models on Differentiable Rewards, [[pdf]](https://arxiv.org/pdf/2309.17400), 2023.09
- Feedback Efficient Online Fine-Tuning of Diffusion Models, [[pdf]](https://arxiv.org/pdf/2402.16359), 2024.02
- Deep Reward Supervisions for Tuning Text-to-Image Diffusion Models, [[pdf]](https://arxiv.org/abs/2405.00760), 2024.05
- Reward Fine-Tuning Two-Step Diffusion Models via Learning Differentiable Latent-Space Surrogate Reward, [[pdf]](https://arxiv.org/pdf/2411.15247), 2024.11
- InstructVideo: Instructing Video Diffusion Models with Human Feedback, [[pdf]](https://arxiv.org/abs/2312.12490), 2023.12
- IterComp: Iterative Composition-Aware Feedback Learning from Model Gallery for Text-to-Image Generation, [[pdf]](https://arxiv.org/abs/2410.07171), 2024.10
- Directly Aligning the Full Diffusion Trajectory with Fine-Grained Human Preference, [[pdf]](https://arxiv.org/pdf/2509.06942), 2025.09
- Identity-Preserving Image-to-Video Generation via Reward-Guided Optimization, [[pdf]](https://arxiv.org/pdf/2510.14255), 2025.10


### Technical Reports 
We only list the reports using alignment methods:
- Step-Video-T2V Technical Report: The Practice, Challenges, and Future of Video Foundation Model, [[pdf]](https://arxiv.org/pdf/2502.10248), 2025.02
- Seedream 2.0: A native chinese-english bilingual image generation foundation model, [[pdf]](https://arxiv.org/pdf/2503.07703), 2025.03
- Seedream 3.0 technical report, [[pdf]](https://arxiv.org/pdf/2504.11346), 2025.04
- Seedance 1.0: Exploring the Boundaries of Video Generation Models, [[pdf]](https://arxiv.org/pdf/2506.09113), 2025.06
- Seedream 4.0: Toward Next-generation Multimodal Image Generation, [[pdf]](https://arxiv.org/pdf/2509.20427), 2025.09
- Qwen-Image Technical Report, [[pdf]](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/Qwen_Image.pdf), 2025.08
- Skywork-UniPic2, [[pdf]](https://github.com/SkyworkAI/UniPic/blob/main/UniPic-2/assets/pdf/UNIPIC2.pdf), 2025.09
- BLIP3o-NEXT: Next Frontier of Native Image Generation, [[pdf]](https://arxiv.org/pdf/2510.15857), 2025.10

## Alignment on AR models (referred to [here](https://github.com/ChaofanTao/Autoregressive-Models-in-Vision-Survey))
- Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step, [[pdf]](https://arxiv.org/abs/2501.13926), 2025.01
- Autoregressive Image Generation Guided by Chains of Thought, [[pdf]](https://arxiv.org/pdf/2502.16965), 2025.02
- LightGen: Efficient Image Generation through Knowledge Distillation and Direct Preference Optimization, [[pdf]](https://arxiv.org/abs/2503.08619), 2025.03
- SimpleAR: Pushing the Frontier of Autoregressive Visual Generation through Pretraining, SFT, and RL, [[pdf]](https://arxiv.org/pdf/2504.11455), 2025.04
- UniGen: Enhanced Training & Test-Time Strategies for Unified Multimodal Understanding and Generation, [[pdf]](https://arxiv.org/pdf/2505.14682), 2025.05
- UniRL: Self-Improving Unified Multimodal Models via Supervised and Reinforcement Learning, [[pdf]](https://arxiv.org/pdf/2505.23380), 2025.05
- ReasonGen-R1: CoT for Autoregressive Image Generation model through SFT and RL, [[pdf]](https://arxiv.org/pdf/2505.24875), 2025.05
- Unlocking Aha Moments via Reinforcement Learning: Advancing Collaborative Visual Comprehension and Generation, [[pdf]](https://arxiv.org/pdf/2506.01480), 2025.06
- Delving into RL for Image Generation with CoT: A Study on DPO vs. GRPO, [[pdf]](https://arxiv.org/abs/2505.17017), 2025.05
- CoT-lized Diffusion: Let’s Reinforce T2I Generation Step-by-step, [[pdf]](https://arxiv.org/pdf/2507.04451), 2025.07
- X-Omni: Reinforcement Learning Makes Discrete Autoregressive Image Generative Models Great Again, [[pdf]](https://arxiv.org/pdf/2507.22058), 2025.07
- T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT, [[pdf]](https://arxiv.org/pdf/2505.00703), 2025.05
- AR-GRPO: Training Autoregressive Image Generation Models via Reinforcement Learning, [[pdf]](https://arxiv.org/pdf/2508.06924), 2025.08
- Group Critical-token Policy Optimization for Autoregressive Image Generation, [[pdf]](https://arxiv.org/pdf/2509.22485), 2025.09
- STAGE: Stable and Generalizable GRPO for Autoregressive Image Generation, [[pdf]](https://arxiv.org/pdf/2509.25027), 2025.09
- Layout-Conditioned Autoregressive Text-to-Image Generation via Structured Masking, [[pdf]](https://arxiv.org/abs/2509.12046), 2025.09


## Benchmarks & Reward Models
**Benchmarks**
- DALL-Eval: Probing the Reasoning Skills and Social Biases of Text-to-Image Generative Transformers, [[pdf]](https://arxiv.org/pdf/2202.04053), 2022.02
- Human Evaluation of Text-to-Image Models on a Multi-Task Benchmark, [[pdf]](https://arxiv.org/pdf/2211.12112), 2022.11
- LLMScore: Unveiling the Power of Large Language Models in Text-to-Image Synthesis Evaluation, [[pdf]](https://arxiv.org/pdf/2305.11116), 2023.05
- VPGen & VPEval: Visual Programming for Text-to-Image Generation and Evaluation, [[pdf]](https://arxiv.org/pdf/2305.15328), 2023.05
- T2I-CompBench: An Enhanced and Comprehensive Benchmark for Compositional Text-to-Image Generation, [[pdf]](https://arxiv.org/abs/2307.06350v1), 2023.07
- GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment, [[pdf]](https://arxiv.org/pdf/2310.11513), 2023.10
- Holistic Evaluation of Text-to-Image Models, [[pdf]](https://arxiv.org/pdf/2311.04287), 2023.11
- Social Reward: Evaluating and Enhancing Generative AI through Million-User Feedback from an Online Creative Community, [[pdf]](https://arxiv.org/pdf/2402.09872), 2024.02
- Rich Human Feedback for Text to Image Generation, [[pdf]](https://arxiv.org/pdf/2312.10240), 2023.12
- Learning Multi-Dimensional Human Preference for Text-to-Image Generation, [[pdf]](https://arxiv.org/pdf/2405.14705), 2024.05
- Evaluating Text-to-Visual Generation with Image-to-Text Generation, [[pdf]](https://arxiv.org/pdf/2404.01291), 2024.04
- Multimodal Large Language Models Make Text-to-Image Generative Models Align Better, [[pdf]](https://arxiv.org/abs/2404.15100), 2024.04
- Measuring Style Similarity in Diffusion Models, [[pdf]](https://arxiv.org/pdf/2404.01292), 2024.04
- DreamBench++: A Human-Aligned Benchmark for Personalized Image Generation, [[pdf]](https://arxiv.org/pdf/2406.16855), 2024.06
- PAL: Sample-Efficient Personalized Reward Modeling for Pluralistic Alignment, [[pdf]](https://openreview.net/pdf?id=1kFDrYCuSu), 2024.06
- Video-Bench: Human-Aligned Video Generation Benchmark, [[pdf]](https://arxiv.org/pdf/2504.04907), 2025.04
- ImageDoctor: Diagnosing Text-to-Image Generation via Grounded Image Reasoning, [[pdf]](https://arxiv.org/pdf/2510.01010), 2025.10

**Reward Models**
- Human Preference Score: Better Aligning Text-to-Image Models with Human Preference, [[pdf]](https://arxiv.org/pdf/2303.14420), 2023.03
- ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation, [[pdf]](https://arxiv.org/pdf/2304.05977), 2023.04
- Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation, [[pdf]](https://arxiv.org/pdf/2305.01569), 2023.05
- Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis, [[pdf]](https://arxiv.org/pdf/2306.09341), 2023.06
- Improving Video Generation with Human Feedback, [[pdf]](https://arxiv.org/pdf/2501.13918), 2025.01
- VideoScore: Building Automatic Metrics to Simulate Fine-grained Human Feedback for Video Generation, [[pdf]](https://arxiv.org/pdf/2406.15252), 2024.06
- VisionReward: Fine-Grained Multi-Dimensional Human Preference Learning for Image and Video Generation, [[pdf]](https://arxiv.org/pdf/2412.21059), 2024.12
- Unified Reward Model for Multimodal Understanding and Generation, [[pdf]](https://arxiv.org/pdf/2503.05236), 2025.03
- LiFT: Leveraging Human Feedback for Text-to-Video Model Alignment, [[pdf]](https://arxiv.org/pdf/2412.04814), 2024.12
- HPSv3: Towards Wide-Spectrum Human Preference Score, [[pdf]](https://arxiv.org/pdf/2508.03789), 2025.08
- Rewarddance: Reward scaling in visual generation, [[pdf]](https://arxiv.org/pdf/2509.08826), 2025.09
- Unlocking the Essence of Beauty: Advanced Aesthetic Reasoning with Relative-Absolute Policy Optimization, [[pdf]](https://arxiv.org/pdf/2509.21871), 2025.09
- EditScore: Unlocking Online RL for Image Editing via High-Fidelity Reward Modeling, [[pdf]](https://arxiv.org/pdf/2509.23909), 2025.09
- VideoScore2: Think before You Score in Generative Video Evaluation, [[pdf]](https://arxiv.org/abs/2509.22799), 2025.09
- EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing, [[pdf]](https://arxiv.org/abs/2509.26346), 2025.09

## Alignment with Prompt Engineering
- Optimizing Prompts for Text-to-Image Generation, [[pdf]](https://arxiv.org/pdf/2212.09611), 2022.12
- RePrompt: Automatic Prompt Editing to Refine AI-Generative Art Towards Precise Expressions, [[pdf]](https://arxiv.org/pdf/2302.09466), 2023.02
- Improving Text-to-Image Consistency via Automatic Prompt Optimization, [[pdf]](https://arxiv.org/pdf/2403.17804), 2024.03
- Dynamic Prompt Optimizing for Text-to-Image Generation, [[pdf]](https://arxiv.org/pdf/2404.04095), 2024.04
- T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT, [[pdf]](https://arxiv.org/pdf/2505.00703), 2025.05
- RePrompt: Reasoning-Augmented Reprompting for Text-to-Image Generation via Reinforcement Learning, [[pdf]](https://arxiv.org/pdf/2505.17540?), 2025.05
