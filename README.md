# Pseudo Chain of Thought Induced Fine-Tuning for Large Language Models

## Overview

The paper introduces a novel framework for improving reasoning in Large Language Models (LLMs) by inducing a Pseudo Chain of Thought (CoT). The approach consists of fine-tuning a base LLM and integrating a distilled Tiny Thought Model (TTM) through a router mechanism. This setup enhances reasoning capabilities while maintaining low inference costs.

## Key Contributions

Chain of Thought Induction: Introduces a method to enable step-by-step reasoning without manual CoT prompting.

Two-Step Knowledge Distillation: Reduces an 8B+ LLM to a 1B parameter Tiny Thought Model while retaining reasoning capabilities.

Routing Mechanism: Dynamically determines whether to generate intermediate thoughts for better answer alignment.

Fine-Tuning with LoRA: Optimizes resource efficiency while training the base LLM.

Synthetic Dataset Generation: Uses large LLMs to generate 5K-10K Q&A pairs from scientific and mathematical sources.

Evaluation Metrics: Assessed using Accuracy, BLEU, and ROUGE on MMLU datasets.

## Model Architecture

Base LLM: Fine-tuned using LoRA on domain-specific data (math and coding).

Tiny Thought Model (TTM): Distilled from the base model to 1B parameters for generating intermediate reasoning steps.

Router: A neural network that decides whether additional reasoning is required.

Integration: The router invokes the TTM iteratively to refine answers before returning final responses.

![image](https://github.com/user-attachments/assets/fdfb0448-6ea6-4ad9-9d98-0854b4d58b58)
![image](https://github.com/user-attachments/assets/f721a51e-770d-4b8a-a198-0679f0239215)


## Results

Mathematical Reasoning: Achieved a 10% accuracy increase over the base model without CoT.

Improved Explainability: Higher BLEU and ROUGE scores indicate more coherent and structured reasoning.

Competitive Performance: Outperforms models like Mistral 7B v0.3 and Gemma 2 9B in MMLU College-level evaluations.

![image](https://github.com/user-attachments/assets/a0d1d0b6-74d6-4b55-a814-ba2fc21a56bc)
### Comparison between Llama 3.1 8B and Pseudo CoT induced Llama 3.1 8B on MMLU dataset for college mathematics and college computer science

![image](https://github.com/user-attachments/assets/20de8c03-2e6b-4c88-ace4-e4c22595631c)
### Comparison of MMLU College Computer Science Scores with other models

![image](https://github.com/user-attachments/assets/68fd73d0-2eee-4078-8272-82b55b336ac2)
### Comparison of MMLU College Mathematics Scores with other models

![image](https://github.com/user-attachments/assets/ac9a7ad9-a7a3-4d8c-9361-777111d5d041)
### Example Output of CoT induced LLM


## Future Directions

Scaling the TTM: Exploring 3B+ parameter models for improved depth.

Optimized Routing: Adaptive reasoning depth based on task complexity.

Extended Domain Specialization: Applying CoT techniques to diverse subject areas.

Model Availability

The models trained as part of this research are available on Hugging Face:

Tiny Thought Model: [Llama 3.2 1B Distilled](https://huggingface.co/pr4nav101/Tiny-Thought-Model-Distilled-Llama-3.2-1B-Instruct-bnb-4bit)

Main LLM: [Llama 3.1 8B Math Finetuned](https://huggingface.co/pr4nav101/Llama-3.1-8B-4bit-bnb-Math-Finetuned)
