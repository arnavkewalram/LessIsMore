# LessIsMore

Catastrophic forgetting poses a fundamental challenge in continual learning, partic-1
ularly when models are quantized for deployment efficiency. We systematically2
investigate the interplay between quantization precision (FP16, INT8, INT4) and3
replay buffer strategies in large language models, revealing unexpected dynamics.4
While FP16 achieves superior initial task performance (74.44% on NLU), we ob-5
serve a striking inversion on subsequent tasks: quantized models outperform FP166
by 8-15% on final task forward accuracy, with INT4 achieving nearly double FP16’s7
performance on Code generation (40% vs 20%). Critically, even minimal replay8
buffers (0.1%) dramatically improve retention—increasing NLU retention after9
Math training from 45% to 65% across all precision levels—with INT8 consistently10
achieving the optimal balance between learning plasticity and knowledge retention.11
We hypothesize that quantization-induced noise acts as implicit regularization,12
preventing the overfitting to new task gradients that plagues high-precision models.13
These findings challenge the conventional wisdom that higher precision is always14
preferable, suggesting instead that INT8 quantization offers both computational15
efficiency and superior continual learning dynamics. Our results provide practical16
guidelines for deploying compressed models in continual learning scenarios: small17
replay buffers (1-2%) suffice for NLU tasks, while Math and Code benefit from18
moderate buffers (5-10%), with quantized models requiring less replay than FP1619
to achieve comparable retention. 
