# Q2 — Literature Review

## 1. Design intent 
### YOLOv8n
YOLOv8n is the nano member of the YOLOv8 family, released by Ultralytics in 2023. The family spans five scales from nano to extra-large, and only the nano is realistically an edge model. What changed from YOLOv5 was mainly the head. Earlier YOLO generations predicted boxes as offsets from predefined anchors, which meant retuning anchor sizes for each dataset and living with poor recall on shapes the anchors did not cover. YOLOv8 drops anchors and regresses box centres directly, and it splits classification and localisation into separate branches so the two tasks stop competing for the same features [1]. The backbone stays CSPNet with an FPN+PAN neck, but the C3 block is replaced by C2f, which routes more gradient paths through each stage at roughly the same cost [1].
Much of YOLOv8's adoption has little to do with architecture. Ultralytics put detection, segmentation, pose estimation and classification behind one Python API and CLI [2], and that convenience is a large part of why it became the default baseline in so much subsequent work. One caveat: there is no paper. YOLOv8 is described in vendor documentation [2] and analysed after the fact by third parties [1], so its claims cannot be checked against peer review.


### YOLO11n
YOLO11 arrived in September 2024, and what makes it interesting is what it did not try to do. It is not a large accuracy jump. It is a rebalancing: YOLO11n reaches 39.5 mAP against YOLOv8n's 37.3 while using fewer parameters, 2.6 M against 3.2 M, and fewer FLOPs, 6.5 B against 8.7 B [4]. On a device running off a battery, that trade matters more than the two mAP points do.
Two blocks carry most of the change. C3k2, a cross-stage partial block with smaller kernels, replaces YOLOv8's C2f, and a C2PSA attention block is inserted after the SPPF stage so the network can weight spatially relevant regions at the deepest feature level before the neck fuses scales [3]. The attention block is the more significant of the two, since small objects are exactly what nano-scale detectors tend to lose in that fusion.
A naming note, because it causes confusion in the literature: Ultralytics dropped the "v" and brands this YOLO11, while most papers including [3] write YOLOv11. As with YOLOv8, Ultralytics published no paper of its own; [3] is a third-party analysis.

### RT-DETR (ResNet-18)
RT-DETR is the only one that set out to settle an argument between two families rather than improve one of them. DETR-style transformers were accurate and needed no NMS, but they were far too slow for real-time work. YOLOs were fast, but the authors make a sharper claim about NMS than is usual: it hurts them twice over, adding latency that grows with how crowded the scene is, and costing accuracy, because the confidence threshold controlling it trades recall against speed [5]. 
The core contribution is a hybrid encoder that splits the work in two. Self-attention runs only at the highest and smallest feature level, where it is cheapest, while cross-scale fusion is left to convolutions [5]. Decoder queries are initialised by uncertainty-minimal query selection rather than arbitrarily. One property gets less attention than it deserves: the number of decoder layers can be changed after training to trade speed against accuracy, with no retraining at all. On an edge project, where the latency budget usually becomes clear late, that is genuinely useful. The ResNet-18 variant reports 46.5 AP with 20 M parameters, 60.7 GFLOPs and 217 FPS on a T4 [6].

### YOLO-NAS-S
YOLO-NAS approaches the problem from a direction none of the others take. Rather than designing an architecture by hand, Deci searched for one. What makes the search interesting is the objective it optimised for. AutoNAC targeted quantization tolerance rather than FLOPs or accuracy alone, and it produced blocks, QSP and QCI, chosen because they survive INT8 conversion [7]. The numbers bear this out. YOLO-NAS-S falls from 47.5 to 47.03 mAP under INT8, losing less than half a point, where conventional detectors usually give up one to three [7].

For an edge target built around INT8-only accelerators, that is the most relevant property any model in this comparison offers. There is a large caveat. No peer-reviewed paper exists, only vendor material, and since NVIDIA acquired Deci the model hub has been retired, so the weight URLs built into SuperGradients no longer resolve. A model whose distribution channel has disappeared is hard to recommend for production whatever its technical merits.



### EfficientDet-Lite0
EfficientDet is older than everything else here, and it targets two inefficiencies in how detectors combine multi-scale features. Feature pyramid networks fused scales in one direction and treated every input as equally important, even though features at different resolutions do not contribute equally. Scaling was also ad-hoc, with depth, width and input resolution tuned separately. The paper contributes BiFPN, a bidirectional pyramid with learned weights on each input, and compound scaling that grows the backbone, feature network and prediction heads together from a single coefficient [9].

For this study the Lite variants matter more than the base model, and they exist for reasons that are entirely about deployment. Its EfficientNet-Lite backbone drops squeeze-and-excitation blocks, which are poorly supported on mobile accelerators, and replaces every swish activation with ReLU6, which makes post-training quantization far better behaved [13]. Lite0 also runs at 320x320 rather than 640. That is worth carrying through the rest of this report, because it is the only model here not evaluated at 640, so part of its apparent latency advantage comes from the smaller input rather than from the architecture.


### YOLO26n
YOLO26 appeared in September 2025 and aims at something different from its predecessors. Where YOLO11 tuned the backbone, YOLO26 goes after post-processing. Two things are removed rather than added: NMS, replaced by an end-to-end one-to-one detection head, and Distribution Focal Loss, leaving direct box regression [11]. Both removals are deployment decisions rather than accuracy ones. NMS costs time that scales with how crowded the scene is, needs a plugin to run under TensorRT, and is patchily supported on mobile delegates. A model without it has predictable latency and a much simpler export path.

Dropping DFL would normally cost accuracy, so the compensation happens during training: a progressive loss schedule, small-target-aware label assignment, and MuSGD, a hybrid of SGD and the Muon optimiser borrowed from language-model training [12]. The nano variant lands at 40.9 mAP with 2.4 M parameters and 5.4 GFLOPs, and Ultralytics reports up to 43% faster CPU ONNX inference than YOLO11n [12].

Both architectural claims are checkable in the shipped configuration, and I confirmed them directly: end2end: True and reg_max: 1, corresponding to the NMS-free head and the removal of DFL bins.

## 2. Comparative table

| Model | Architecture family | Params (M) | FLOPs | Anchors | NMS-free | Backbone | Head type |
|---|---|---|---|---|---|---|---|
| YOLOv8n | Single-stage CNN | 3.2 [2] | 8.7 G @640 [2] | anchor-free | no | CSPDarknet, C2f blocks [1] | decoupled, DFL (reg_max 16) |
| YOLO11n | Single-stage CNN | 2.6 [4] | 6.5 G @640 [4] | anchor-free | no | C3k2 + SPPF + C2PSA [3] | decoupled, DFL |
| RT-DETR-R18 | Transformer, DETR family | 20 [6] | 60.7 G @640 [6] | anchor-free | yes | ResNet-18 (vd), depths [2,2,2,2] <sup>a</sup> [5] | DETR decoder, 300 queries <sup>a</sup> |
| YOLO-NAS-S | NAS-designed single-stage CNN | 19.05 <sup>b</sup> | 33.9 G @640 <sup>b,c</sup> | anchor-free | no | NAS-searched, QSP/QCI blocks [7] | decoupled, DFL |
| EfficientDet-Lite0 | Single-stage CNN + BiFPN | 3.2 [10] | 1.95 G @320 <sup>b,d</sup> | anchor-based | no | EfficientNet-Lite0 [10] | shared class/box heads |
| YOLO26n | Single-stage CNN, end-to-end | 2.4 [12] | 5.4 G @640 [12] | anchor-free | yes | C3k2 + SPPF + C2PSA [12] | one-to-one, DFL removed (reg_max 1) <sup>a</sup> |

<sup>a</sup> Confirmed by inspecting the shipped model configuration in this project, in addition to the cited source.

<sup>b</sup> Measured in this project using PyTorch's `FlopCounterMode`, which counts at the dispatcher level and therefore includes custom layer types that module-hook profilers such as `thop` silently skip. The method reproduces published figures to within 1% (YOLO11n: 6.54 G measured against 6.5 G published), which is what validates the two measured entries. Parameter counts are taken post Conv+BatchNorm fusion, matching the convention used in the official model tables.

<sup>c</sup> Deci publish no FLOPs, parameter count or model size for YOLO-NAS-S, and the latency figure they do give (3.21 ms) states no hardware. The value here was measured in this project, inside the isolated Python 3.10 environment that model requires.

<sup>d</sup> EfficientDet-Lite0 runs natively at 320x320. A 640 figure is omitted deliberately, since the model is neither designed nor trained for that resolution and the number would be misleading in a column headed @640.



## 3. Strengths & weaknesses 
### YOLOv8n
Strengths
- Anchor-free head, so there is no anchor tuning when moving to a new dataset [2].
- Classification and box regression sit in separate branches instead of sharing one [1].

Weaknesses
- The heaviest of the three nano YOLOs here, 3.2 M parameters and 8.7 GFLOPs. YOLO11n beats it on accuracy and cost at the same time [2], [4].
- Needs NMS, so post-processing time depends on how many objects are in the frame [1].

### YOLO11n
Strengths
- More accurate than YOLOv8n while being smaller: 39.5 mAP against 37.3, at 2.6 M parameters against 3.2 M [4].
- The C2PSA block adds attention at P5, before the neck fuses scales [3].

Weaknesses
- Still needs NMS [3].
- No paper from Ultralytics, so the architecture is only described in vendor material and third-party analysis [3], [4].

### RT-DETR (ResNet-18)
Strengths
- No NMS. Boxes come straight out of set prediction, so post-processing time is fixed [5].
- Decoder layers can be dropped after training to trade accuracy for speed, with no retraining [5].

Weaknesses
- 60.7 GFLOPs and 20 M parameters, roughly nine times YOLO11n's compute. This is what rules it out on CPU-only targets [6].
- Portability. It was the only model here that would not run on Apple's MPS backend at all, because `transformers` builds its position embedding in float64 and MPS has no float64. The four CNN detectors all ran unchanged.

### YOLO-NAS-S
Strengths
- Barely loses accuracy under INT8, 47.5 down to 47.03, where most detectors give up one to three points [7].
- Highest fp32 accuracy of the six at 47.5 [7].

Weaknesses
- Deci publish no parameter count, no FLOPs and no model size, and the single latency figure they give does not say what hardware it ran on [7].
- The weight host has been offline since NVIDIA acquired Deci, so the URLs compiled into SuperGradients no longer resolve. That is why this model needed its own pinned environment here.

### EfficientDet-Lite0
Strengths
- Smallest model in the set: 3.2 M parameters and 1.95 GFLOPs at its native 320x320 [9].
- The EfficientNet-Lite backbone drops squeeze-and-excite, which mobile accelerators support poorly, and uses ReLU6 instead of swish, which quantizes much better [13].

Weaknesses
- Least accurate of the six, 26.4 mAP against YOLO11n's 39.5 [10].
- Low FLOPs did not make it fast. It measured 470 ms on CPU, second-slowest of the six. Depthwise-separable convolutions are limited by memory bandwidth rather than compute, which is the likely cause, though the timing also includes the effdet bench's own post-processing, so the two are not separated here.
- The only anchor-based model in the set [9].

### YOLO26n
Strengths
- No NMS, so post-processing time does not change with the number of objects and export is simpler [11].
- Dropping DFL leaves the simplest head here, and it still reaches the highest nano accuracy at 40.9 mAP and 5.4 GFLOPs [12].

Weaknesses
- Training needs ProgLoss, small-target-aware assignment and the MuSGD optimiser, which is harder to debug when it does not converge [11].
- Released September 2025, so its export and quantization paths are less exercised than YOLO11n's. That is why I carried YOLO11n into Q5 rather than this.



## 4. Deployment suitability
### (a) Jetson Orin Nano 8GB
The Jetson Orin Nano 8GB features an NVIDIA Ampere GPU with 1024 CUDA cores and 32 Tensor Cores, delivering up to 40 TOPS (the later Orin Nano Super raises this to 67 TOPS via a software update). This environment heavily favors models that can be compiled into FP16 or INT8 TensorRT engines to maximize parallel GPU utilization.
- **Top performers:** YOLO-NAS-S is practically built for this hardware. Its architecture was discovered via Neural Architecture Search optimised for INT8 quantization, and the payoff is measurable: it loses only 0.47 mAP under INT8, falling from 47.5 to 47.03 [7], where conventional detectors give up one to three points. On a device whose tensor cores are fastest in INT8, that tolerance is worth more than it looks. YOLO26n and YOLO11n also excel here; exporting to TensorRT yields large throughput gains, which my own [Q5](../Assignment2.ipynb) measurements confirm at 15.90 ms down to 7.60 ms for FP16, a 2.1x speedup at no accuracy cost. YOLO26n's NMS-free design additionally removes post-processing that runs on the CPU regardless of how fast the GPU is. RT-DETR-R18 remains viable because 8 GB of unified memory comfortably absorbs the larger footprint of its transformer attention layers.

- **Sub-optimal:** EfficientDet-Lite0. It will run, but its architecture relies on depthwise-separable convolutions, which are heavily optimised for mobile CPUs and do not scale efficiently across massively parallel CUDA cores. The Orin's GPU ends up underutilised, and my CPU benchmark shows the same effect from the other direction: the lowest FLOPs of all six models at 1.95 G, yet the second-slowest measured latency.


### (b) Raspberry Pi 5 (CPU-only)
Powered by a quad-core ARM Cortex-A76, a stock Raspberry Pi 5 lacks a dedicated neural processing unit (NPU). Inference here relies entirely on raw CPU cycles, executing models typically via TensorFlow Lite or ONNX Runtime.
- **Top performers:** YOLO11n and YOLO26n. With no GPU path everything falls back to the Cortex-A76, and these two measured 87 ms and 88 ms in my [Q3](Q3_model_selection_criteria.md) CPU benchmark. That benchmark ran on Apple Silicon, a considerably faster CPU than the Pi's Cortex-A76, so treat these as optimistic lower bounds rather than Pi figures; even so they are the only numbers in the set anywhere near usable. YOLO26n has a second advantage here that it does not have on GPU: NMS runs on the CPU regardless of accelerator, so removing it removes a cost that would otherwise sit directly in the critical path. YOLOv8n is workable at 112 ms but is strictly dominated by YOLO11n on both accuracy and speed.

- **Sub-optimal:** RT-DETR-R18 (434 ms) and YOLO-NAS-S (533 ms). At 60.7 G and 33.9 GFLOPs they carry 9x and 5x YOLO11n's compute, and without a GPU there is nothing to hide it behind. Neither is viable for real-time work on this class of hardware.

- **The surprise:** EfficientDet-Lite0 also lands in the sub-optimal group, despite having the lowest FLOPs of all six at 1.95 G. It measured 470 ms, second-slowest overall. Depthwise-separable convolutions have low arithmetic intensity and are bound by memory bandwidth rather than compute, so they only pay off on hardware with the memory subsystem to suit them. On a general-purpose CPU the FLOPs advantage does not survive.


### (c) Mid-range Android (NNAPI/TFLite)
Deployment on a mid-range Android phone depends entirely on graph compatibility. To achieve real-time performance without draining the battery, the model must export cleanly to TensorFlow Lite (TFLite) and delegate its operations successfully to the Android Neural Networks API (NNAPI) to leverage the phone's integrated NPU or mobile GPU.

- **Top performers:** EfficientDet-Lite0, which was designed for precisely this target. The Lite variants exist because squeeze-and-excitation is poorly supported by mobile accelerators and swish quantizes badly, so the first was removed and the second replaced with ReLU6 [13], and the model ships natively as a TFLite graph. YOLO26n is the other strong fit, for a different reason: NMS is either unsupported or falls back to CPU on most delegates, which erases the acceleration you went to NNAPI for. An NMS-free head sidesteps that entirely, and on this profile that matters more than its 1.4 mAP advantage over YOLO11n.

- **Workable with caveats:** YOLOv8n and YOLO11n export to TFLite through Ultralytics, but both carry the NMS problem above, so expect part of the pipeline to drop back to CPU.

- **Sub-optimal:** RT-DETR-R18. Transformer attention has patchy NNAPI operator coverage, and I hit a concrete instance of this class of problem in my own work: its float64 position embedding made it the only model in this study that would not run on Apple's MPS backend at all. Precision and operator assumptions that hold on CUDA do not transfer to mobile accelerators. YOLO-NAS-S is impractical here for toolchain reasons rather than architectural ones.

## References  [IEEE style]

[1] M. Yaseen, "What is YOLOv8: An In-Depth Exploration of the Internal Features of the Next-Generation Object Detector," arXiv preprint arXiv:2408.15857, 2024.

[2] G. Jocher, A. Chaurasia, and J. Qiu, Ultralytics YOLOv8 (Version 8.0.0) [Software]. 2023. Available: https://github.com/ultralytics/ultralytics.

[3] R. Khanam and M. Hussain, "YOLOv11: An Overview of the Key Architectural Enhancements," arXiv preprint arXiv:2410.17725, 2024.

[4] G. Jocher and J. Qiu, Ultralytics YOLO11 (Version 11.0.0) [Software]. 2024. Available: https://github.com/ultralytics/ultralytics.

[5] Y. Zhao, W. Lv, S. Xu, J. Wei, G. Wang, Q. Dang, Y. Liu, and J. Chen, "DETRs Beat YOLOs on Real-time Object Detection," arXiv:2304.08069, Apr. 2023. (Proc. IEEE/CVF CVPR, 2024.)

[6] PekingU, "rtdetr_r18vd," Hugging Face model card. [Online]. Available: https://huggingface.co/PekingU/rtdetr_r18vd. [Accessed: Aug. 12, 2026].

[7] Deci-AI, "YOLO-NAS," GitHub, 2023. [Online]. Available: https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md. 

[8] S. Aharon, Louis-Dupont, O. Masad, K. Yurkova, L. Fridman, Lkdci, E. Khvedchenya, R. Rubin, N. Bagrov, B. Tymchenko, T. Keren, A. Zhilko, and Eran-Deci, Super-Gradients [Software]. GitHub, 2021. doi: 10.5281/ZENODO.7789328. Available: https://zenodo.org/records/7789328.

[9] M. Tan, R. Pang, and Q. V. Le, "EfficientDet: Scalable and Efficient Object Detection," arXiv preprint arXiv:1911.09070, 2020.

[10] Google Research, AutoML: EfficientDet [Software]. GitHub, 2020. Available: https://github.com/google/automl/tree/master/efficientdet.

[11] G. Jocher, J. Qiu, M. Liu, S. Lyu, F. C. Akyon, and M. E. Kalfaoglu, "Ultralytics YOLO26: Unified real-time end-to-end vision models," arXiv:2606.03748, 2026. doi: 10.48550/arXiv.2606.03748.

[12] Ultralytics, "Ultralytics YOLO26," Ultralytics YOLO Docs, 2026. [Online]. Available: https://docs.ultralytics.com/models/yolo26.

[13] TensorFlow, "Higher accuracy on vision models with EfficientNet-Lite," TensorFlow Blog, Mar. 2020. [Online]. Available: https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html. [Accessed: Aug. 12, 2026].