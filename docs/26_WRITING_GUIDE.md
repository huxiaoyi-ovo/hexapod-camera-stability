# Writing guide for the research manuscript

## One manuscript entry point

`paper/main.tex` is the only manuscript entry point. It owns the title, abstract, document metadata, and section order; section files are included from it. Build only from `paper/main.tex`, regenerate `paper/main.pdf` from that source, and review the PDF after content changes. Keep author details, funding, declarations, references, and quantitative results unresolved until verified; do not add boilerplate or plausible values.

## Mark executable status precisely

The paper describes a proposed compensation-aware gait method. The current executable is a standalone flat-ground stand diagnostic with an uncalibrated PD controller. It does not execute gait generation, walking IK, body compensation, delay identification, camera calibration, prediction, scheduling, paired ON/OFF collection, or comparative evaluation. Use “proposed”, “planned”, or “will evaluate” for these components until their code and tests support “implemented”; use “measured” only for a trace or instrumented observation that actually contains that quantity.

The current stand trace is a simulator bring-up diagnostic, not a result for gait adaptation or camera stability. Do not use its gate status, contacts, or body motion as comparative study evidence. Synthetic examples and unit tests can verify software behavior but cannot establish a locomotion or image-stability mechanism.

## Align each section to the evidence

- **Introduction:** state the research question as a hypothesis and avoid presenting generic gait adaptation or body stabilization as novel. Do not call the planned compensator an existing project controller.
- **Method:** distinguish formal method specification from implemented code. Name dependencies, decision-time inputs, outputs, timing, reset behavior, and all unimplemented blocks. No future contacts or other-rollout outcomes may leak into online inputs.
- **Experiments:** separate planned protocols from completed trials. Specify the local asset provenance and independent runtime. Record exact configs, code and asset hashes, full initial states, seeds, sampled randomization, simulator versions, failures, and achieved speed. Pair ON/OFF trials by the same planned conditions while allowing their trajectories to diverge.
- **Results:** report only evidence from qualified runs. Label the stand diagnostic as operational evidence outside the study results; do not write a comparative walking/camera result until those experiments exist. Retain negative, failed, and inconclusive outcomes in the study record.
- **Discussion and conclusions:** bound every statement to the evaluated terrain, speed, controller, image metric, and hardware conditions. Do not generalize from GPU simulation to the physical robot.

Use [the architecture and experiment contract](25_ARCHITECTURE_AND_EXPERIMENT_CONTRACT.md) for the runtime and reproducibility boundary, and [the refactor verification record](27_REFACTOR_VERIFICATION.md) for the current verified state. Preserve the working draft and data-required markers; this guide does not authorize submission or imply that unresolved claims are verified.
