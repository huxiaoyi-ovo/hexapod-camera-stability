# Manuscript draft skeleton — Drones

**Working title:** Compensation-Aware Gait Adaptation for Camera-Stable Hexapod Locomotion on Uneven Terrain

> **PROPOSAL ONLY**. This is not a paper with results. All [RESULT PENDING] placeholders must be filled from reproducible experiment logs before submission. Never invent a number to make the story sound complete.

## Abstract

Camera rigidly mounted on a walking hexapod experiences gait-induced image motion, especially when body stabilization is constrained by stance configuration, leg workspace, and terrain geometry. We propose a compensation-aware gait adaptation approach that estimates, for each feasible gait candidate, the image motion remaining after the corrections attainable by a constrained body controller. A lightweight gait-conditioned disturbance model and an inner bounded body-compensation problem drive an online gait scheduler under locomotion-progress and foot-clearance constraints. We plan to evaluate the method in four categories of terrain with controlled effective speed, strong gait and body-control baselines, and lightweight on-robot camera measurements. [Insert only experimentally supported quantitative results and limitations after completion.]

## 1 Introduction

Problem → close prior work (Shaw/Smart Gait/PAMPC) → gap specific to compensation-conditioned gait selection → hypotheses → contributions conditional on evidence.

## 2 Related Work

2.1 Stable gaze and body pose in legged robots (Shaw 2019, Bazeille 2017).
2.2 Terrain-adaptive hexapod gait/body optimization (Smart Gait 2024, Yu 2025, Chen 2026).
2.3 Perception-aware motion and image blur constraints (PAMPC 2018, PANTHER 2022, Robic 2024).

## 3 Problem Statement

Camera-base transform, projected residual image motion, task-consistent camera motion, allowable body correction set, speed and terrain constraints.

## 4 Method

4.1 Camera disturbance estimation and image-plane projection.
4.2 Gait-conditioned prediction across candidates.
4.3 Compensation capacity from stance configuration and kinematic limits.
4.4 Nested body correction and gait selection.
4.5 Runtime considerations and fallback.

## 5 Experimental Setup

Dynamics simulation, camera model, baselines M0–M5, terrains, held-out split, effective-speed matching, metrics, repeated trials, confidence intervals.

## 6 Results — [RESULTS PENDING]

6.1 Prediction accuracy/ordering.
6.2 Multi-terrain main comparison.
6.3 Ablations.
6.4 Speed–stability–vision tradeoffs.
6.5 Lightweight physical validation.

## 7 Discussion

Comparable novelty boundary; controllable vs uncontrollable frequency range; real camera rolling-shutter limitations; generality beyond robot geometry; failures.

## 8 Conclusions — [RESULTS PENDING]

No completed claims until Results are supported by exported data/figures.

## Reference entry points

Use verified links in `docs/02_LITERATURE_EVIDENCE.md`. Zotero/BibTeX import only after matching DOI metadata.
