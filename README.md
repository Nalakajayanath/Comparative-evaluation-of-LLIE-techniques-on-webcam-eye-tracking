# Comparative Evaluation of Low-Light Image Enhancement Techniques for Webcam-Based Gaze Tracking

## Project Overview
This research evaluates how different Low-Light Image Enhancement (LLIE) techniques affect the accuracy and real-time feasibility of webcam-based gaze estimation.

The study integrates LLIE methods as preprocessing modules before a pre-trained L2CS gaze estimation model.

## Research Questions
RQ1: Which LLIE techniques most effectively improve gaze accuracy?
RQ2: Which LLIE techniques offer the best trade-off between accuracy and computational efficiency?

## Architecture

Low-Light Image
        ↓
LLIE Module
        ↓
L2CS Gaze Estimator (Black Box)
        ↓
Angular Error Evaluation

## Implemented LLIE Techniques

- Histogram Equalization
- CLAHE
- SSR
- MSR
- Zero-DCE
- MIRNet

## Evaluation Metrics

- Angular Error (degrees)
- FPS
- Enhancement latency

## Setup

```bash
pip install -r requirements.txt