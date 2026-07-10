# ADR 0008: ffmpeg concat-demuxer assembly with EBU R128 loudness normalization

Date: 2026-07-10
Status: Accepted

## Context

Stitched per-line WAVs need joining with natural pauses and consistent loudness.
Prior-art tools that skip normalization produce episodes with jarring level jumps
between voices; pure-Python audio stacks add heavy dependencies.

## Decision

Assembly shells out to ffmpeg (documented system dependency, verified by `doctor`):
concat demuxer over a generated file list, randomized 200–1000 ms inter-turn silence,
resample on sample-rate mismatch, single-pass EBU R128 `loudnorm`, WAV intermediate,
MP3 export.

## Consequences

- No heavy Python audio dependencies; ffmpeg must be on PATH (checked early with a
  clear error).
- Randomized gap lengths avoid the metronomic cadence of fixed-gap stitching.
