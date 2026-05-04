# Final Report — Group 7

| File | What it is |
|---|---|
| `Final_Report.pdf` | Submitted final report (19 pages, B/W) |
| `Final_Report.docx` | Editable source of the report |
| `Final_Report_Slides.pptx` | Slide deck (15 slides) |
| `Final_Report_Slides.pdf` | PDF export of the deck |
| `ppo_vs_ppolag_demo.mp4` | ~33 s merge-v0 side-by-side: PPO vs PPOLag |
| `all_envs_demo.mp4` | 20 s sequential tour across the four scenarios |
| `all_envs_demo_meta.json` | Crash counts + action distributions for the 4-env demo |
| `figures/` | Plots embedded in the report and slides |
| `policies/` | Quick-trained policy weights for highway, roundabout, intersection |
| `make_*.py`, `train_quick.py`, `build_report.js` | Scripts used to regenerate the demo videos, figures, and Final_Report.docx |

## Headline numbers (merge-v0, 100 evaluation episodes)

| Method | Collision rate | Avg reward | Avg cost |
| --- | --- | --- | --- |
| PPO | 40.0 % | 9.69 | 0.40 |
| PPOLag | **9.0 %** | 8.08 | **0.09** |

## Demo videos at a glance

| | Crashes | Top actions |
|---|---|---|
| PPO on merge-v0 | 6 / 24 (25 %) | brake / accelerate mix |
| PPOLag on merge-v0 | 3 / 24 (12 %) | 98 % SLOWER (brake-only) |
| Highway (Quick PPO 3k steps) | 0 / 5 (0 %) | 100 % SLOWER |
| Roundabout (Quick PPO 3k steps) | 2 / 6 (33 %) | FAST 53 %, IDLE 22 %, LANE 13 % |
| Intersection (Quick PPO 3k steps) | 9 / 18 (50 %) | FAST 67 %, LANE 15 % |

## Reproducing the deliverables

```bash
# Demo videos
python make_demo.py             # merge PPO vs PPOLag
python make_sequential_demo.py  # 4-env tour

# Figures
python make_figures.py

# Final report (docx → soffice → pdf)
node build_report.js
soffice --headless --convert-to pdf Final_Report.docx
```
