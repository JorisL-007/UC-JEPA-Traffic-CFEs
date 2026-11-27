# UC-JEPA-Traffic-CFEs

## Input Data

Download the METR-LA data from the following sources and store in 'data/'.
- [METR-LA.h5](https://www.kaggle.com/datasets/annnnguyen/metr-la-dataset)
- [graph_sensor_locations.csv](https://github.com/liyaguang/DCRNN/blob/master/data/sensor_graph/graph_sensor_locations.csv)

## Build the contextual datasets

Run from the repository root with the checked-in `cache/` directory and weather CSV.

```sh
python dataset.py --snapshot 2012  # data/metr_la_context.npz
python dataset.py --snapshot 2026  # data/metr_la_context_2026.npz
```

Both builds use 2012 METR-LA speeds and weather. The static OSM context comes from the September 2012 snapshot or the November 2026 cached queries. Use `--output` to write to a different `.npz` path.

## Upstream reproduction code

`T-GCN-PyTorch/` contains the forecasting and NSGA-II code from Wang et al. (2025), *Counterfactual explanations for deep learning-based traffic forecasting*. The import comes from commit `9618e0fbb1318043b8939515656edf62daa95dda` and includes the original MIT license. [Paper](https://doi.org/10.1016/j.commtr.2025.100176) · [Pinned source](https://github.com/RushanWang1/CFforTrafficForecast/tree/9618e0fbb1318043b8939515656edf62daa95dda/T-GCN-PyTorch) · [License](T-GCN-PyTorch/LICENSE)
