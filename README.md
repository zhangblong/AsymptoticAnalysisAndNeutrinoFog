[![arXiv](https://img.shields.io/badge/arXiv-2109.03116-B31B1B.svg)](https://arxiv.org/abs/2109.03116)
[![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)

# Asymptotic Analysis and Neutrino Floor&Fog
Codes for reproducing results from my paper arXiv:[2304.xxxxx]
---
<!-- [<img align="right" src="plots/plots_png/NuFloorExplanation.png" height="350">](https://github.com/cajohare/NeutrinoFog/raw/master/plots/plots_png/NuFloorExplanation.png) -->

# Attention
Our codes for obtaining the event rates and ploting  reference [`Ciaran's codes`](https://github.com/cajohare/NeutrinoFog/).

# Requirements
* [`CMasher`](https://cmasher.readthedocs.io/)
* [`Numba`](https://numba.pydata.org/)

# Running the code
Run the notebooks, and get your wants.

# Contents
* [`src/`] - Contains functions used in producing  results
* [`plots/`] - Contains all the plots in pdf and png formats
* [`notebooks/`] - Jupyter notebooks for obtaining and plotting results
* [`data/`] - data files, including neutrino fluxes, experimental exclusion limits, and the MC data

# Python modules
* [`WIMPFuncs.py`] - Functions needed to calculate WIMP rates
* [`NeutrinoFuncs.py`] - Functions needed to calculate neutrino rates
* [`LabFuncs.py`] - Various utilities
* [`PlotFuncs.py`] - Plotting functions
* [`NeutrinoFogFuncs.py`] - Functions for obtaining the neutrino fog data
* [`NeutrinoFogPlotFuncs.py`] - Functions for showing the results

# Notebooks
* [`DifferentTargets.ipynb`] - Figures 7
* [`DiscoveryLimitCurve.ipynb`] - Figure 2
* [`DLwithLSR.ipynb`] - Figure 3
* [`MCLSR.ipynb`] - Producing MC pseudo-experiments for the test statistic considering the velocity of the local standard of rest(LSR)
* [`MCNeutrino.ipynb`] - Producing MC pseudo-experiments for the test statistic only considering the neutrino fluxes
* [`MCPlot.ipynb`] - Figure 5 & 6
* [`RegeneratingNeutrinoFog.ipynb`] - Figure 1
* [`RegeneratingNeutrinoFloor.ipynb`] - Figure 4

---

If you need any further assistance or have any questions, contact me at zhangblong1036@gmail.com. And if you do use anything here please cite the paper, [Bing-Long Zhang](https://arxiv.org/abs/2304.xxxxx)
```
@article{a
}
```
