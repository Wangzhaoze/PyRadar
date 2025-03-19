<div align="center">
    <h1>scikit-radar: A Toolbox for FMCW Radar Signal Processing and Simulation</h1>
    <p>
        <a href="https://wangzhaoze.github.io/" target="_blank">Zhaoze Wang</a><sup>*</sup>,
        <a href="SECOND AUTHOR PERSONAL LINK" target="_blank">Changxu Zhang</a><sup>*</sup>,
    </p>
    <p>Authors marked with * have contributed equally to this work.</p>
</div>


## Setup Python Environment
You can setup environment by:
- Using pip:
```bash
source shell/setup_env_no_conda.sh
```
- Using conda
```bash
source shell/setup_env_conda.sh
```


## Data
Raw ADC data from ColoRadr is given as example. More follows

## TODO

- [ ] Window FFT (Rec., Hann, Hamming)
- [ ] 2D/3D CFAR variant
- [ ] MUSIC
- [ ] CPU/GPU acceleration
- [ ] CI Test and pre-commit
- [ ] Radar Antenna visualization

## License

This work is released under the MIT license.

## Citation
In this code, we use the subset of **[ColoRadar](https://arpg.github.io/coloradar/)** and **[CRUW](https://www.cruwdataset.org/)** dataset to acknowledge their contributions.


- Kramer, Andrew, Kyle Harlow, Christopher Williams, and Christoffer Heckman. “ColoRadar: The direct 3D millimeter wave radar dataset.” The International Journal of Robotics Research 41, no. 4 (2022): 351-360.
- Wang, Yizhou, Zhongyu Jiang, Yudong Li, Jenq-Neng Hwang, Guanbin Xing, and Hui Liu. “RODNet: A Real-Time Radar Object Detection Network Cross-Supervised by Camera-Radar Fused Object 3D Localization.” IEEE Journal of Selected Topics in Signal Processing 15, no. 4 (2021): 954-967.