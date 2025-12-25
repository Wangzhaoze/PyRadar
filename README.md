<div align="center">
    <h1>pyradar: A Toolbox for FMCW Radar Signal Processing and Simulation</h1>
    <p> 
        Authors:
        <a href="https://wangzhaoze.github.io/" target="_blank">Zhaoze Wang</a><sup></sup>,
        <a href="https://github.com/changxu-zhang" target="_blank">Changxu Zhang</a><sup></sup>
    </p>
    
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

<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Files</th>
      <th>Assigned To</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><del>Window FFT (Rec., Hann, Hamming)</td>
      <td><a href="./pyradar/rsp/">pyradar/rsp</a></td>
      <td></td>
    </tr>
    <tr>
      <td>Micro-Doppler STFT</td>
      <td><a href="./pyradar/rsp/">pyradar/rsp</a></td>
      <td></td>
    </tr>
    <tr>
      <td><del>2D CFAR </td>
      <td><a href="./pyradar/rsp/">pyradar/rsp</a></td>
      <td></td>
    </tr>
    <tr>
      <td>Clustering</td>
      <td><a href="./pyradar/rsp/">pyradar/rsp</a></td>
      <td></td>
    </tr>
    <tr>
      <td>Point Cloud</td>
      <td><a href="./pyradar/rsp/">pyradar/rsp</a></td>
      <td></td>
    </tr>
    <tr>
      <td><del>MUSIC</td>
      <td><a href="./pyradar/rsp/">pyradar/rsp</a></td>
      <td></td>
    </tr>
    <tr>
      <td><del>Beamforming</td>
      <td><a href="./pyradar/rsp/">pyradar/rsp</a></td>
      <td></td>
    </tr>
    <tr>
      <td>Path-Tracer</td>
      <td><a href="./pyradar/rsp/">pyradar/rsp</a></td>
      <td></td>
    </tr>
    <tr>
      <td>Radar Simulation Model</td>
      <td><a href="./pyradar/rsp/">pyradar/rsp</a></td>
      <td></td>
    </tr>
    <tr>
      <td>CPU/GPU acceleration</td>
      <td></td>
      <td></td>
    </tr>
    <!-- <tr>
      <td>-</td>
      <td><a href="-">-</a></td>
      <td><a href="-">-</a></td>
    </tr> -->
  </tbody>
</table>

## License

This work is released under the MIT license.

## Citation
In this code, we use the subset of **[ColoRadar](https://arpg.github.io/coloradar/)** and **[CRUW](https://www.cruwdataset.org/)** dataset to acknowledge their contributions.


- Kramer, Andrew, Kyle Harlow, Christopher Williams, and Christoffer Heckman. “ColoRadar: The direct 3D millimeter wave radar dataset.” The International Journal of Robotics Research 41, no. 4 (2022): 351-360.
- Wang, Yizhou, Zhongyu Jiang, Yudong Li, Jenq-Neng Hwang, Guanbin Xing, and Hui Liu. “RODNet: A Real-Time Radar Object Detection Network Cross-Supervised by Camera-Radar Fused Object 3D Localization.” IEEE Journal of Selected Topics in Signal Processing 15, no. 4 (2021): 954-967.


## Build via pip
```bash
pip install .
python -m build
pip install dist/pyradar-0.1.0-py3-none-any.whl
python -c "import pyradar; print(pyradar.__file__)"
```
