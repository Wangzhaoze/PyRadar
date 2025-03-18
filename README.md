# scikit-radar


<div align="center">
    <h1>PyRadar: A Toolbox for FMCW Radar Signal Processing and Simulation</h1>
    <p>
        <a href="FIRST AUTHOR PERSONAL LINK" target="_blank">Zhaoze Wang</a><sup>*</sup>,
        <a href="SECOND AUTHOR PERSONAL LINK" target="_blank">Changxu Zhang</a><sup>*</sup>,
    </p>
    <p>Authors marked with * have contributed equally to this work.</p>
</div>


scikit-radar/
├── setup.py
├── src/
│   ├── cpp/
│   │   ├── rsp.cpp
│   │   ├── rsp.h
│   │   └── CMakeLists.txt
│   ├── cython/
│   │   ├── radar_signal.pyx
│   │   └── __init__.py
│   └── CMakeLists.txt
├── scripts/
│   ├── __init__.py
│   └── radar_signal.py  # Pure Python implementation
└── tests/
│    ├── test_cpp.py
│    ├── test_cython.py
│    └── test_python.py
├── docs/
│   ├── demo.ipynb
│   └── README.md
└── README.md