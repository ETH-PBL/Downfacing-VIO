Efficient and Accurate Downfacing Visual Inertial Odometry
====
This project provides the code accompanying the paper entitled Efficient and Accurate Downfacing Visual Inertial Odometry (preprint available soon).

Requirements
----

This project requires the [GAP SDK for GAP9](https://github.com/GreenWaves-Technologies). The project can either be run in GVSoC or on a physical GAP9 board.

Running the Code
----
The downfacing VIO system can be executed on GVSoC using the following commands:

**ORB Feature Tracker**

``` bash
cd orb_gap9_project
cmake -B build
cmake --build build --target run
```

**SuperPoint Feature Tracker**

``` bash
cd superpoint_gap9_project
cmake -B build
cmake --build build --target run
```

**Parallelized PX4FLOW Feature Tracker**

``` bash
cd px4flow_gap9_project
cmake -B build
cmake --build build --target run
```

**Note:** The code is executed on GVSoC by default. If you want to execute it on a physical GAP9 system, adjust the target in the menuconfig.

``` bash
cmake --build build --target menuconfig
```

ORB and PX4FLOW will be executed on a single cluster core by default. For multicore execution change the following line in the `main.c` file: 
``` c
uint8_t SINGLE_CORE = 1;
```

Citing this Work
----
If you found our work helpful in your research, we would appreciate if you cite it as follows:

**Efficient and Accurate Downfacing Visual Inertial Odometry**
* *To be added later*

**Parallelizing Optical Flow Estimation on an Ultra-Low Power RISC-V Cluster for Nano-UAV Navigation**
[arXiv](https://arxiv.org/abs/2305.13055)
```
@inproceedings{kuhne2022parallelizing,
  title={Parallelizing optical flow estimation on an ultra-low power risc-v cluster for nano-uav navigation},
  author={K{\"u}hne, Jonas and Magno, Michele and Benini, Luca},
  booktitle={2022 IEEE International Symposium on Circuits and Systems (ISCAS)},
  pages={301--305},
  year={2022},
  organization={IEEE}
}
```
