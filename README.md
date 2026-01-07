# MKCIKF: Multikernel Correntropy Invariant Kalman Filter for Localization of Legged Robot

## Overview  
This work proposes a multikernel correntropy invariant extended Kalman filter (MKCIEKF) that integrates multikernel correntropy with IEKF methodology to enhance robustness against heavy-tailed noise. Evaluations on biped robot simulations and quadruped robot experiments demonstrate that the proposed MKCIKF improves localization accuracy by approximately \textbf{38.1\%} and \textbf{18.3\%}, respectively, compared with IKF, while exhibiting enhanced robustness under complex and unstructured terrain conditions. 

---

## System Requirements
- **MATLAB 2017b**

---

## Example Codes and Descriptions

### Unicycle Robot
- **Objective**: Orientation and velocity estimation.
- **Implementation**: Run "main.m" to compare the performance of EKF, IKF, MCIKF, and MKCIKF on the unicycle robot.

### Biped Robot
- **Objective**: Critical orientation, pose, and velocity estimation.
- **Implementation**: Run "run_RIEKF_test.m" to test the algorithm on the Cassie-series biped robot. The algorithm fuses IMU information and kinematic data.

### Quadruped Robot
- **Objective**: Performance comparison of ESKF, IKF, MCIKF, and MKCIKF.
- **Implementation**:
  - **Simulation**: Use "iekf_mc_sim.m" for MCIKF, "iekf_sim.m" for IKF and "iekf_mck_sim.m" for MKCIKF.
  - **Real-world experiments**: Use "iekf_mck_real.m" for MKCIKF and "iekf_real.m" for IKF.
    
### Image  
Pre-generated figures in /image folder show.

---

## Citation
If you use this work in your research, please cite the relevant publication.

