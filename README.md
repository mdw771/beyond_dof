# Tiling-based Fresnel multislice propagation
This repo contains the codes for tiling-based Fresnel multislice propagation, which was demonstrated and benchmarked in the following publication(s):

- Ali, S., Du, M., Adams, M., Smith, B., & Jacobsen, C. A comparison of distributed memory algorithms for x-ray wave propagation in inhomogeneous media. Opt Exp, (2020). (Submitted)

To reproduce the results shown in the paper, please switch to branch `mpi_conv_bp`, and run the following scrpts in directory `cnn_propagator`:

- `benchmark_pfft_zp.py` is used for convergence test on the zone plate object (or any axially repeating objects).
- `benchmark_pfft.py` is used for convergence test on the charcoal object. Please contact the authors for raw data. 
- `s_scaling_pfft_zp.py` is used for zone plate scaling test.
- `s_scaling_pfft_zp_gpu.py` is used for GPU benchmarking, which requires PyTorch and Adorym ([https://github.com/mdw771/adorym_dev](https://github.com/mdw771/adorym_dev)).

# Adorym (Automatic-Differentiation-based Object Retrieval using dYnamical Modeling) (THIS REPO IS OUTDATED)
Optimization-based 3D reconstruction algorithm for objects beyond the depth-of-focus using Tensorflow as an iterative engine

This code repo was used in the following publication(s):

- Du, M., Nashed, Y. S. G., Kandel, S., Gürsoy, D. & Jacobsen, C. Three dimensions, two microscopes, one code: Automatic differentiation for x-ray nanotomography beyond the depth of focus limit. Sci Adv 6, eaay3700 (2020).

**This is NOT the updated repo for Adorym. To retrieve the latest version of the software, visit [https://github.com/mdw771/adorym_dev](https://github.com/mdw771/adorym_dev) (developing repo) or [https://github.com/mdw771/adorym](https://github.com/mdw771/adorym) (stable version).**
