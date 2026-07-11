# Aquarium.jl

## Aquarium V2 coming soon!!

New open-source, refactored version of Aquarium based on our recent acceepted RSS paper will be coming soon! It will feature the following:

- Strongly-coupled multirigid-body-fluid interaction based on unified discrete variational mechanics
- Differentiability improvements that allow for interfacing with more optimization solvers (not just L-BFGS-B)
- Bug fixes in the 2D FVM, particularly boundary conditions.

This will also be presented at JuliaCon 2026 as part of the Computational Physics Minisymposium! 

## Citing

If you use Aquarium.jl and the Aquarium algorithm as part of your research, teaching, or other activities, we would be grateful if you could cite our works:

[1] J. H. Lee, M. Y. Michelis, R. Katzschmann and Z. Manchester, "Aquarium: A Fully Differentiable Fluid-Structure Interaction Solver for Robotics Applications," 2023 IEEE International Conference on Robotics and Automation (ICRA), London, United Kingdom, 2023, pp. 11272-11279, doi: 10.1109/ICRA48891.2023.10161494.

```
@INPROCEEDINGS{10161494,
  author={Lee, Jeong Hun and Michelis, Mike Y. and Katzschmann, Robert and Manchester, Zachary},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Aquarium: A Fully Differentiable Fluid-Structure Interaction Solver for Robotics Applications}, 
  year={2023},
  volume={},
  number={},
  pages={11272-11279},
  doi={10.1109/ICRA48891.2023.10161494}}
```
[2] J. H. Lee, J. Hu, S. Kwok, C. Majidi, and Z. Manchester, “Realizing Robotic Swimming with Unified Fluid-Robot Multiphysics,” in 2026 Robotics: Science and Systems, 2026. [Online]. Available: https://arxiv.org/abs/2506.05012
```
@misc{lee2026realizingroboticswimmingunified,
      title={Realizing Robotic Swimming with Unified Fluid-Robot Multiphysics}, 
      author={Jeong Hun Lee and Junzhe Hu and Sofia Kwok and Carmel Majidi and Zachary Manchester},
      year={2026},
      eprint={2506.05012},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2506.05012}, 
}
