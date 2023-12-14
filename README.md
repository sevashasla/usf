# uSF: Learning Neural Semantic Field with Uncertainty
The official implementation of the [paper](https://arxiv.org/abs/2312.08012)

combination of Semantic NeRF & instant-ngp (in implementation of torch-ngp) & uncertainty.\

### How to install
The install steps are pretty similar with the steps from [torch-ngp](https://github.com/ashawkey/torch-ngp)


### How to use manytest
There are lots of parameters to torch-ngp, semantic-ngp, semantic_nerf.\
So, I choose not bad combinations (for me) and set as default in `manytest/manytest_yaml.py`.\
But, many of parameters one can change, please look at `manytest/configs/`

### Thanks
- [torch-ngp](https://github.com/ashawkey/torch-ngp)
- [instant-ngp](https://github.com/NVlabs/instant-ngp)
- [semantic_nerf](https://github.com/Harry-Zhi/semantic_nerf)
