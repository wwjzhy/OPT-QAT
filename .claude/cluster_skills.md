1. 集训配置，slurm脚本提交任务


2. 节点调试

申请节点调试：

'''
#!/bin/bash
set -x

container_name="verl+cu126+0503"
container_workdir="/work/projects/polyullm/wenjun/posttrain/MOP-SD"
container_image="/lustre/projects/polyullm/container/verl+cu126+0503.sqsh"
container_mounts="/lustre/projects/polyullm:/lustre/projects/polyullm,/work/projects/polyullm:/work/projects/polyullm"

srun -n 1 \
  -c 64 \
  --container-name=${container_name} \
  --container-image=${container_image} \
  --container-mounts=${container_mounts} \
  --container-workdir=${container_workdir} \
  --container-remap-root \
  --container-writable \
  --pty bash

'''

进入已申请节点，如03节点:

'''
ssh wenjun@kb3-a1-nv-dgx03
'''



3. 环境安装

miniconda路径：/lustre/projects/polyullm/wenjun/envs/miniconda3


4. 下载conda环境

pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package名

5. 下载数据集

用hf-mirror下载