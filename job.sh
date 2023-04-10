#!/bin/bash
# 系统环境变量 wiki: http://wiki.baidu.com/pages/viewpage.action?pageId=1053742013#2.4%20CaaS
current_rank_index=${POD_INDEX}
rank_0_ip=${POD_0_IP}
free_port=${TRAINER_PORTS}
dist_url="tcp://${rank_0_ip}:${free_port}"
world_size=2
echo "current_rank_index: ${current_rank_index}"
echo "dist_url: ${dist_url}"
echo "world_size: ${dist_url}"

lsof -i:${free_port}

function switch_py36() {
    # switch python3
    export PY36_HOME=/opt/conda/envs/py36
    export PATH=${PY36_HOME}/bin:$PATH
    export LD_LIBRARY_PATH=${PY36_HOME}/lib/:${LD_LIBRARY_PATH}

}

function py27_local() {
    # default py27
    echo "this is py27 local train job..."
    python train.py -a resnet18 ./afs/imagenet2012
}

function py36_local() {
    echo " this is py36 local train job..."
    #switch_py36
    #pip install torch===1.7.1 torchvision===1.8.2 -i http://pip.baidu-int.com/simple/
    #https://pypi.tuna.tsinghua.edu.cn/simple
    #http://pip.baidu-int.com/simple/
    #https://pypi.tuna.tsinghua.edu.cn/simple
    export PYTHON_HOME=/opt/conda/
    export LD_LIBRARY_PATH=${PYTHON_HOME}/lib:${LD_LIBRARY_PATH}
    export PATH=${PYTHON_HOME}/bin:${PATH}
    export LANG="en_US.UTF-8"
    export PYTHONIOENCODING=utf-8
    python3 test.py &> test.txt
    #tar -zxvf afs/tmp_small.tar.gz
    tar -xf afs/ModelNet40_rendered_rgb.tar
    pip install yacs
    pip install setuptools==40.6.2
    pip install Pillow
    pip install einops
    pip install dataclasses
    #tar -zcvf timm-0.4.5.tar.gz timm-0.4.5
    #pip install timm-0.4.5.tar.gz
    #.tar.gz
    cp afs/apex.tar.gz .
    tar -zxvf apex.tar.gz
    cd apex
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    cd ..
    #sh scripts/run_sop.sh &> sop.txt
    # ----- attention tiny w/o pretrain ----- #
    python -m torch.distributed.launch --master_addr 127.0.0.2 --master_port 23918 --nproc_per_node=2 --use_env modelnet40_12view_main.py --model deit_tiny_patch16_224 --batch-size 64 --data-path modelnet40_ply_hdf5_2048 --output_dir outputs --num_workers 4  2>&1 | tee .alog.txt
    # ----- attention small w/o pretrain ----- #
#    python -m torch.distributed.launch --master_addr 127.0.0.2 --master_port 23918 --nproc_per_node=2 --use_env modelnet40_12view_main.py --model deit_small_patch16_224 --batch-size 64 --data-path modelnet40_ply_hdf5_2048 --output_dir outputs --num_workers 4 2>&1 | tee alog.txt
    #python3
}

function py36_local_multicard() {
    # pytorch spawn 只支持 py3.4 或者个更高，因此该实例单机多卡，仅跑 py3
    echo "this is py36 local multicard train job..."
    switch_py36
    # pytorch 1.3.0 需要降低 Pillow 的版本， pytorch 1.4.0 无此问题
    pip install Pillow==6.2.2
    pip install tensorboard_logger
    python train.py -a resnet50 --dist-url ${dist_url} --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ./afs/imagenet2012

}

function py36_distribute() {
    # pytorch spawn 只支持 py3.4 或者个更高，因此该实例单机多卡，仅跑 py3
    echo "this is py36 distribute train job..."
    switch_py36
    # pytorch 1.3.0 需要降低 Pillow 的版本， pytorch 1.4.0 无此问题
    pip install Pillow==6.2.2
    python train.py -a resnet50 \
                    --dist-url ${dist_url} \
                    --dist-backend 'nccl' \
                    --multiprocessing-distributed \
                    --world-size ${world_size} \
                    --rank ${current_rank_index} ./afs/imagenet2012
}

function main() {
    if [[ "${IS_STANDALONE}" = "1" ]]; then
        echo "this is local mode job, will run py27 and py36 local train job..."
        if [[ "${TRAINER_GPU_CARD_COUNT}" = "1" ]]; then
            echo "this one gpu card train job..."
            py27_local
            sleep 2
            py36_local
        else
            echo "this multi gpu card train job..."
            py36_local_multicard
        fi
    else
        echo " this is distribute train job..."
        py36_distribute
    fi
    echo "finished!"
}


#main
#cp afs/train_kdd.tar.gz .
#tar -zxvf afs/train_kdd.tar.gz #&> unzipt.txt
#ls train_kdd > i.txt
#cp afs/valid_kdd.tar.gz .
#python writekddtrain.py &> wt.txt
#tar -zxvf afs/valid_kdd.tar.gz #&> unzipv.txt
#ls train_kdd > t.txt
#mv train_kdd data
#mv valid_kdd data

py36_local
