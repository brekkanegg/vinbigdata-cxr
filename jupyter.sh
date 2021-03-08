HASH=$(date | sha256sum | cut -c1-4)
docker run \
-p 8001:8001 \
-p 8002:8002 \
-p 6001:6001 \
-p 6002:6002 \
-it \
--rm \
--runtime=nvidia \
-u $(id -u):$(id -g) \
--name jupyter-${HASH} \
--shm-size=128g \
-v /home/minki/kaggle:/home/minki/kaggle \
-v /nfs3/minki:/nfs3/minki \
-v /data2:/data2 \
-v /data:/data \
-w /home/minki/kaggle/vinbigdata-cxr \
minki/cxr:v1.1
