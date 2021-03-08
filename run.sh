HASH=$(date | sha256sum | cut -c1-4)
docker run -it \
--rm \
--runtime=nvidia \
--name vidr-cxr-${HASH} \
--shm-size=128g \
-u $(id -u):$(id -g) \
-v /home/minki/kaggle:/home/minki/kaggle \
-v /nfs3/minki:/nfs3/minki \
-v /nfs3/chestpa:/nfs3/chestpa \
-v /data2:/data2 \
-v /data:/data \
-w /home/minki/kaggle/vinbigdata-cxr \
minki/cxr:v1.1

