To build the gpu enable docker image, you have to run docker build in
a gpu enabled machine, such as one created in EC2 from the AMI
Deep Learning AMI GPU PyTorch 1.11.0 (Amazon Linux 2) 20220526

This AMI correspond to an ec2 linux with GPU drivers and nvidia-docker

However docker build runs temporal containers that are not GPU enabled
and the provided Dockerfile-gpu requires gpu also during the building.

The solution is to edit /etc/docker/daemon.json to add a default runtime:

```json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```

After doing this, the provided `Dockerfile-gpu*` can be used with
`docker build` to create the required images.

The are two different dockefiles:

* `Dockerfile-gpu` uses as base images the ones provided by Amazon as DLC
  (Deep Learning Containers). These are images that include cuda, conda,
  and lots of scientific python tools. There are huge images (12GB)
* `Dockerfile-gpu-smaller` is based on `pytorch/torchserve` and it produces
  a much smaller image (5.2GB), and for yolo is enough

