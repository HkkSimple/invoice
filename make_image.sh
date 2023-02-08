#! /bin/bash

image_name=kksimple/invoice:cpu_version_0.02

docker image build . -t $image_name

docker run --rm -p 30500:30500 -v /mnt/data/rz/project/share:/mnt/share -it $image_name