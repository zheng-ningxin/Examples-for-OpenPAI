# Examples-for-OpenPAI
This project provides the examples to run on the [OpenPAI](https://github.com/microsoft/pai). The sample program supports CPU, GPU, multiple GPU, [Horovod](https://github.com/horovod/horovod) modes to train a classification model for cifar10. Through this project, users can quickly and easily run an OpenPAI instance, which can be used to learn and understand the OpenPAI, or test the performance of the OpenPAI.

| Network | Hardware | Time |GPU & CPU Utilization | Accuracy (Avg of 3 runs) | Yaml Example|
| :----:| :----: | :----: | :----: | :----: | :----: |
| Resnet18 | V100 * 1 | 59m(200 epoch) | [Details](jpgs/Resnet18_1gpu.jpg) | 95.2% | [Resnet18_1gpu.yaml](yaml/Resnet18_1gpu.yaml) |
| Resnet18 | V100 * 4 | 30m(200 epoch) | [Details](jpgs/Resnet18_4gpus.jpg) | 94.9% | [Resnet18_4gpu.yaml](yaml/Resnet18_4gpu.yaml) |
| Resnet18 | CPU  * 12| 21h 33m(200 epoch) | [Details](jpgs/Resnet18_12cpu.jpg) | 95.15% | [Resnet18_12cpu.yaml](yaml/Resnet18_12cpu.yaml)
| Resnet18 | V100 * 4(Horovod) | 23m(200 epoch) | [Details](jpgs/Resnet18_horovod.jpg) | 93.6% | [Restnet18_horovod.yaml](yaml/Resnet18_horovod.yaml)

