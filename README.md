# Cloud cover

![Python 3.9](https://img.shields.io/badge/Python-3.9-blue) [![GPU Docker Image](https://img.shields.io/badge/Docker%20image-gpu--latest-green)](https://hub.docker.com/r/drivendata/cloud-cover-competition/tags?page=1&name=gpu-latest) [![CPU Docker Image](https://img.shields.io/badge/Docker%20image-cpu--latest-green)](https://hub.docker.com/r/drivendata/cloud-cover-competition/tags?page=1&name=cpu-latest) 

Welcome to the runtime repository for the [Cloud Cover Challenge](https://www.drivendata.org/competitions/83/cloud-cover/). This repository contains the definition of the environment where your code submissions will run. It specifies both the operating system and the software packages that will be available to your solution.

<div style="background-color: lightgoldenrodyellow">

**Note:** This repository is designed to be compatible with Microsoft's [Planetary Computer](https://github.com/microsoft/planetary-computer-containers) containers.

The [Planetary Computer Hub](https://planetarycomputer.microsoft.com/docs/overview/environment) provides a convenient way to compute on data from the Planetary Computer. In this competition, you can train your model in the Planetary Computer Hub and test it using this repo. To request beta access to the Planetary Computer Hub, fill out [this form](https://planetarycomputer.microsoft.com/account/request) and include "DrivenData" in your area of study.

</div>

This repository has three primary uses for competitors:

- **Example for developing your solutions**: You can find two examples to help you develop your solution. The first is a [baseline solution](https://github.com/drivendataorg/cloud-cover-runtime/tree/main/submission_src) which does not do very much but will run in the runtime environment and outputs a proper submission. You can use this as a guide to bring in your model and generate a submission. The second is an implementation of the [PyTorch benchmark](https://github.com/drivendataorg/cloud-cover-runtime/tree/main/benchmark_src) based on the [TODO: benchmark blog post](https://www.drivendata.co/blog/).

- **Test your code submission**: Test your `submission.zip` file with a locally running version of the container to discover errors before submitting it to the competition site. You can also find an [evaluation script](https://github.com/drivendataorg/cloud-cover-runtime/blob/main/runtime/scripts/metric.py) for implementing the competition metric.

- **Request new packages in the official runtime**: It lets you test adding additional packages to the official runtime [CPU](https://github.com/drivendataorg/cloud-cover-runtime/blob/main/runtime/environment-cpu.yml) and [GPU](https://github.com/drivendataorg/cloud-cover-runtime/blob/main/runtime/environment-gpu.yml) environments. The official runtime uses **Python 3.9.6** environments managed by [Anaconda](https://docs.conda.io/en/latest/). You can then submit a PR to request compatible packages be included in the official container image.

 ----

### [Getting started](#0-getting-started)
 - [Prerequisites](#prerequisites)
 - [Quickstart](#quickstart)
### [Testing your submission locally](#1-testing-your-submission-locally)
 - [Implement your solution](#implement-your-solution)
 - [Example benchmark submission](#example-benchmark-submission)
 - [Running your submission](#running-your-submission)
 - [Reviewing the logs](#reviewing-the-logs)
### [Updating the runtime packages](#2-updating-the-runtime-packages)

----

## (0) Getting started

### Prerequisites

 - A clone or fork of this repository
 - [Docker](https://docs.docker.com/get-docker/)
 - At least ~12 GB of free space for both the training images and the Docker container images
 - [GNU make](https://www.gnu.org/software/make/) (optional, but useful for running the commands in the Makefile)

Additional requirements to run with GPU:

 - [NVIDIA drivers](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) with **CUDA 11**
 - [NVIDIA Docker container runtime](https://nvidia.github.io/nvidia-container-runtime/)

### Quickstart

This section explains how to test out the full execution pipeline, including how to get the Docker images, zip up an example submission, and run the submission on a local version of the container.  The `make` commands will try to select the CPU or GPU image automatically by setting the `CPU_OR_GPU` variable based on whether `make` detects `nvidia-smi`.

**Note:** On machines with `nvidia-smi` but a CUDA version other than 11, `make` will automatically select the GPU image, which will fail. In this case, you will have to set `CPU_OR_GPU=cpu` manually in the commands, e.g., `CPU_OR_GPU=cpu make pull`, `CPU_OR_GPU=cpu make test-submission`. If you want to try using the GPU image on your machine but you don't have a GPU device that can be recognized, you can use `SKIP_GPU=true` which will invoke `docker` without the `--gpus all` argument.

### Fake test data

First you will need some data for your submission to run on. Since we do not give direct access to the test data, you'll need to fake the data in one of two ways:

- Use the train data: On the [Data download page](https://www.drivendata.org/competitions/83/cloud-cover/data/) you'll find instructions on how to download the training data (`data_download_instructions.txt`). Move the TIF images from `train_features` into `runtime/data/test_features`. Also download `train_metadata.csv` to `runtime/data/test_metadata.csv`. Now you should be able to test your submission locally by pretending your training data is the test data expected by the execution environment.

```sh
$ tree runtime/data
├── test_features
│	├── adwp
│	│   ├── B02.tif
│	│   ├── B03.tif
│	│   ├── B04.tif
│	│   └── B08.tif
│	├── adwu
│	│   ├── B02.tif
│	│   ├── B03.tif
│	│   ├── B04.tif
│       └── B08.tif
│   ...
└── test_metadata.csv
```

- Generate fake data: We have included a script that will generate random images and metadata that are the same format as the actual test data. Don't expect to do very well on these!

```sh
$ python runtime/scripts/generate_fake_inputs.py runtime/data

$ tree runtime/data
├── test_features
│   ├── 0000
│   │   ├── B02.tif
│   │   ├── B03.tif
│   │   ├── B04.tif
│   │   └── B08.tif
│   ├── 0001
│   │   ├── B02.tif
│   │   ├── B03.tif
│   ...
└── test_metadata.csv
```

Whichever version of the fake data you choose, now you're ready to run the benchmark code:

```bash
make pull
make pack-benchmark
make test-submission
```

You should see output like this in the end (and find the same logs in the folder `submission/log.txt`):

```
$ make pack-benchmark
cd benchmark_src; zip -r ../submission/submission.zip ./*
  adding: assets/ (stored 0%)
  adding: assets/torch/ (stored 0%)
  adding: assets/torch/hub/ (stored 0%)
  adding: assets/torch/hub/checkpoints/ (stored 0%)
  adding: assets/torch/hub/checkpoints/resnet34-333f7ec4.pth (deflated 7%)
  adding: assets/cloud_model.pt (deflated 8%)
  adding: cloud_dataset.py (deflated 63%)
  adding: cloud_model.py (deflated 74%)
  adding: losses.py (deflated 57%)
  adding: main.py (deflated 65%)

$ SKIP_GPU=true make test-submission
chmod -R 0777 submission/
docker run \
        -it \
         \
        --rm \
        --name cloud_cover_submission \
        --mount type=bind,source="/home/robert/projects/cloud-cover-runtime"/runtime/data,target=/codeexecution/data,readonly \
        --mount type=bind,source="/home/robert/projects/cloud-cover-runtime"/runtime/tests,target=/codeexecution/tests,readonly \
        --mount type=bind,source="/home/robert/projects/cloud-cover-runtime"/runtime/entrypoint.sh,target=/codeexecution/entrypoint.sh \
        --mount type=bind,source="/home/robert/projects/cloud-cover-runtime"/submission,target=/codeexecution/submission \
        --shm-size 8g \
        5c6354f18833
+ exit_code=0
+ tee /codeexecution/submission/log.txt
+ cd /codeexecution
+ echo 'List installed packages'
List installed packages
+ echo '######################################'
######################################
+ conda list -n condaenv
# packages in environment at /srv/conda/envs/condaenv:
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                 conda_forge    conda-forge
_openmp_mutex             4.5                      1_llvm    conda-forge
...
zlib                      1.2.11            h36c2ea0_1013    conda-forge
zstd                      1.5.0                ha95c52a_0    conda-forge
+ echo '######################################'
######################################
+ echo 'Unpacking submission...'
Unpacking submission...
+ unzip ./submission/submission.zip -d ./
Archive:  ./submission/submission.zip
   creating: ./assets/
   creating: ./assets/torch/
   creating: ./assets/torch/hub/
   creating: ./assets/torch/hub/checkpoints/
  inflating: ./assets/torch/hub/checkpoints/resnet34-333f7ec4.pth
  inflating: ./assets/cloud_model.pt
  inflating: ./cloud_dataset.py
  inflating: ./cloud_model.py
  inflating: ./losses.py
  inflating: ./main.py
+ ls -alh
total 56K
drwxr-xr-x 1 appuser appuser 4.0K Nov 18 19:12 .
drwxr-xr-x 1 root    root    4.0K Nov 18 19:12 ..
drwxrwxr-x 3 appuser appuser 4.0K Nov 18 18:39 assets
-rw-rw-r-- 1 appuser appuser 2.4K Nov 18 17:33 cloud_dataset.py
-rw-rw-r-- 1 appuser appuser 6.9K Nov 18 18:03 cloud_model.py
drwxrwxr-x 4 appuser appuser 4.0K Nov 18 17:55 data
-rw-rw-r-- 1 appuser appuser 1.1K Oct  8 19:58 entrypoint.sh
-rw-rw-r-- 1 appuser appuser  699 Nov 18 17:34 losses.py
-rw-rw-r-- 1 appuser appuser 4.7K Nov 18 18:38 main.py
drwxr-xr-x 2 appuser appuser 4.0K Nov 17 21:38 scripts
drwxrwxrwx 2 appuser appuser 4.0K Nov 18 19:12 submission
drwxrwxr-x 2 appuser appuser 4.0K Nov 18 18:25 tests
+ '[' -f main.py ']'
+ echo 'Running submission with Python'
Running submission with Python
+ conda run --no-capture-output -n condaenv python main.py
/srv/conda/envs/condaenv/lib/python3.9/site-packages/pretrainedmodels/models/dpn.py:255: SyntaxWarning: "is" with a literal. Did you mean "=="?
  if block_type is 'proj':
/srv/conda/envs/condaenv/lib/python3.9/site-packages/pretrainedmodels/models/dpn.py:258: SyntaxWarning: "is" with a literal. Did you mean "=="?
  elif block_type is 'down':
/srv/conda/envs/condaenv/lib/python3.9/site-packages/pretrainedmodels/models/dpn.py:262: SyntaxWarning: "is" with a literal. Did you mean "=="?
  assert block_type is 'normal'
2021-11-18 19:12:40.810 | INFO     | __main__:main:111 - Loading model
2021-11-18 19:12:41.307 | INFO     | __main__:main:115 - Loading test metadata
2021-11-18 19:12:41.313 | INFO     | __main__:main:119 - Found 10 chips
2021-11-18 19:12:41.314 | INFO     | __main__:main:121 - Generating predictions in batches
  0%|          | 0/1 [00:00<?, ?it/s]/srv/conda/envs/condaenv/lib/python3.9/site-packages/rasterio/__init__.py:220: NotGeoreferencedWarning: Dataset has no geotransform, gcps, or rpcs. The identity matrix be returned.
  s = DatasetReader(path, driver=driver, sharing=sharing, **kwargs)
100%|██████████| 1/1 [00:08<00:00,  8.81s/it]
2021-11-18 19:12:50.137 | INFO     | __main__:main:124 - Saving predictions to /codeexecution/submission
100%|██████████| 10/10 [00:00<00:00, 169.59it/s]
2021-11-18 19:12:50.197 | INFO     | __main__:main:126 - Saved 10 predictions
+ echo 'Testing that submission is valid'
Testing that submission is valid
+ conda run -n condaenv pytest -v tests/test_submission.py
============================= test session starts ==============================
platform linux -- Python 3.9.7, pytest-6.2.4, py-1.11.0, pluggy-0.13.1 -- /srv/conda/envs/condaenv/bin/python
cachedir: .pytest_cache
rootdir: /codeexecution
collecting ... collected 3 items

tests/test_submission.py::test_all_files_in_format_have_corresponding_submission_file PASSED [ 33%]
tests/test_submission.py::test_no_unexpected_tif_files_in_submission PASSED [ 66%]
tests/test_submission.py::test_file_sizes_are_within_limit PASSED        [100%]

=============================== warnings summary ===============================
../srv/conda/envs/condaenv/lib/python3.9/site-packages/skimage/data/__init__.py:111: 29 warnings
  /srv/conda/envs/condaenv/lib/python3.9/site-packages/skimage/data/__init__.py:111: DeprecationWarning:
      Importing file_hash from pooch.utils is DEPRECATED. Please import from the
      top-level namespace (`from pooch import file_hash`) instead, which is fully
      backwards compatible with pooch >= 0.1.

    return file_hash(path) == expected_hash

-- Docs: https://docs.pytest.org/en/stable/warnings.html
======================== 3 passed, 29 warnings in 1.14s ========================

+ echo 'Compressing files in a gzipped tar archive for submission'
Compressing files in a gzipped tar archive for submission
+ cd ./submission
+ tar czf ./submission.tar.gz 0000.tif 0001.tif 0002.tif 0003.tif 0004.tif 0005.tif 0006.tif 0007.tif 0008.tif 0009.tif
+ rm ./0000.tif ./0001.tif ./0002.tif ./0003.tif ./0004.tif ./0005.tif ./0006.tif ./0007.tif ./0008.tif ./0009.tif
+ cd ..
+ echo '... finished'
... finished
+ du -h submission/submission.tar.gz
376K    submission/submission.tar.gz
+ echo '================ END ================'
================ END ================
+ cp /codeexecution/submission/log.txt /tmp/log
+ exit 0
```

## (1) Testing your submission locally

Your submission will run inside a Docker container, a virtual operating system that allows for a consistent software environment across machines. This means that if your submission successfully runs in the container on your local machine, you can be pretty sure it will successfully run when you make an official submission to the DrivenData site.

In Docker parlance, your computer is the "host" that runs the container. The container is isolated from your host machine, with the exception of the following directories:

 - the `data` directory on the host machine is mounted in the container as a read-only directory `/codeexecution/data`
 - the `submission` directory on the host machine is mounted in the container as `/codeexecution/submission`

When you make a code submission, the code execution platform will unzip your submission assets to the `/codeexecution` folder. This must result in a `main.py` in the top level `/codeexecution` working directory. (Important: make sure your `main.py` is included at the top level of your zipped submission and not in a directory.)

On the official code execution platform, we will take care of mounting the data―you can assume your submission will have access to `/codeexecution/data/test_features` and `/codeexecution/data/test_metadata.csv`. You are responsible for creating the submission script that will read from `/codeexecution/data` and write out `.tif`s to `/codeexecution/predictions`. Once your code finishes, we run some validation tests on your predictions and then the script will compress all the `.tif`s into a tar archive to be sent to the platform for scoring.

For reference, here is the relevant directory structure inside the container. **Your `main.py` should read from `/codeexecution/data/test_features` and write to `/codeexecution/predictions` in order to generate a valid submission.**

```
$ tree /codeexecution
.
├── data
│   ├── test_features  <-- read chips from this directory
│   │   ├── aaaa  <-- your code makes predictions for each chip, e.g., aaaa, ..., zzzz
│   │   │   ├── B02.tif
│   │   │   ├── B03.tif
│   │   │   ├── B04.tif
│   │   │   └── B08.tif
│   │   ├── ...
│   │   └── zzzz
│   │       ├── B02.tif
│   │       ├── B03.tif
│   │       ├── B04.tif
│   │       └── B08.tif
│   └── test_metadata.csv
├── main.py  <-- your code submission main.py and any additional assets
├── my_model.py  <-- additional assets from your submission.zip
├── ...  <-- additional assets from your submission.zip
├── predictions  <-- write your predictions to this directory
│   ├── aaaa.tif  <-- your predicted cloud cover masks
│   ├── ...
│   └── zzzz.tif
└── submission
    └── log.txt  <-- log messages emitted while running your code
```

There is one important difference between your local test runtime and the official code execution runtime. **In the real competition runtime, all internet access is blocked except to the Planetary Computer STAC API.** `make test-submission` does not impose the same network restrictions. Any web requests outside of the Planetery Computer STAC API will work in the local test runtime, but fail in the actual competition runtime. It's up to you to make sure that your code only makes requests to the Planetary Computer API and no other web resources.

If you are not making calls to the Planetary Computer API, you can test your submission _without_ internet access by running `BLOCK_INTERNET=true make test-submission`.

**A note for models that download pre-trained weights from the internet**: It is common for models to download some of their weights from the internet. Since submissions do not have open access to the internet, you will need to include all weights along with your `submission.zip` and make sure that your code loads them from disk and does not try to download them from the internet. For example, PyTorch uses a local cache which by default is saved to `~/.cache/torch`.

```sh
# Copy your local pytorch cache into submission_src/assets
cp -R ~/.cache/torch submission_src/assets/

# Zip it all up in your submission.zip
zip -r submission.zip submission_src
```

When the platform runs your code, it will extract `assets` to `/codeexecution/assets`. You'll need to tell PyTorch to use your custom cache directory instead of `~/.cache/torch` by setting the `TORCH_HOME` environment variable in your Python code (in `main.py` for example).

```python
import os
os.environ["TORCH_HOME"] = "/codeexecution/assets/torch"
```

Now PyTorch will load the model weights from the local cache, and your submission will run correctly in the code execution environment.


### Implement your solution

In order to test your code submission, you will need a code submission! Implement your solution as a Python script named `main.py`. Next, create a `submission.zip` file containing your code and model assets.

**Note: You will implement all of your training and experiments on your machine. It is highly recommended that you use the same package versions that are in the runtime conda environments ([Python (CPU)](runtime/environment-cpu.yml), [Python (GPU)](runtime/environment-gpu.yml). If you don't wish to use Docker these exact packages can be installed with `conda`.**

The [submission format page](https://www.drivendata.org/competitions/83/cloud-cover/page/412/) contains the detailed information you need to prepare your code submission.

### Example benchmark submission

The [benchmark solution](todo blogpost link) can serve as a concrete example of how to create a submission. Use `make pack-benchmark` to create the benchmark submission from the source code. The command zips everything in the `benchmark` folder and saves the zip archive to `submission/submission.zip`. To prevent losing your work, this command will not overwrite an existing submission. To generate a new submission, you will first need to remove the existing `submission/submission.zip`.

### Running your submission

Now you can make sure your submission runs locally prior to submitting it to the platform. Make sure you have the [prerequisites](#prerequisites) installed, and have [pset up some fake data](#fake-test-data). Then, run the following command to download the official image:

```bash
make pull
```

Again, make sure you have packed up your solution in `submission/submission.zip` (or generated the sample submission with `make pack-submission`), then try running it:

```bash
make test-submission
```

This will start the container, mount the local data and submission folders as folders within the container, and follow the same steps that will run on the platform to unpack your submission and run your code.

### Scoring your submission.tar.gz

We have included a [metric script](https://github.com/drivendataorg/cloud-cover-runtime/blob/main/runtime/scripts/metric.py) that shows how the metric can be calculated locally on your training data the same way we score your actual submitted files. You can run this, too:

```bash
# unzip your submission so the .tifs are actually around then come back up to the project root
$ cd submission && tar xvf submission.tar.gz && cd ..

# show usage instructions
$ python runtime/scripts/metric.py --help
Usage: metric.py [OPTIONS] SUBMISSION_DIR ACTUAL_DIR

  Given a directory with the predicted mask files (all values in {0, 1}) and
  the actual mask files (all values in {0, 1}), get the overall
  intersection-over-union score

Arguments:
  SUBMISSION_DIR  [required]
  ACTUAL_DIR      [required]

$ python runtime/scripts/metric.py submission runtime/data/test_labels
2021-11-18 14:17:19.934 | INFO     | __main__:main:44 - calculating score for 10 image pairs ...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 723.88it/s]
2021-11-18 14:17:19.950 | SUCCESS  | __main__:main:46 - overall score: 0.2673616412947126
```

### Reviewing the logs

When you run `make test-submission` the logs will be printed to the terminal. They will also be written to the `submission` folder as `log.txt`. You can always review that file and copy any versions of it that you want from the `submission` folder. The errors there will help you to determine what changes you need to make sure your code executes successfully.

## (2) Updating the runtime packages

We accept contributions to add dependencies to the runtime environment. To do so, follow these steps:

1. Fork this repository
2. Make your changes
3. Test them and commit using git
3. Open a pull request to this repository

If you're new to the GitHub contribution workflow, check out [this guide by GitHub](https://guides.github.com/activities/forking/).

### Adding new Python packages

We use [conda](https://docs.conda.io/en/latest/) to manage Python dependencies. Add your new dependencies to both `runtime/environment-cpu.yml` and `runtime/environment-gpu.yml`.

Your new dependency should follow the format in the yml.

### Opening a pull request

After making and testing your changes, commit your changes and push to your fork. Then, when viewing the repository on github.com, you will see a banner that lets you open the pull request. For more detailed instructions, check out [GitHub's help page](https://help.github.com/en/articles/creating-a-pull-request-from-a-fork).

Once you open the pull request, Github Actions will automatically try building the Docker images with your changes and run the tests in `runtime/tests`. These tests can take up to 30 minutes to run through, and may take longer if your build is queued behind others. You will see a section on the pull request page that shows the status of the tests and links to the logs.

You may be asked to submit revisions to your pull request if the tests fail, or if a DrivenData team member asks for revisions. Pull requests won't be merged until all tests pass and the team has reviewed and approved the changes.

---

## Good luck; have fun!

Thanks for reading! Enjoy the competition, and [hit up the forums](https://community.drivendata.org/) if you have any questions!
