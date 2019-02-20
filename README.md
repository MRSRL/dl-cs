# Compressed Sensing: From Research to Clinical Practice with Data-Driven Learning

## Setup

This project uses the open source toolbox for computational magnetic resonance imaging: BART. The main `bart` binary should be installed before proceeding. More information about BART along with installation instructions can be found here: <https://mrirecon.github.io/bart/>.

Install the required python packages (tested with python 3.6):

```bash
pip install -r requirements.txt
```

Fully sampled datasets can be downloaded from <http://mridata.org> using the python script and text file.

```bash
python3 data_prep.py --verbose mridata_org.txt
```

The download and pre-processing will take some time! This script will create a `data` folder with the following sub-folders:

* `raw/ismrmrd`: Contains files with ISMRMRD files directly from <http://mridata.org>
* `raw/cfl`: Contains the data converted to the bart format (`cfl` and `hdr` files)
* `tfrecord/train`: Training examples converted to TFRecords
* `tfrecord/validate`: Validation examples converted to TFRecords
* `tfrecord/test`: Test examples converted to TFRecords
* `test_cfl`: Sub-sampled volumetric test examples in bart format
* `masks`: Sampling masks generated using bart in bart format

All TFRecords contain fully sampled raw k-space data and sensitivity maps.

## Training

The training can be performed using the following command.

```bash
python3 --model_dir summary/model recon_train.py
```

All the parameters (dimensions and etc) assume that the training is performed with the knee datasets from mridata.org. See the `--help` flag for more information on how to adjust the training for new datasets.

## Inference

For bart file input `kspace_input.{hdr,cfl}`, the data can be reconstructed with the following command. The test data in `data/test_cfl` can be used to test the inference script.

```bash
python3 recon_run.py summary/model kspace_input kspace_output
```