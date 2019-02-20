"""Wrapper for bart functions"""
import os
import subprocess
import numpy as np
from utils import cfl

BIN_BART = "bart"


def remove_bart_files(filenames):
    for f in filenames:
        os.remove(f + ".hdr")
        os.remove(f + ".cfl")


def espirit(kspace, flags="-c 1e-9 -m 1"):
    """Runs bart ecalib command."""
    randnum = np.random.randint(1e8)
    fileinput = "tmp.in.{}".format(randnum)
    fileoutput = "tmp.out.{}".format(randnum)
    cfl.write(fileinput, kspace)
    espirit_f(fileinput, fileoutput, flags)
    sensemap = np.squeeze(cfl.read(fileoutput))
    remove_bart_files([fileoutput, fileinput])
    return sensemap


def espirit_f(filekspace, filesensemap, flags="-c 1e-9 -m 1"):
    cmd = "{} ecalib {} {} {}".format(BIN_BART, flags, filekspace, filesensemap)
    subprocess.check_output(['bash', '-c', cmd])


def poisson_f(filemask, flags="-C 10 -Y 320 -Z 256 -y 2 -z 2", random_seed=0):
    cmd = "{} poisson {} -s {} {}".format(BIN_BART, flags, random_seed, filemask)
    subprocess.check_output(['bash', '-c', cmd])
