from __future__ import annotations

import os
ENV = {"TF_FORCE_UNIFIED_MEMORY":"1", "XLA_PYTHON_CLIENT_MEM_FRACTION":"4.0"}
for k,v in ENV.items():
    if k not in os.environ: os.environ[k] = v

import warnings
from Bio import BiopythonDeprecationWarning # what can possibly go wrong...
warnings.simplefilter(action='ignore', category=BiopythonDeprecationWarning)

import json
import logging
import math
import random
import sys
import time
import zipfile
import shutil
import pickle
import gzip
import lzma
import uuid

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors
import re
import hashlib
from datetime import datetime, timezone

from threading import Thread, Event
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from io import StringIO

import importlib_metadata
import numpy as np
import pandas

import tensorflow as tf

try:
    import alphafold
except ModuleNotFoundError:
    raise RuntimeError(
        "\n\nalphafold is not installed. Please run `pip install colabfold[alphafold]`\n"
    )

from alphafold.common import protein, residue_constants

# delay imports of tensorflow, jax and numpy
# loading these for type checking only can take around 10 seconds just to show a CLI usage message
if TYPE_CHECKING:
    import haiku
    from alphafold.model import model
    from numpy import ndarray

from alphafold.common.protein import Protein
from alphafold.data import (
    feature_processing,
    msa_pairing,
    pipeline,
    pipeline_multimer,
    templates,
)
from alphafold.data.tools import hhsearch
from colabfold.citations import write_bibtex
from colabfold.download import default_data_dir, download_alphafold_params
from colabfold.utils import (
    ACCEPT_DEFAULT_TERMS,
    DEFAULT_API_SERVER,
    NO_GPU_FOUND,
    CIF_REVISION_DATE,
    get_commit,
    safe_filename,
    setup_logging,
    CFMMCIFIO,
)

from Bio.PDB import MMCIFParser, PDBParser, MMCIF2Dict

# logging settings
logger = logging.getLogger(__name__)
import jax
import jax.numpy as jnp
logging.getLogger('jax._src.lib.xla_bridge').addFilter(lambda _: False)


c1 = (47, 117, 214, 255)
c1 = tuple(ti/255 for ti in c1)

c2 = (255, 255, 255, 255)
c2 = tuple(ti/255 for ti in c2)

c3 = (237, 64, 64,255)
c3 = tuple(ti/255 for ti in c3)

norm = plt.Normalize(-2,2)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [c1, c2, c3])

def plot_ticks(Ls, line_thickness=0.5):
    Ln = sum(Ls)
    L_prev = 0
    for L_i in Ls[:-1]:
        L = L_prev + L_i
        L_prev += L_i
        plt.plot([0, Ln], [L, L], color="black", linewidth=line_thickness)
        plt.plot([L, L], [0, Ln], color="black", linewidth=line_thickness)
    ticks = np.cumsum([0] + Ls)
    ticks = (ticks[1:] + ticks[:-1]) / 2
    plt.yticks(ticks)


def plot_pae(pae, pae_filename, Ls=None,img_size = 600):

    fig = plt.figure(num=1, clear=True, facecolor='w')
    width = img_size/300
    fig.set_size_inches(width, width)
    ax = fig.add_subplot()
    Ln = pae.shape[0]
    ax.imshow(pae,cmap=cmap,vmin=0,vmax=30,extent=(0, Ln, Ln, 0))
    ax.axis('off')
    if Ls is not None and len(Ls) > 1: plot_ticks(Ls)
    plt.savefig(pae_filename, bbox_inches='tight', pad_inches = 0, dpi =300)
    fig.clear()
    plt.close(fig)
    img = Image.open(pae_filename).convert("RGB");
    img.save(pae_filename.replace("png", "webp"), "webp", lossless=True)
    os.remove(pae_filename)

def patch_openmm():
    from simtk.openmm import app
    from simtk.unit import nanometers, sqrt

    # applied https://raw.githubusercontent.com/deepmind/alphafold/main/docker/openmm.patch
    # to OpenMM 7.5.1 (see PR https://github.com/openmm/openmm/pull/3203)
    # patch is licensed under CC-0
    # OpenMM is licensed under MIT and LGPL
    # fmt: off
    def createDisulfideBonds(self, positions):
        def isCyx(res):
            names = [atom.name for atom in res._atoms]
            return 'SG' in names and 'HG' not in names
        # This function is used to prevent multiple di-sulfide bonds from being
        # assigned to a given atom.
        def isDisulfideBonded(atom):
            for b in self._bonds:
                if (atom in b and b[0].name == 'SG' and
                    b[1].name == 'SG'):
                    return True

            return False

        cyx = [res for res in self.residues() if res.name == 'CYS' and isCyx(res)]
        atomNames = [[atom.name for atom in res._atoms] for res in cyx]
        for i in range(len(cyx)):
            sg1 = cyx[i]._atoms[atomNames[i].index('SG')]
            pos1 = positions[sg1.index]
            candidate_distance, candidate_atom = 0.3*nanometers, None
            for j in range(i):
                sg2 = cyx[j]._atoms[atomNames[j].index('SG')]
                pos2 = positions[sg2.index]
                delta = [x-y for (x,y) in zip(pos1, pos2)]
                distance = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2])
                if distance < candidate_distance and not isDisulfideBonded(sg2):
                    candidate_distance = distance
                    candidate_atom = sg2
            # Assign bond to closest pair.
            if candidate_atom:
                self.addBond(sg1, candidate_atom)
    # fmt: on
    app.Topology.createDisulfideBonds = createDisulfideBonds


def crop_msa(msa_str, start, end, drop_empty = True):
    
    new_lines = []
    active_id = None
    lines = msa_str.splitlines()
    for line in lines:
        if line[0] != '>' and active_id is not None:
            new_line = re.sub(r"[a-z\n]",'', line)[start - 1:end]

            if drop_empty:
                if len(new_line.replace('-', '')) > 0:
                    new_lines.append(active_id)
                    new_lines.append(new_line)
            else:
                new_lines.append(active_id)
                new_lines.append(new_line)
        else:
            active_id = line

    return "\n".join(new_lines)


global_template_a3m_lines_mmseqs2_storage = {}
global_unpaired_a3m_lines_storage = {}

def aa_seq_to_id(sequence):
    sequence_bytes = sequence.encode('utf-8')
    md5_hash = hashlib.md5(sequence_bytes).hexdigest()
    identifier = ''.join(c for c in md5_hash if c.isalnum())
    return identifier



def get_msa_and_templates_v2(
    jobname: str,
    query_sequences: Union[str, List[str]],
    result_dir: Path,
    msa_mode: str,
    use_templates: bool,
    custom_template_path: str,
    pair_mode: str,
    host_url: str = DEFAULT_API_SERVER,
    use_proxy:bool=False
) -> Tuple[
    Optional[List[str]], Optional[List[str]], List[str], List[int], List[Dict[str, Any]]
]:
    from colabfold.colabfold import run_mmseqs2

    use_env = msa_mode == "mmseqs2_uniref_env"
    if isinstance(query_sequences, str): query_sequences = [query_sequences]

    #DPOLQ_HUMAN.1-500.__MSH3_HUMAN__1637aa
    comps = re.split('[^0-9]\-[^0-9]|__', jobname)
    chain_regions = []
    valid_name = len(comps) - 1 == len(query_sequences)
    if valid_name:
        for i in range(0, len(comps) - 1):
            c = comps[i]
            if '-' and '.' in c:
                r = c.split('.')[1]
                if len(r.split('-')) != 2:
                    valid_name = False
                    break
                start, end = r.split('-')
                start = int(start)
                end = int(end)
                if (start < 1 or end < start or end > len(query_sequences[i])):
                    valid_name = False
                chain_regions.append([int(start),int(end)])
            else:
                chain_regions.append([1,-1])


    if not os.path.exists('colabfold_template_store'):
        os.makedirs('colabfold_template_store')

    if not os.path.exists('colabfold_unpaired_msa_store'):
        os.makedirs('colabfold_unpaired_msa_store')


    query_seqs_unique = []
    for x in query_sequences:
        query_seqs_unique.append(x)

    # determine how many times is each sequence is used Running on GPU
    query_seqs_cardinality = [1] * len(query_seqs_unique)

    # get template features
    template_features = {}
    if use_templates:

        seqs_to_fetch = []
        indices_fetched = []

        for index in range(0, len(query_seqs_unique)):
            msa_seq = query_seqs_unique[index]
            actual_seq = msa_seq
            if valid_name and chain_regions[index][1] != -1:
                actual_seq = query_seqs_unique[index][chain_regions[index][0] - 1:chain_regions[index][1]]

            stored_feature = None
            id = aa_seq_to_id(actual_seq)
            stored_templates_filename = f'colabfold_template_store/{id}.pkl'
            if os.path.isfile(stored_templates_filename) and (time.time() - os.path.getmtime(stored_templates_filename)) > 20:
                with open(stored_templates_filename, 'rb') as f:
                    stored_feature = pickle.load(f)

            template_features[index] = stored_feature
            if stored_feature is None:
                seqs_to_fetch.append(msa_seq)
                indices_fetched.append(index)

        if len(seqs_to_fetch) > 0:

            a3m_lines_mmseqs2, template_paths = run_mmseqs2(
                seqs_to_fetch,
                str(result_dir.joinpath(jobname)),
                use_env,
                use_templates=True,
                host_url=host_url,
                use_proxy=False)
            if valid_name:
                for i, index in enumerate(indices_fetched):
                    region = chain_regions[index]
                    if region[1] != -1:
                        logger.info(f"cropping template MSA from {region[0]} to {region[1]}")
                        a3m_lines_mmseqs2[i] = crop_msa(a3m_lines_mmseqs2[i], region[0], region[1])
                        if len(a3m_lines_mmseqs2[i]) == 0:
                            template_paths[i] = None
                    
            if custom_template_path is not None:
                template_paths = {}
                for index in range(0, len(query_seqs_unique)):
                    template_paths[index] = custom_template_path
            if template_paths is None:
                logger.info("No template detected")
                for i, index in enumerate(indices_fetched):
                    seq = seqs_to_fetch[i]
                    template_feature = mk_mock_template(seq)
                    template_features[index] = template_feature
                    id = aa_seq_to_id(seq)
                    logger.info(f"Writing out empty template with ID: {id} for seq: {seq}")
                    with open(f'colabfold_template_store/{id}.pkl', 'wb') as f:
                        pickle.dump(template_feature, f)

            else:

                for i, index in enumerate(indices_fetched):
                    
                    seq = seqs_to_fetch[i]
                    if valid_name and chain_regions[index][1] != -1:
                        seq = seq[chain_regions[index][0] - 1:chain_regions[index][1]]

                    if template_paths[i] is not None:
                        logger.info(f"Generating new template")
                        lines = a3m_lines_mmseqs2[i].splitlines()
                        logger.info(f"TEMPLATE MSA LINE 0: {lines[0]}")
                        logger.info(f"TEMPLATE MSA LINE 1: {lines[1]}")
                        
                        template_feature = mk_template(a3m_lines_mmseqs2[i],template_paths[i],seq)

                        if len(template_feature["template_domain_names"]) == 0:
                            template_feature = mk_mock_template(seq)
                            logger.info(f"Sequence {index} found no templates")
                        else:
                            logger.info(f"Sequence {index} found templates: {template_feature['template_domain_names'].astype(str).tolist()}")
                    else:
                        template_feature = mk_mock_template(seq)
                        logger.info(f"Sequence {index} found no templates")

                    template_features[index] = template_feature
                    id = aa_seq_to_id(seq)
                    with open(f'colabfold_template_store/{id}.pkl', 'wb') as f:
                        pickle.dump(template_feature, f)

        else:
            logger.info("No need to fetch templates, already have finished features ready!")
    else:

        for index in range(0, len(query_seqs_unique)):
            seq = query_seqs_unique[index]
            if valid_name and chain_regions[index][1] != -1:
                seq = query_seqs_unique[index][chain_regions[index][0] - 1:chain_regions[index][1]]
            template_feature = mk_mock_template(seq)
            template_features[index] = template_feature
    
    final_template_features = []
    for index in range(0, len(query_seqs_unique)):
        final_template_features.append(template_features[index])

    template_features = final_template_features
    # template_features = list(template_features.values())

    if len(query_sequences) == 1:
        pair_mode = "none"

    if pair_mode == "none" or pair_mode == "unpaired" or pair_mode == "unpaired_paired":
        if msa_mode == "single_sequence":
            a3m_lines = []
            num = 101
            for i, seq in enumerate(query_seqs_unique):
                if valid_name and chain_regions[index][1] != -1:
                    seq = query_seqs_unique[index][chain_regions[index][0] - 1:chain_regions[index][1]]
                a3m_lines.append(f">{num + i}\n{seq}")
        else:
            # find normal a3ms
            unpaired_msa_lines = {}

            seqs_to_fetch = []
            indices_fetched = []

            for index in range(0, len(query_seqs_unique)):
                seq = query_seqs_unique[index]

                stored_msa = None
                id = aa_seq_to_id(seq)
                stored_msa_filename = f'colabfold_unpaired_msa_store/{id}.pkl'
                if os.path.isfile(stored_msa_filename) and os.path.getsize(stored_msa_filename) > 100 and (time.time() - os.path.getmtime(stored_msa_filename)) > 20:
                    logger.info(f"Used {id} for an unpaired msa")
                    with open(stored_msa_filename, 'rb') as f:
                        stored_msa = pickle.load(f)
                    
                    if stored_msa.splitlines()[1] != seq:
                        logger.info(f"Unpaired MSA contained a mismatch, will get correct sequence now")
                        stored_msa = None

                if stored_msa is not None:
                    msa_str = stored_msa
                    if valid_name and chain_regions[index][1] != -1:
                        msa_str = crop_msa(msa_str, chain_regions[index][0], chain_regions[index][1])
                    unpaired_msa_lines[index] = msa_str
                else:
                    seqs_to_fetch.append(seq)
                    indices_fetched.append(index)
            
            if len(seqs_to_fetch) > 0:
 
                logger.info(f"Fetching unpaired MSAs for sequences:{seqs_to_fetch}")
                newly_fetched_a3ms = run_mmseqs2(
                    seqs_to_fetch,
                    str(result_dir.joinpath(jobname)),
                    use_env,
                    use_pairing=False,
                    host_url=host_url,
                    use_proxy=False)
                
                for i, index in enumerate(indices_fetched):
                    seq = query_seqs_unique[index]
                    logger.info(f"Working on seq:{seq}")
                    id = aa_seq_to_id(seq)
                    logger.info(f"Seq hash id is:{id}")
                    stored_msa_filename = f"colabfold_unpaired_msa_store/{id}.pkl"
                    with open(stored_msa_filename, 'wb') as f:
                        logger.info(f"writing out new unpaired MSA with ID:{id}")
                        pickle.dump(newly_fetched_a3ms[i], f)

                    msa_str = newly_fetched_a3ms[i]
                    if valid_name and chain_regions[index][1] != -1:
                        region = chain_regions[index]
                        logger.info(f"cropping msa seq from:{region[0]} to {region[1]} ")
                        msa_str = crop_msa(msa_str, region[0], region[1])
                    
                    unpaired_msa_lines[index] = msa_str
                
                final_lines = []
                for index in range(0, len(query_seqs_unique)):
                    final_lines.append(unpaired_msa_lines[index])
                a3m_lines = final_lines
                # a3m_lines = list(unpaired_msa_lines.values())
            else:
                logger.info("No need to fetch unpaired MSAs, already have finished MSAs ready!")
                final_lines = []
                for index in range(0, len(query_seqs_unique)):
                    final_lines.append(unpaired_msa_lines[index])
                a3m_lines = final_lines
                # a3m_lines = list(unpaired_msa_lines.values())

    else:
        a3m_lines = None

    if msa_mode != "single_sequence" and ( pair_mode == "paired" or pair_mode == "unpaired_paired"):
        # find paired a3m if not a homooligomers
        if len(query_seqs_unique) > 1:
            paired_a3m_lines = run_mmseqs2(
                query_seqs_unique,
                str(result_dir.joinpath(jobname)),
                use_env,
                use_pairing=True,
                host_url=host_url,
               use_proxy=False
            )

            if valid_name:
                for index in range(0, len(query_seqs_unique)):
                    region = chain_regions[index]
                    if region[1] != -1:
                        paired_a3m_lines[index] = crop_msa(paired_a3m_lines[index], region[0], region[1], False)
        else:
            # homooligomers
            num = 101
            paired_a3m_lines = []
            for i in range(0, query_seqs_cardinality[0]):
                paired_a3m_lines.append(f">{num+i}\n{query_seqs_unique[0]}\n")
    else:
        paired_a3m_lines = None

    if valid_name:
        for index in range(0, len(query_seqs_unique)):
            region = chain_regions[index]
            if region[1] != -1:
                query_seqs_unique[index] = query_seqs_unique[index][region[0]-1:region[1]]

    for index in range(0, len(query_seqs_unique)):
        if len(a3m_lines[index]) == 0:
            print("MSA file for index has no lines" + str(index))

    return (
        a3m_lines,
        paired_a3m_lines,
        query_seqs_unique,
        query_seqs_cardinality,
        template_features,)




def get_msa_and_templates_v3(
    jobname: str,
    query_sequences: Union[str, List[str]],
    result_dir: Path,
    msa_mode: str,
    use_templates: bool,
    custom_template_path: str,
    pair_mode: str,
    host_url: str = DEFAULT_API_SERVER,
    saved_template_features_folder:str = None,
    saved_unpaired_msa_features_folder:str = None,
) -> Tuple[
    Optional[List[str]], Optional[List[str]], List[str], List[int], List[Dict[str, Any]]
]:
    from colabfold.colabfold import run_mmseqs2

    use_env = msa_mode == "mmseqs2_uniref_env"
    if isinstance(query_sequences, str): query_sequences = [query_sequences]

    # remove duplicates before searching
    query_seqs_unique = []
    for x in query_sequences:
        if x not in query_seqs_unique:
            query_seqs_unique.append(x)

    # determine how many times is each sequence is used
    query_seqs_cardinality = [0] * len(query_seqs_unique)
    for seq in query_sequences:
        seq_idx = query_seqs_unique.index(seq)
        query_seqs_cardinality[seq_idx] += 1

    #-------------------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------------------
    #TEMPLATES FETCH-------------------------------------------------------------------------------------------------
    template_features = {}
    if use_templates:

        if(saved_template_features_folder):

            for index in range(0, len(query_seqs_unique)):
                seq = query_seqs_unique[index]
                seq_id = aa_seq_to_id(seq)
                file_path = os.path.join(saved_template_features_folder, f'{seq_id}.pkl')
                if os.path.isfile(file_path):
                    with open(file_path, 'rb') as f:
                        store_template_features = pickle.load(f)
                        if 'template_aatype' in store_template_features and store_template_features['template_aatype'][0].shape[0] == len(seq):
                            #double check that this is correct sequence via length
                            template_features[index] = store_template_features
                            logger.info(f"Retreived sequence {index}: {seq} from template file {file_path}")

        search_ix_to_template_ix = {}
        seqs_to_search = []
        search_ix = 0
        for index in range(0, len(query_seqs_unique)):
            if(index in template_features): continue

            seqs_to_search.append(query_seqs_unique[index])
            search_ix_to_template_ix[search_ix] = index
            search_ix += 1

        if len(seqs_to_search) > 0:
            try:
                a3m_lines_mmseqs2, template_paths = run_mmseqs2(
                    seqs_to_search,
                    str(result_dir.joinpath(jobname)),
                    use_env,
                    use_templates=True,
                    host_url=host_url,
                )
            except:
                return None
                
            if custom_template_path is not None:
                template_paths = {}
                for seq_ix in range(0, len(seqs_to_search)):
                    template_paths[seq_ix] = custom_template_path
            if template_paths is None:
                logger.info("No template detected")
                for seq_ix in range(0, len(seqs_to_search)):
                    template_feature = mk_mock_template(seqs_to_search[seq_ix])
                    template_ix = search_ix_to_template_ix[seq_ix]

                    template_features[template_ix] = template_feature
            else:
                for seq_ix in range(0, len(seqs_to_search)):
                    if template_paths[seq_ix] is not None:
                        template_feature = mk_template(
                            a3m_lines_mmseqs2[seq_ix],
                            template_paths[seq_ix],
                            seqs_to_search[seq_ix],
                        )
                        if len(template_feature["template_domain_names"]) == 0:
                            template_feature = mk_mock_template(seqs_to_search[seq_ix])
                            logger.info(f"Sequence {seq_ix} found no templates")
                        else:
                            logger.info(
                                f"Sequence {seq_ix} found templates: {template_feature['template_domain_names'].astype(str).tolist()}"
                            )
                    else:
                        template_feature = mk_mock_template(seqs_to_search[seq_ix])
                        logger.info(f"Sequence {seq_ix} found no templates")

                    template_ix = search_ix_to_template_ix[seq_ix]
                    template_features[template_ix] = template_feature
    else:
        for index in range(0, len(query_seqs_unique)):
            template_feature = mk_mock_template(query_seqs_unique[index])
            template_features[index] = template_feature

    final_template_features = [template_features[ix] for ix in range(0, len(query_seqs_unique))]
    template_features = final_template_features

    if len(query_sequences) == 1:
        pair_mode = "none"


    #-------------------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------------------
    #UNPAIRED MSA FETCH-------------------------------------------------------------------------------------------------




    if pair_mode == "none" or pair_mode == "unpaired" or pair_mode == "unpaired_paired":
        if msa_mode == "single_sequence":
            a3m_lines = []
            num = 101
            for i, seq in enumerate(query_seqs_unique):
                a3m_lines.append(f">{num + i}\n{seq}")
        else:

            a3m_lines = {}
            seqs_to_search = []
            

            if(saved_unpaired_msa_features_folder):
                for index in range(0, len(query_seqs_unique)):
                    seq = query_seqs_unique[index]
                    seq_id = aa_seq_to_id(seq)
                    file_path = os.path.join(saved_unpaired_msa_features_folder, f'{seq_id}.pkl')
                    if os.path.isfile(file_path):
                        with open(file_path, 'rb') as f:
                            saved_msa_str = pickle.load(f)
                            saved_msa_lines = saved_msa_str.split('\n')
                            msa_seq = saved_msa_lines[1]
                            if(seq == msa_seq):
                                a3m_lines[index] = saved_msa_str
                                logger.info(f"Retreived unpaired MSA sequence {index}: {seq} from template file {file_path}")


            
            search_ix_to_msa_ix = {}
            seqs_to_search = []
            search_ix = 0
            for index in range(0, len(query_seqs_unique)):
                if(index in a3m_lines): continue
                seqs_to_search.append(query_seqs_unique[index])
                search_ix_to_msa_ix[search_ix] = index
                search_ix += 1

            if len(seqs_to_search) > 0:
                # find normal a3ms

                try:
                    new_a3m_lines = run_mmseqs2(
                        seqs_to_search,
                        str(result_dir.joinpath(jobname)),
                        use_env,
                        use_pairing=False,
                        host_url=host_url,
                    )
                except:
                    return None

                for seq_ix in range(0, len(seqs_to_search)):
                    msa_ix = search_ix_to_msa_ix[seq_ix]
                    a3m_lines[msa_ix] = new_a3m_lines[seq_ix]


            final_a3ms = [a3m_lines[ix] for ix in range(0, len(query_seqs_unique))]
            a3m_lines = final_a3ms

    else:
        a3m_lines = None



    #-------------------------------------------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------------------------------------------
    #PAIRED MSA FETCH-------------------------------------------------------------------------------------------------
    if msa_mode != "single_sequence" and (
        pair_mode == "paired" or pair_mode == "unpaired_paired"
    ):
        # find paired a3m if not a homooligomers
        if len(query_seqs_unique) > 1:
            try:
                paired_a3m_lines = run_mmseqs2(
                    query_seqs_unique,
                    str(result_dir.joinpath(jobname)),
                    use_env,
                    use_pairing=True,
                    host_url=host_url,
                )
            except:
                return None
        else:
            # homooligomers
            num = 101
            paired_a3m_lines = []
            for i in range(0, query_seqs_cardinality[0]):
                paired_a3m_lines.append(f">{num+i}\n{query_seqs_unique[0]}\n")
    else:
        paired_a3m_lines = None


    return (
        a3m_lines,
        paired_a3m_lines,
        query_seqs_unique,
        query_seqs_cardinality,
        final_template_features,
    )





def mk_mock_template(
    query_sequence: Union[List[str], str], num_temp: int = 1
) -> Dict[str, Any]:
    ln = (
        len(query_sequence)
        if isinstance(query_sequence, str)
        else sum(len(s) for s in query_sequence)
    )
    output_templates_sequence = "A" * ln
    output_confidence_scores = np.full(ln, 1.0)

    templates_all_atom_positions = np.zeros(
        (ln, templates.residue_constants.atom_type_num, 3)
    )
    templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
    templates_aatype = templates.residue_constants.sequence_to_onehot(
        output_templates_sequence, templates.residue_constants.HHBLITS_AA_TO_ID
    )
    template_features = {
        "template_all_atom_positions": np.tile(
            templates_all_atom_positions[None], [num_temp, 1, 1, 1]
        ),
        "template_all_atom_masks": np.tile(
            templates_all_atom_masks[None], [num_temp, 1, 1]
        ),
        "template_sequence": [f"none".encode()] * num_temp,
        "template_aatype": np.tile(np.array(templates_aatype)[None], [num_temp, 1, 1]),
        "template_confidence_scores": np.tile(
            output_confidence_scores[None], [num_temp, 1]
        ),
        "template_domain_names": [f"none".encode()] * num_temp,
        "template_release_date": [f"none".encode()] * num_temp,
        "template_sum_probs": np.zeros([num_temp], dtype=np.float32),
    }
    return template_features

def mk_template(
    a3m_lines: str, template_path: str, query_sequence: str
) -> Dict[str, Any]:
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=template_path,
        max_template_date="2100-01-01",
        max_hits=20,
        kalign_binary_path="kalign",
        release_dates_path=None,
        obsolete_pdbs_path=None,
    )

    hhsearch_pdb70_runner = hhsearch.HHSearch(
        binary_path="hhsearch", databases=[f"{template_path}/pdb70"]
    )

    hhsearch_result = hhsearch_pdb70_runner.query(a3m_lines)
    hhsearch_hits = pipeline.parsers.parse_hhr(hhsearch_result)
    templates_result = template_featurizer.get_templates(
        query_sequence=query_sequence, hits=hhsearch_hits
    )
    return dict(templates_result.features)

def validate_and_fix_mmcif(cif_file: Path):
    """validate presence of _entity_poly_seq in cif file and add revision_date if missing"""
    # check that required poly_seq and revision_date fields are present
    cif_dict = MMCIF2Dict.MMCIF2Dict(cif_file)
    required = [
        "_chem_comp.id",
        "_chem_comp.type",
        "_struct_asym.id",
        "_struct_asym.entity_id",
        "_entity_poly_seq.mon_id",
    ]
    for r in required:
        if r not in cif_dict:
            raise ValueError(f"mmCIF file {cif_file} is missing required field {r}.")
    if "_pdbx_audit_revision_history.revision_date" not in cif_dict:
        logger.info(
            f"Adding missing field revision_date to {cif_file}. Backing up original file to {cif_file}.bak."
        )
        shutil.copy2(cif_file, str(cif_file) + ".bak")
        with open(cif_file, "a") as f:
            f.write(CIF_REVISION_DATE)

def convert_pdb_to_mmcif(pdb_file: Path):
    """convert existing pdb files into mmcif with the required poly_seq and revision_date"""
    i = pdb_file.stem
    cif_file = pdb_file.parent.joinpath(f"{i}.cif")
    if cif_file.is_file():
        return
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(i, pdb_file)
    cif_io = CFMMCIFIO()
    cif_io.set_structure(structure)
    cif_io.save(str(cif_file))

def mk_hhsearch_db(template_dir: str):
    template_path = Path(template_dir)

    cif_files = template_path.glob("*.cif")
    for cif_file in cif_files:
        validate_and_fix_mmcif(cif_file)

    pdb_files = template_path.glob("*.pdb")
    for pdb_file in pdb_files:
        convert_pdb_to_mmcif(pdb_file)

    pdb70_db_files = template_path.glob("pdb70*")
    for f in pdb70_db_files:
        os.remove(f)

    with open(template_path.joinpath("pdb70_a3m.ffdata"), "w") as a3m, open(
        template_path.joinpath("pdb70_cs219.ffindex"), "w"
    ) as cs219_index, open(
        template_path.joinpath("pdb70_a3m.ffindex"), "w"
    ) as a3m_index, open(
        template_path.joinpath("pdb70_cs219.ffdata"), "w"
    ) as cs219:
        n = 1000000
        index_offset = 0
        cif_files = template_path.glob("*.cif")
        for cif_file in cif_files:
            with open(cif_file) as f:
                cif_string = f.read()
            cif_fh = StringIO(cif_string)
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("none", cif_fh)
            models = list(structure.get_models())
            if len(models) != 1:
                raise ValueError(
                    f"Only single model PDBs are supported. Found {len(models)} models."
                )
            model = models[0]
            for chain in model:
                amino_acid_res = []
                for res in chain:
                    if res.id[2] != " ":
                        raise ValueError(
                            f"PDB contains an insertion code at chain {chain.id} and residue "
                            f"index {res.id[1]}. These are not supported."
                        )
                    amino_acid_res.append(
                        residue_constants.restype_3to1.get(res.resname, "X")
                    )

                protein_str = "".join(amino_acid_res)
                a3m_str = f">{cif_file.stem}_{chain.id}\n{protein_str}\n\0"
                a3m_str_len = len(a3m_str)
                a3m_index.write(f"{n}\t{index_offset}\t{a3m_str_len}\n")
                cs219_index.write(f"{n}\t{index_offset}\t{len(protein_str)}\n")
                index_offset += a3m_str_len
                a3m.write(a3m_str)
                cs219.write("\n\0")
                n += 1

def pad_input(
    input_features: model.features.FeatureDict,
    model_runner: model.RunModel,
    model_name: str,
    pad_len: int,
    use_templates: bool,
) -> model.features.FeatureDict:
    from colabfold.alphafold.msa import make_fixed_size

    model_config = model_runner.config
    eval_cfg = model_config.data.eval
    crop_feats = {k: [None] + v for k, v in dict(eval_cfg.feat).items()}

    max_msa_clusters = eval_cfg.max_msa_clusters
    max_extra_msa = model_config.data.common.max_extra_msa
    # templates models
    if (model_name == "model_1" or model_name == "model_2") and use_templates:
        pad_msa_clusters = max_msa_clusters - eval_cfg.max_templates
    else:
        pad_msa_clusters = max_msa_clusters

    max_msa_clusters = pad_msa_clusters

    # let's try pad (num_res + X)
    input_fix = make_fixed_size(
        input_features,
        crop_feats,
        msa_cluster_size=max_msa_clusters,  # true_msa (4, 512, 68)
        extra_msa_size=max_extra_msa,  # extra_msa (4, 5120, 68)
        num_res=pad_len,  # aatype (4, 68)
        num_templates=4,
    )  # template_mask (4, 4) second value
    return input_fix

def relax_me(pdb_filename=None, pdb_lines=None, pdb_obj=None, use_gpu=False):
    if "relax" not in dir():
        patch_openmm()
        from alphafold.common import residue_constants
        from alphafold.relax import relax

    if pdb_obj is None:        
        if pdb_lines is None:
            pdb_lines = Path(pdb_filename).read_text()
        pdb_obj = protein.from_pdb_string(pdb_lines)
    
    amber_relaxer = relax.AmberRelaxation(
        max_iterations=0,
        tolerance=2.39,
        stiffness=10.0,
        exclude_residues=[],
        max_outer_iterations=3,
        use_gpu=use_gpu)
    
    relaxed_pdb_lines, _, _ = amber_relaxer.process(prot=pdb_obj)
    return relaxed_pdb_lines






class file_manager:
    def __init__(self, prefix: str, result_dir: Path):
        self.prefix = prefix
        self.result_dir = result_dir
        self.tag = None
        self.files = {}
    
    def get(self, x: str, ext:str) -> Path:
        if self.tag not in self.files:
            self.files[self.tag] = []
        file = self.result_dir.joinpath(f"{self.prefix}_{x}_{self.tag}.{ext}")
        self.files[self.tag].append([x,ext,file])
        return file

    def set_tag(self, tag):
        self.tag = tag

def predict_structure(
    prefix: str,
    result_dir: Path,
    feature_dict: Dict[str, Any],
    is_complex: bool,
    use_templates: bool,
    template_domains:Dict,
    sequences_lengths: List[int],
    pad_len: int,
    model_type: str,
    model_runner_and_params: List[Tuple[str, model.RunModel, haiku.Params]],
    num_relax: int = 0,
    rank_by: str = "auto",
    random_seed: int = 0,
    num_seeds: int = 1,
    stop_at_score: float = 100,
    prediction_callback: Callable[[Any, Any, Any, Any, Any], Any] = None,
    use_gpu_relax: bool = False,
    save_all: bool = False,
    save_single_representations: bool = False,
    save_pair_representations: bool = False,
    save_recycles: bool = False,
):
    """Predicts structure using AlphaFold for the given sequence."""

    mean_scores = []
    conf = []
    unrelaxed_pdb_lines = []
    prediction_times = []
    model_names = []
    files = file_manager(prefix, result_dir)
    seq_len = sum(sequences_lengths)
    fold_id = get_fold_id(prefix)

    # iterate through random seeds
    for seed_num, seed in enumerate(range(random_seed, random_seed+num_seeds)):
        
        # iterate through models
        for model_num, (model_name, model_runner, params) in enumerate(model_runner_and_params):
            
            # swap params to avoid recompiling
            model_runner.params = params
            
            #########################
            # process input features
            #########################
            if "multimer" in model_type:
                if model_num == 0 and seed_num == 0:
                    # TODO: add pad_input_mulitmer()
                    input_features = feature_dict
                    input_features["asym_id"] = input_features["asym_id"] - input_features["asym_id"][...,0]
            else:
                if model_num == 0:
                    input_features = model_runner.process_features(feature_dict, random_seed=seed)            
                    r = input_features["aatype"].shape[0]
                    input_features["asym_id"] = np.tile(feature_dict["asym_id"],r).reshape(r,-1)
                    if seq_len < pad_len:
                        input_features = pad_input(input_features, model_runner, 
                            model_name, pad_len, use_templates)
                        logger.info(f"Padding length to {pad_len}")
            

            tag = f"{model_type}_{model_name}_seed_{seed:03d}"
            model_names.append(tag)
            files.set_tag(tag)
            
            ########################
            # predict
            ########################
            start = time.time()

            recycle_stats = []
            
            # monitor intermediate results
            def callback(result, recycles):
                if recycles == 0: result.pop("tol",None)
                if not is_complex: result.pop("iptm",None)
                print_line = ""
                for x,y in [["mean_plddt","pLDDT"],["ptm","pTM"],["iptm","ipTM"],["tol","tol"]]:
                  if x in result:
                    print_line += f" {y}={result[x]:.3g}"

                iptm = 0
                if "iptm" in result: iptm = result['iptm']
                tol = 0
                if "tol" in result: tol = result['tol']

                recycle_stats.append({'recycle_index':recycles, 'mean_plddt':result['mean_plddt'], 'ptm':result['ptm'], 'iptm':iptm, 'tol':tol})
                logger.info(f"{tag} recycle={recycles}{print_line}")

                if save_recycles:
                    final_atom_mask = result["structure_module"]["final_atom_mask"]
                    b_factors = result["plddt"][:, None] * final_atom_mask
                    unrelaxed_protein = protein.from_prediction(
                        features=input_features,
                        result=result, b_factors=b_factors,
                        remove_leading_feature_dimension=("multimer" not in model_type))
                    files.get("unrelaxed",f"r{recycles}.pdb").write_text(protein.to_pdb(unrelaxed_protein))
                
                    if save_all:
                        with files.get("all",f"r{recycles}.pickle").open("wb") as handle:
                            pickle.dump(result, handle)
                    del unrelaxed_protein
            
            return_representations = save_all or save_single_representations or save_pair_representations

            # predict
            result, recycles = \
            model_runner.predict(input_features,
                random_seed=seed,
                return_representations=return_representations,
                callback=callback)

            prediction_times.append(time.time() - start)

            ########################
            # parse results
            ########################
            
            # summary metrics
            mean_scores.append(result["ranking_confidence"])         
            if recycles == 0: result.pop("tol",None)
            if not is_complex: result.pop("iptm",None)
            print_line = ""
            conf.append({})
            for x,y in [["mean_plddt","pLDDT"],["ptm","pTM"],["iptm","ipTM"]]:
              if x in result:
                print_line += f" {y}={result[x]:.3g}"
                conf[-1][x] = float(result[x])
            conf[-1]["print_line"] = print_line
            logger.info(f"{tag} took {prediction_times[-1]:.1f}s ({recycles} recycles)")

            # create protein object
            final_atom_mask = result["structure_module"]["final_atom_mask"]
            b_factors = result["plddt"][:, None] * final_atom_mask
            unrelaxed_protein = protein.from_prediction(
                features=input_features,
                result=result,
                b_factors=b_factors,
                remove_leading_feature_dimension=("multimer" not in model_type))

            # callback for visualization
            if prediction_callback is not None:
                prediction_callback(unrelaxed_protein, sequences_lengths,
                                    result, input_features, (tag, False))

            #########################
            # save results
            #########################      

            #write out distogram probability distribution for every residue pair ij               
            # dist_probabilities = np.rint(100*np.apply_along_axis(tf.nn.softmax, 2, result['distogram']['logits']))
            # dist_probabilities_list = dist_probabilities.astype(int).tolist()

            # dist_maxes = np.apply_along_axis(np.amax, 2, dist_probabilities)
            # dist_max_list = dist_maxes.astype(int).tolist()

            # with lzma.open(str(files.get("dgram","json.xz")), "wb") as write_file:
            #     dist_txt = json.dumps(dist_probabilities_list, separators=(',', ':'))
            #     write_file.write(dist_txt.encode("utf-8"))

            # with lzma.open(str(files.get("dgram_max","json.xz")), "wb") as write_file:
            #     dist_txt = json.dumps(dist_max_list, separators=(',', ':'))
            #     write_file.write(dist_txt.encode("utf-8"))

            # save raw outputs
            if save_all:
                with files.get("all","pickle").open("wb") as handle:
                    pickle.dump(result, handle)
            if save_single_representations:
                np.save(files.get("single_repr","npy"),result["representations"]["single"])
            if save_pair_representations:
                np.save(files.get("pair_repr","npy"),result["representations"]["pair"])


            json_id = str(uuid.uuid4())

            # write an easy-to-use format (pAE and pLDDT)
            score_filename = str(files.get("scores","json.xz"))
            with lzma.open(score_filename, 'wb') as handle:
                if "predicted_aligned_error" in result:
                  pae   = result["predicted_aligned_error"][:seq_len,:seq_len]
                  plot_pae(pae, score_filename + "_pae.png", sequences_lengths)
                  scores = {
                    "id":json_id,
                    "fold_id":fold_id,
                    "max_pae": pae.max().astype(float).item(),
                    "pae": pae.astype(int).tolist(),
                  }
                  for k in ["ptm","iptm"]:
                    if k in conf[-1]: scores[k] = np.around(conf[-1][k], 2).item()
                  pae_txt = json.dumps(scores, separators=(',', ':'))
                  handle.write(pae_txt.encode("utf-8"))
                  del pae
                  del scores
            
            pae_file_md5_hash = hashlib.md5(pae_txt.encode('utf-8')).hexdigest()
            
            # save pdb
            protein_lines = protein.to_pdb(unrelaxed_protein)

            config_out_file = result_dir.joinpath("config.json")
            # Read configuration data from the JSON file
            with open(config_out_file, 'r') as json_file:
                config = json.load(json_file)


            #ADD SETTINGS TO THE FINAL PDB FILE ES EDIT
            new_settings_lines_str = ''
            remark_index = 800
            new_settings_lines_str += f"REMARK {remark_index + 1}  DATA generated_by=jwalterlab_fold_portal_HMS\n"
            datetimestr = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S') + " UTC"
            new_settings_lines_str += f"REMARK {remark_index + 2}  DATA fold_id={fold_id}\n"
            new_settings_lines_str += f"REMARK {remark_index + 3}  DATA model_gen_time={datetimestr}\n"
            new_settings_lines_str += f"REMARK {remark_index + 4}  DATA model_name={model_name}\n"
            new_settings_lines_str += f"REMARK {remark_index + 5}  DATA PAE_json_file_md5={pae_file_md5_hash}\n"
            new_settings_lines_str += f"REMARK {remark_index + 6}  DATA PAE_file_id={json_id}\n"
            remark_index += 7

            if(config.get('use_templates', False)):
                for index, (chain, templates) in enumerate(template_domains.items()):

                    templates = templates[0:feature_processing.MAX_TEMPLATES]
                    new_settings_lines_str += f"REMARK {remark_index}  TEMPLATE CHAIN:{chain} " + " ".join(templates) + "\n"
                    remark_index += 1

                
            
            json_remarks = [
                f"REMARK {remark_index}  SETTING use_templates={config.get('use_templates', False)}",
                f"REMARK {remark_index + 1}  SETTING use_dropout={config.get('use_dropout', False)}",
                f"REMARK {remark_index + 2}  SETTING msa_mode=\"{config.get('msa_mode', 'null')}\"",
                f"REMARK {remark_index + 3}  SETTING model_type=\"{config.get('model_type', 'null')}\"",
                f"REMARK {remark_index + 4}  SETTING num_recycles={config.get('num_recycles', 'null')}",
                f"REMARK {remark_index + 5}  SETTING recycle_early_stop_tolerance={config.get('recycle_early_stop_tolerance', 'null')}",
                f"REMARK {remark_index + 6}  SETTING num_ensemble={config.get('num_ensemble', 'null')}",
                f"REMARK {remark_index + 7}  SETTING max_seq={config.get('max_seq', 'null')}",
                f"REMARK {remark_index + 8}  SETTING max_extra_seq={config.get('max_extra_seq', 'null')}",
                f"REMARK {remark_index + 9}  SETTING pair_mode=\"{config.get('pair_mode', 'null')}\"",
                f"REMARK {remark_index + 10}  SETTING host_url=\"{config.get('host_url', 'null')}\"",
                f"REMARK {remark_index + 11}  SETTING stop_at_score={config.get('stop_at_score', 'null')}",
                f"REMARK {remark_index + 12}  SETTING random_seed={config.get('random_seed', 'null')}",
                f"REMARK {remark_index + 13}  SETTING use_cluster_profile={config.get('use_cluster_profile', False)}",
                f"REMARK {remark_index + 14}  SETTING use_fuse={config.get('use_fuse', False)}",
                f"REMARK {remark_index + 15}  SETTING use_bfloat16={config.get('use_bfloat16', False)}",
                f"REMARK {remark_index + 16}  SETTING version=\"{config.get('version', 'null')}\""
            ]

            for remark in json_remarks:
                new_settings_lines_str += f"{remark}\n"
                remark_index += 1

            for r in recycle_stats:
                new_settings_lines_str += f"REMARK {remark_index}  RECYCLESTAT pLDDT={r['mean_plddt']:.1f} pTM={r['ptm']:.3f} ipTM={r['iptm']:.3f} recycle={r['recycle_index']} tol={r['tol']:.2f}\n"
                remark_index += 1

            protein_lines = new_settings_lines_str + protein_lines
            pdb_filename = str(files.get("unrelaxed","pdb.xz"))
            with lzma.open(pdb_filename, 'wb') as handle:
                handle.write(protein_lines.encode("utf-8"))
            
            
            del result, unrelaxed_protein

            # early stop criteria fulfilled
            if mean_scores[-1] > stop_at_score: break
        
        # early stop criteria fulfilled
        if mean_scores[-1] > stop_at_score: break

        # cleanup
        if "multimer" not in model_type: del input_features
    if "multimer" in model_type: del input_features

    ###################################################
    # rerank models based on predicted confidence
    ###################################################
    
    rank, metric = [],[]
    result_files = []
    logger.info(f"reranking models by '{rank_by}' metric")
    model_rank = np.array(mean_scores).argsort()[::-1]
    for n, key in enumerate(model_rank):
        metric.append(conf[key])
        tag = model_names[key]
        files.set_tag(tag)
        # save relaxed pdb
        if n < num_relax:
            start = time.time()
            pdb_lines = relax_me(pdb_lines=unrelaxed_pdb_lines[key], use_gpu=use_gpu_relax)
            files.get("relaxed","pdb").write_text(pdb_lines)            
            logger.info(f"Relaxation took {(time.time() - start):.1f}s")

        # rename files to include rank
        new_tag = f"rank_{(n+1):03d}_{tag}"
        rank.append(new_tag)
        logger.info(f"{new_tag}{metric[-1]['print_line']}")
        for x, ext, file in files.files[tag]:
            new_file = result_dir.joinpath(f"{prefix}_{x}_{new_tag}.{ext}")
            file.rename(new_file)
            result_files.append(new_file)
        
    return {"rank":rank,
            "metric":metric,
            "result_files":result_files}

def parse_fasta(fasta_string: str) -> Tuple[List[str], List[str]]:
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Arguments:
      fasta_string: The string contents of a FASTA file.

    Returns:
      A tuple of two lists:
      * A list of sequences.
      * A list of sequence descriptions taken from the comment lines. In the
        same order as the sequences.
    """
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append("")
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions

def get_queries(
    input_path: Union[str, Path], sort_queries_by: str = "length"
) -> Tuple[List[Tuple[str, str, Optional[List[str]]]], bool]:
    """Reads a directory of fasta files, a single fasta file or a csv file and returns a tuple
    of job name, sequence and the optional a3m lines"""

    input_path = Path(input_path)
    if not input_path.exists():
        raise OSError(f"{input_path} could not be found")

    if input_path.is_file():
        if input_path.suffix == ".csv" or input_path.suffix == ".tsv":
            sep = "\t" if input_path.suffix == ".tsv" else ","
            df = pandas.read_csv(input_path, sep=sep)
            assert "id" in df.columns and "sequence" in df.columns
            queries = [
                (seq_id, sequence.upper().split(":"), None)
                for seq_id, sequence in df[["id", "sequence"]].itertuples(index=False)
            ]
            for i in range(len(queries)):
                if len(queries[i][1]) == 1:
                    queries[i] = (queries[i][0], queries[i][1][0], None)
        elif input_path.suffix == ".a3m":
            (seqs, header) = parse_fasta(input_path.read_text())
            if len(seqs) == 0:
                raise ValueError(f"{input_path} is empty")
            query_sequence = seqs[0]
            # Use a list so we can easily extend this to multiple msas later
            a3m_lines = [input_path.read_text()]
            queries = [(input_path.stem, query_sequence, a3m_lines)]
        elif input_path.suffix == ".a3m.xz":

            msa_text = lzma.open(str(input_path),mode='rt', encoding='utf-8').read()
            (seqs, header) = parse_fasta(msa_text)
            if len(seqs) == 0:
                raise ValueError(f"{input_path} is empty")
            query_sequence = seqs[0]
            # Use a list so we can easily extend this to multiple msas later
            a3m_lines = [msa_text]
            queries = [(input_path.stem, query_sequence, a3m_lines)]
        elif input_path.suffix in [".fasta", ".faa", ".fa"]:
            (sequences, headers) = parse_fasta(input_path.read_text())
            queries = []
            for sequence, header in zip(sequences, headers):
                sequence = sequence.upper()
                if sequence.count(":") == 0:
                    # Single sequence
                    queries.append((header, sequence, None))
                else:
                    # Complex mode
                    queries.append((header, sequence.upper().split(":"), None))
        else:
            raise ValueError(f"Unknown file format {input_path.suffix}")
    else:
        assert input_path.is_dir(), "Expected either an input file or a input directory"
        queries = []
        for file in sorted(input_path.iterdir()):
            if not file.is_file():
                continue
            if file.suffix.lower() not in [".a3m", ".fasta", ".faa"]:
                logger.warning(f"non-fasta/a3m file in input directory: {file}")
                continue
            (seqs, header) = parse_fasta(file.read_text())
            if len(seqs) == 0:
                logger.error(f"{file} is empty")
                continue
            query_sequence = seqs[0]
            if len(seqs) > 1 and file.suffix in [".fasta", ".faa", ".fa"]:
                logger.warning(
                    f"More than one sequence in {file}, ignoring all but the first sequence"
                )

            if file.suffix.lower() == ".a3m":
                a3m_lines = [file.read_text()]
                queries.append((file.stem, query_sequence.upper(), a3m_lines))
            else:
                if query_sequence.count(":") == 0:
                    # Single sequence
                    queries.append((file.stem, query_sequence, None))
                else:
                    # Complex mode
                    queries.append((file.stem, query_sequence.upper().split(":"), None))

    # sort by seq. len
    if sort_queries_by == "length":
        queries.sort(key=lambda t: len("".join(t[1])))
    
    elif sort_queries_by == "random":
        random.shuffle(queries)
    
    is_complex = False
    for job_number, (raw_jobname, query_sequence, a3m_lines) in enumerate(queries):
        if isinstance(query_sequence, list):
            is_complex = True
            break
        if a3m_lines is not None and a3m_lines[0].startswith("#"):
            a3m_line = a3m_lines[0].splitlines()[0]
            tab_sep_entries = a3m_line[1:].split("\t")
            if len(tab_sep_entries) == 2:
                query_seq_len = tab_sep_entries[0].split(",")
                query_seq_len = list(map(int, query_seq_len))
                query_seqs_cardinality = tab_sep_entries[1].split(",")
                query_seqs_cardinality = list(map(int, query_seqs_cardinality))
                is_single_protein = (
                    True
                    if len(query_seq_len) == 1 and query_seqs_cardinality[0] == 1
                    else False
                )
                if not is_single_protein:
                    is_complex = True
                    break
    return queries, is_complex

def pair_sequences(
    a3m_lines: List[str], query_sequences: List[str], query_cardinality: List[int]
) -> str:
    a3m_line_paired = [""] * len(a3m_lines[0].splitlines())
    for n, seq in enumerate(query_sequences):
        lines = a3m_lines[n].splitlines()
        for i, line in enumerate(lines):
            if line.startswith(">"):
                if n != 0:
                    line = line.replace(">", "\t", 1)
                a3m_line_paired[i] = a3m_line_paired[i] + line
            else:
                a3m_line_paired[i] = a3m_line_paired[i] + line * query_cardinality[n]
    return "\n".join(a3m_line_paired)

def pad_sequences(
    a3m_lines: List[str], query_sequences: List[str], query_cardinality: List[int]
) -> str:
    _blank_seq = [
        ("-" * len(seq))
        for n, seq in enumerate(query_sequences)
        for _ in range(query_cardinality[n])
    ]
    a3m_lines_combined = []
    pos = 0
    for n, seq in enumerate(query_sequences):
        for j in range(0, query_cardinality[n]):
            lines = a3m_lines[n].split("\n")
            for a3m_line in lines:
                if len(a3m_line) == 0:
                    continue
                if a3m_line.startswith(">"):
                    a3m_lines_combined.append(a3m_line)
                else:
                    a3m_lines_combined.append(
                        "".join(_blank_seq[:pos] + [a3m_line] + _blank_seq[pos + 1 :])
                    )
            pos += 1
    return "\n".join(a3m_lines_combined)

def get_msa_and_templates(
    jobname: str,
    query_sequences: Union[str, List[str]],
    result_dir: Path,
    msa_mode: str,
    use_templates: bool,
    custom_template_path: str,
    pair_mode: str,
    host_url: str = DEFAULT_API_SERVER,
) -> Tuple[
    Optional[List[str]], Optional[List[str]], List[str], List[int], List[Dict[str, Any]]
]:
    from colabfold.colabfold import run_mmseqs2

    use_env = msa_mode == "mmseqs2_uniref_env"
    if isinstance(query_sequences, str): query_sequences = [query_sequences]

    # remove duplicates before searching
    query_seqs_unique = []
    for x in query_sequences:
        if x not in query_seqs_unique:
            query_seqs_unique.append(x)

    # determine how many times is each sequence is used
    query_seqs_cardinality = [0] * len(query_seqs_unique)
    for seq in query_sequences:
        seq_idx = query_seqs_unique.index(seq)
        query_seqs_cardinality[seq_idx] += 1

    # get template features
    template_features = []
    if use_templates:
        a3m_lines_mmseqs2, template_paths = run_mmseqs2(
            query_seqs_unique,
            str(result_dir.joinpath(jobname)),
            use_env,
            use_templates=True,
            host_url=host_url,
        )
        if custom_template_path is not None:
            template_paths = {}
            for index in range(0, len(query_seqs_unique)):
                template_paths[index] = custom_template_path
        if template_paths is None:
            logger.info("No template detected")
            for index in range(0, len(query_seqs_unique)):
                template_feature = mk_mock_template(query_seqs_unique[index])
                template_features.append(template_feature)
        else:
            for index in range(0, len(query_seqs_unique)):
                if template_paths[index] is not None:
                    template_feature = mk_template(
                        a3m_lines_mmseqs2[index],
                        template_paths[index],
                        query_seqs_unique[index],
                    )
                    if len(template_feature["template_domain_names"]) == 0:
                        template_feature = mk_mock_template(query_seqs_unique[index])
                        logger.info(f"Sequence {index} found no templates")
                    else:
                        logger.info(
                            f"Sequence {index} found templates: {template_feature['template_domain_names'].astype(str).tolist()}"
                        )
                else:
                    template_feature = mk_mock_template(query_seqs_unique[index])
                    logger.info(f"Sequence {index} found no templates")

                template_features.append(template_feature)
    else:
        for index in range(0, len(query_seqs_unique)):
            template_feature = mk_mock_template(query_seqs_unique[index])
            template_features.append(template_feature)

    if len(query_sequences) == 1:
        pair_mode = "none"

    if pair_mode == "none" or pair_mode == "unpaired" or pair_mode == "unpaired_paired":
        if msa_mode == "single_sequence":
            a3m_lines = []
            num = 101
            for i, seq in enumerate(query_seqs_unique):
                a3m_lines.append(f">{num + i}\n{seq}")
        else:
            # find normal a3ms
            a3m_lines = run_mmseqs2(
                query_seqs_unique,
                str(result_dir.joinpath(jobname)),
                use_env,
                use_pairing=False,
                host_url=host_url,
            )
    else:
        a3m_lines = None

    if msa_mode != "single_sequence" and (
        pair_mode == "paired" or pair_mode == "unpaired_paired"
    ):
        # find paired a3m if not a homooligomers
        if len(query_seqs_unique) > 1:
            paired_a3m_lines = run_mmseqs2(
                query_seqs_unique,
                str(result_dir.joinpath(jobname)),
                use_env,
                use_pairing=True,
                host_url=host_url,
            )
        else:
            # homooligomers
            num = 101
            paired_a3m_lines = []
            for i in range(0, query_seqs_cardinality[0]):
                paired_a3m_lines.append(f">{num+i}\n{query_seqs_unique[0]}\n")
    else:
        paired_a3m_lines = None

    return (
        a3m_lines,
        paired_a3m_lines,
        query_seqs_unique,
        query_seqs_cardinality,
        template_features,
    )

def build_monomer_feature(
    sequence: str, unpaired_msa: str, template_features: Dict[str, Any]
):
    msa = pipeline.parsers.parse_a3m(unpaired_msa)
    # gather features
    return {
        **pipeline.make_sequence_features(
            sequence=sequence, description="none", num_res=len(sequence)
        ),
        **pipeline.make_msa_features([msa]),
        **template_features,
    }

def build_multimer_feature(paired_msa: str) -> Dict[str, ndarray]:
    parsed_paired_msa = pipeline.parsers.parse_a3m(paired_msa)
    return {
        f"{k}_all_seq": v
        for k, v in pipeline.make_msa_features([parsed_paired_msa]).items()
    }

def process_multimer_features(
    features_for_chain: Dict[str, Dict[str, ndarray]],
    min_num_seq: int = 512,
) -> Dict[str, ndarray]:
    all_chain_features = {}
    for chain_id, chain_features in features_for_chain.items():
        all_chain_features[chain_id] = pipeline_multimer.convert_monomer_features(
            chain_features, chain_id
        )

    all_chain_features = pipeline_multimer.add_assembly_features(all_chain_features)
    # np_example = feature_processing.pair_and_merge(
    #    all_chain_features=all_chain_features, is_prokaryote=is_prokaryote)
    feature_processing.process_unmerged_features(all_chain_features)
    np_chains_list = list(all_chain_features.values())
    # noinspection PyProtectedMember
    pair_msa_sequences = not feature_processing._is_homomer_or_monomer(np_chains_list)
    chains = list(np_chains_list)
    chain_keys = chains[0].keys()
    updated_chains = []
    for chain_num, chain in enumerate(chains):
        new_chain = {k: v for k, v in chain.items() if "_all_seq" not in k}
        for feature_name in chain_keys:
            if feature_name.endswith("_all_seq"):
                feats_padded = msa_pairing.pad_features(
                    chain[feature_name], feature_name
                )
                new_chain[feature_name] = feats_padded
        new_chain["num_alignments_all_seq"] = np.asarray(
            len(np_chains_list[chain_num]["msa_all_seq"])
        )
        updated_chains.append(new_chain)
    np_chains_list = updated_chains
    np_chains_list = feature_processing.crop_chains(
        np_chains_list,
        msa_crop_size=feature_processing.MSA_CROP_SIZE,
        pair_msa_sequences=pair_msa_sequences,
        max_templates=feature_processing.MAX_TEMPLATES,
    )
    # merge_chain_features crashes if there are additional features only present in one chain
    # remove all features that are not present in all chains
    common_features = set([*np_chains_list[0]]).intersection(*np_chains_list)
    np_chains_list = [
        {key: value for (key, value) in chain.items() if key in common_features}
        for chain in np_chains_list
    ]
    np_example = feature_processing.msa_pairing.merge_chain_features(
        np_chains_list=np_chains_list,
        pair_msa_sequences=pair_msa_sequences,
        max_templates=feature_processing.MAX_TEMPLATES,
    )
    np_example = feature_processing.process_final(np_example)

    # Pad MSA to avoid zero-sized extra_msa.
    np_example = pipeline_multimer.pad_msa(np_example, min_num_seq=min_num_seq)
    return np_example

def pair_msa(
    query_seqs_unique: List[str],
    query_seqs_cardinality: List[int],
    paired_msa: Optional[List[str]],
    unpaired_msa: Optional[List[str]],
) -> str:
    if paired_msa is None and unpaired_msa is not None:
        a3m_lines = pad_sequences(
            unpaired_msa, query_seqs_unique, query_seqs_cardinality
        )
    elif paired_msa is not None and unpaired_msa is not None:
        a3m_lines = (
            pair_sequences(paired_msa, query_seqs_unique, query_seqs_cardinality)
            + "\n"
            + pad_sequences(unpaired_msa, query_seqs_unique, query_seqs_cardinality)
        )
    elif paired_msa is not None and unpaired_msa is None:
        a3m_lines = pair_sequences(
            paired_msa, query_seqs_unique, query_seqs_cardinality
        )
    else:
        raise ValueError(f"Invalid pairing")
    return a3m_lines

def generate_input_feature(
    query_seqs_unique: List[str],
    query_seqs_cardinality: List[int],
    unpaired_msa: List[str],
    paired_msa: List[str],
    template_features: List[Dict[str, Any]],
    is_complex: bool,
    model_type: str,
    max_seq: int,
) -> Tuple[Dict[str, Any], Dict[str, str]]:

    input_feature = {}
    domain_names = {}
    if is_complex and "multimer" not in model_type:

        full_sequence = ""
        Ls = []
        for sequence_index, sequence in enumerate(query_seqs_unique):
            for cardinality in range(0, query_seqs_cardinality[sequence_index]):
                full_sequence += sequence
                Ls.append(len(sequence))

        # bugfix
        a3m_lines = f">0\n{full_sequence}\n"
        a3m_lines += pair_msa(query_seqs_unique, query_seqs_cardinality, paired_msa, unpaired_msa)        

        input_feature = build_monomer_feature(full_sequence, a3m_lines, mk_mock_template(full_sequence))
        input_feature["residue_index"] = np.concatenate([np.arange(L) for L in Ls])
        input_feature["asym_id"] = np.concatenate([np.full(L,n) for n,L in enumerate(Ls)])
        if any(
            [
                template != b"none"
                for i in template_features
                for template in i["template_domain_names"]
            ]
        ):
            logger.warning(
                "alphafold2_ptm complex does not consider templates. Chose multimer model-type for template support."
            )

    else:
        features_for_chain = {}
        chain_cnt = 0
        # for each unique sequence
        for sequence_index, sequence in enumerate(query_seqs_unique):
            
            # get unpaired msa
            if unpaired_msa is None:
                input_msa = f">{101 + sequence_index}\n{sequence}"
            else:
                input_msa = unpaired_msa[sequence_index]

            feature_dict = build_monomer_feature(
                sequence, input_msa, template_features[sequence_index])

            if "multimer" in model_type:
                # get paired msa
                if paired_msa is None:
                    input_msa = f">{101 + sequence_index}\n{sequence}"
                else:
                    input_msa = paired_msa[sequence_index]
                feature_dict.update(build_multimer_feature(input_msa))

            # for each copy
            for cardinality in range(0, query_seqs_cardinality[sequence_index]):
                features_for_chain[protein.PDB_CHAIN_IDS[chain_cnt]] = feature_dict
                chain_cnt += 1

        if "multimer" in model_type:
            # combine features across all chains
            input_feature = process_multimer_features(features_for_chain, min_num_seq=max_seq + 4)
            domain_names = {
                chain: [
                    name.decode("UTF-8")
                    for name in feature["template_domain_names"]
                    if name != b"none"
                ]
                for (chain, feature) in features_for_chain.items()
            }
        else:
            input_feature = features_for_chain[protein.PDB_CHAIN_IDS[0]]
            input_feature["asym_id"] = np.zeros(input_feature["aatype"].shape[0],dtype=int)
            domain_names = {
                protein.PDB_CHAIN_IDS[0]: [
                    name.decode("UTF-8")
                    for name in input_feature["template_domain_names"]
                    if name != b"none"
                ]
            }
    return (input_feature, domain_names)

def unserialize_msa(
    a3m_lines: List[str], query_sequence: Union[List[str], str]
) -> Tuple[
    Optional[List[str]],
    Optional[List[str]],
    List[str],
    List[int],
    List[Dict[str, Any]],
]:
    a3m_lines = a3m_lines[0].replace("\x00", "").splitlines()
    if not a3m_lines[0].startswith("#") or len(a3m_lines[0][1:].split("\t")) != 2:
        assert isinstance(query_sequence, str)
        return (
            ["\n".join(a3m_lines)],
            None,
            [query_sequence],
            [1],
            [mk_mock_template(query_sequence)],
        )

    if len(a3m_lines) < 3:
        raise ValueError(f"Unknown file format a3m")
    tab_sep_entries = a3m_lines[0][1:].split("\t")
    query_seq_len = tab_sep_entries[0].split(",")
    query_seq_len = list(map(int, query_seq_len))
    query_seqs_cardinality = tab_sep_entries[1].split(",")
    query_seqs_cardinality = list(map(int, query_seqs_cardinality))
    is_homooligomer = (
        True if len(query_seq_len) == 1 and query_seqs_cardinality[0] > 1 else False
    )
    is_single_protein = (
        True if len(query_seq_len) == 1 and query_seqs_cardinality[0] == 1 else False
    )
    query_seqs_unique = []
    prev_query_start = 0
    # we store the a3m with cardinality of 1
    for n, query_len in enumerate(query_seq_len):
        query_seqs_unique.append(
            a3m_lines[2][prev_query_start : prev_query_start + query_len]
        )
        prev_query_start += query_len
    paired_msa = [""] * len(query_seq_len)
    unpaired_msa = [""] * len(query_seq_len)
    already_in = dict()
    for i in range(1, len(a3m_lines), 2):
        header = a3m_lines[i]
        seq = a3m_lines[i + 1]
        if (header, seq) in already_in:
            continue
        already_in[(header, seq)] = 1
        has_amino_acid = [False] * len(query_seq_len)
        seqs_line = []
        prev_pos = 0
        for n, query_len in enumerate(query_seq_len):
            paired_seq = ""
            curr_seq_len = 0
            for pos in range(prev_pos, len(seq)):
                if curr_seq_len == query_len:
                    prev_pos = pos
                    break
                paired_seq += seq[pos]
                if seq[pos].islower():
                    continue
                if seq[pos] != "-":
                    has_amino_acid[n] = True
                curr_seq_len += 1
            seqs_line.append(paired_seq)

        # is sequence is paired add them to output
        if (
            not is_single_protein
            and not is_homooligomer
            and sum(has_amino_acid) == len(query_seq_len)
        ):
            header_no_faster = header.replace(">", "")
            header_no_faster_split = header_no_faster.split("\t")
            for j in range(0, len(seqs_line)):
                paired_msa[j] += ">" + header_no_faster_split[j] + "\n"
                paired_msa[j] += seqs_line[j] + "\n"
        else:
            for j, seq in enumerate(seqs_line):
                if has_amino_acid[j]:
                    unpaired_msa[j] += header + "\n"
                    unpaired_msa[j] += seq + "\n"
    if is_homooligomer:
        # homooligomers
        num = 101
        paired_msa = [""] * query_seqs_cardinality[0]
        for i in range(0, query_seqs_cardinality[0]):
            paired_msa[i] = ">" + str(num + i) + "\n" + query_seqs_unique[0] + "\n"
    if is_single_protein:
        paired_msa = None
    template_features = []
    for query_seq in query_seqs_unique:
        template_feature = mk_mock_template(query_seq)
        template_features.append(template_feature)

    return (
        unpaired_msa,
        paired_msa,
        query_seqs_unique,
        query_seqs_cardinality,
        template_features,
    )

def msa_to_str(
    unpaired_msa: List[str],
    paired_msa: List[str],
    query_seqs_unique: List[str],
    query_seqs_cardinality: List[int],
) -> str:
    msa = "#" + ",".join(map(str, map(len, query_seqs_unique))) + "\t"
    msa += ",".join(map(str, query_seqs_cardinality)) + "\n"
    # build msa with cardinality of 1, it makes it easier to parse and manipulate
    query_seqs_cardinality = [1 for _ in query_seqs_cardinality]
    msa += pair_msa(query_seqs_unique, query_seqs_cardinality, paired_msa, unpaired_msa)
    return msa


global_fold_ids = {}
def get_fold_id(prefix:str):
    if prefix in global_fold_ids: return global_fold_ids[prefix]

    global_fold_ids[prefix] = str(uuid.uuid4())

    return  global_fold_ids[prefix]

def run(
    queries: List[Tuple[str, Union[str, List[str]], Optional[List[str]]]],
    result_dir: Union[str, Path],
    num_models: int,
    is_complex: bool,
    num_recycles: Optional[int] = None,
    recycle_early_stop_tolerance: Optional[float] = None,
    model_order: List[int] = [1,2,3,4,5],
    num_ensemble: int = 1,
    model_type: str = "auto",
    msa_mode: str = "mmseqs2_uniref_env",
    use_templates: bool = False,
    custom_template_path: str = None,
    num_relax: int = 0,
    keep_existing_results: bool = True,
    rank_by: str = "auto",
    pair_mode: str = "unpaired_paired",
    data_dir: Union[str, Path] = default_data_dir,
    host_url: str = DEFAULT_API_SERVER,
    random_seed: int = 0,
    num_seeds: int = 1,
    recompile_padding: Union[int, float] = 10,
    zip_results: bool = False,
    prediction_callback: Callable[[Any, Any, Any, Any, Any], Any] = None,
    save_single_representations: bool = False,
    save_pair_representations: bool = False,
    save_all: bool = False,
    save_recycles: bool = False,
    use_dropout: bool = False,
    use_gpu_relax: bool = False,
    stop_at_score: float = 100,
    dpi: int = 200,
    max_seq: Optional[int] = None,
    max_extra_seq: Optional[int] = None,
    use_cluster_profile: bool = True,
    feature_dict_callback: Callable[[Any], Any] = None,
    **kwargs
):
    # check what device is available
    try:
        # check if TPU is available
        import jax.tools.colab_tpu
        jax.tools.colab_tpu.setup_tpu()
        logger.info('Running on TPU')
        DEVICE = "tpu"
        use_gpu_relax = False
    except:
        if jax.local_devices()[0].platform == 'cpu':
            logger.info("WARNING: no GPU detected, will be using CPU")
            DEVICE = "cpu"
            use_gpu_relax = False
        else:
            import tensorflow as tf
            tf.get_logger().setLevel(logging.ERROR)
            logger.info('Running on GPU')
            DEVICE = "gpu"
            # disable GPU on tensorflow
            tf.config.set_visible_devices([], 'GPU')

    from alphafold.notebooks.notebook_utils import get_pae_json
    from colabfold.alphafold.models import load_models_and_params
    from colabfold.colabfold import plot_paes, plot_plddts
    from colabfold.plot import plot_msa_v2
    from colabfold.plot import plot_msa_v3

    data_dir = Path(data_dir)
    result_dir = Path(result_dir)
    result_dir.mkdir(exist_ok=True)
    model_type = set_model_type(is_complex, model_type)

    # determine model extension
    if   model_type == "alphafold2_multimer_v1": model_suffix = "_multimer"
    elif model_type == "alphafold2_multimer_v2": model_suffix = "_multimer_v2"
    elif model_type == "alphafold2_multimer_v3": model_suffix = "_multimer_v3"
    elif model_type == "alphafold2_ptm":         model_suffix = "_ptm"
    elif model_type == "alphafold2":             model_suffix = ""
    else: raise ValueError(f"Unknown model_type {model_type}")

    # backward-compatibility with old options
    old_names = {"MMseqs2 (UniRef+Environmental)":"mmseqs2_uniref_env",
                 "MMseqs2 (UniRef only)":"mmseqs2_uniref",
                 "unpaired+paired":"unpaired_paired"}
    msa_mode   = old_names.get(msa_mode,msa_mode)
    pair_mode  = old_names.get(pair_mode,pair_mode)
    feature_dict_callback = kwargs.pop("input_features_callback", feature_dict_callback)
    use_dropout           = kwargs.pop("training", use_dropout)
    use_fuse              = kwargs.pop("use_fuse", True)
    use_bfloat16          = kwargs.pop("use_bfloat16", True)
    max_msa               = kwargs.pop("max_msa",None)
    if max_msa is not None:
        max_seq, max_extra_seq = [int(x) for x in max_msa.split(":")]

    if kwargs.pop("use_amber", False) and num_relax == 0: 
        num_relax = num_models * num_seeds

    if len(kwargs) > 0:
        print(f"WARNING: the following options are not being used: {kwargs}")


    saved_template_paths  = kwargs.pop("saved_template_paths",None)
    saved_unpaired_msa_paths  = kwargs.pop("saved_unpaired_msa_paths",None)


    # decide how to rank outputs
    if rank_by == "auto":
        rank_by = "multimer" if is_complex else "plddt"
    if "ptm" not in model_type and "multimer" not in model_type:
        rank_by = "plddt"

    # get max length
    max_len = 0
    max_num = 0
    for _, query_sequence, _ in queries:
        N = 1 if isinstance(query_sequence,str) else len(query_sequence)
        L = len("".join(query_sequence))
        if L > max_len: max_len = L
        if N > max_num: max_num = N
    
    # get max sequences
    # 512 5120 = alphafold_ptm (models 1,3,4)
    # 512 1024 = alphafold_ptm (models 2,5)
    # 508 2048 = alphafold-multimer_v3 (models 1,2,3)
    # 508 1152 = alphafold-multimer_v3 (models 4,5)
    # 252 1152 = alphafold-multimer_v[1,2]
    
    set_if = lambda x,y: y if x is None else x
    if model_type in ["alphafold2_multimer_v1","alphafold2_multimer_v2"]:
        (max_seq, max_extra_seq) = (set_if(max_seq,252), set_if(max_extra_seq,1152))
    elif model_type == "alphafold2_multimer_v3":
        (max_seq, max_extra_seq) = (set_if(max_seq,508), set_if(max_extra_seq,2048))
    else:
        (max_seq, max_extra_seq) = (set_if(max_seq,512), set_if(max_extra_seq,5120))
    
    if msa_mode == "single_sequence":
        num_seqs = 1
        if is_complex and "multimer" not in model_type: num_seqs += max_num
        if use_templates: num_seqs += 4
        max_seq = min(num_seqs, max_seq)
        max_extra_seq = max(min(num_seqs - max_seq, max_extra_seq), 1)

    # sort model order
    model_order.sort()

    # Record the parameters of this run
    config = {
        "num_queries": len(queries),
        "use_templates": use_templates,
        "num_relax": num_relax,
        "msa_mode": msa_mode,
        "model_type": model_type,
        "num_models": num_models,
        "num_recycles": num_recycles,
        "recycle_early_stop_tolerance": recycle_early_stop_tolerance,
        "num_ensemble": num_ensemble,
        "model_order": model_order,
        "keep_existing_results": keep_existing_results,
        "rank_by": rank_by,
        "max_seq": max_seq,
        "max_extra_seq": max_extra_seq,
        "pair_mode": pair_mode,
        "host_url": host_url,
        "stop_at_score": stop_at_score,
        "random_seed": random_seed,
        "num_seeds": num_seeds,
        "recompile_padding": recompile_padding,
        "commit": get_commit(),
        "use_dropout": use_dropout,
        "use_cluster_profile": use_cluster_profile,
        "use_fuse": use_fuse,
        "use_bfloat16":use_bfloat16,
        "version": importlib_metadata.version("colabfold"),
    }
    config_out_file = result_dir.joinpath("config.json")
    config_out_file.write_text(json.dumps(config, indent=4))
    use_env = "env" in msa_mode
    use_msa = "mmseqs2" in msa_mode
    use_amber = num_relax > 0

    bibtex_file = write_bibtex(
        model_type, use_msa, use_env, use_templates, use_amber, result_dir
    )

    if custom_template_path is not None:
        mk_hhsearch_db(custom_template_path)

    pad_len = 0
    ranks, metrics = [],[]
    
    msa_data_store = {}
    def fetch_msas_in_background(queries,result_dir,keep_existing_results,use_templates,get_msa_and_templates,msa_to_str, unserialize_msa,logger,data_store, template_store, unpaired_msa_store):
        
        for job_number, (raw_jobname, query_sequence, a3m_lines) in enumerate(queries):
            jobname = safe_filename(raw_jobname)
            is_done_marker = result_dir.joinpath(jobname + ".done.txt")
            if keep_existing_results and is_done_marker.is_file():
                logger.info(f"Skipping {jobname} (already done)")
                continue
            try:
                if use_templates or a3m_lines is None:
                    data_store[jobname] = get_msa_and_templates_v3(jobname, query_sequence, result_dir, msa_mode, use_templates, custom_template_path, pair_mode, host_url, template_store, unpaired_msa_store)
                if a3m_lines is not None:
                    (unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality, template_features_) \
                    = unserialize_msa(a3m_lines, query_sequence)
                    data_store[jobname] = (unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality, template_features_)
                    if not use_templates: template_features = template_features_
                # save a3m
                (unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality, template_features) = data_store[jobname]
                msa = msa_to_str(unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality)
                msa_filename = str(result_dir.joinpath(f"{jobname}.a3m.xz"))
                last_line_len = len(msa.split("\n").pop())

                datetimestr = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S') + " UTC"
                fold_id = get_fold_id(jobname)
                msa += f"\n>NON_MSA_FILE_METADATA_LINE  fold_id={fold_id}  gen_time={datetimestr}"#+"X"*last_line_len
                with lzma.open(msa_filename, 'wb') as handle:
                    handle.write(msa.encode("utf-8"))
            except Exception as e:
                logger.exception(f"Could not generate MSA for {jobname}: {e}")
                data_store[jobname] = 'error'
                continue
    
    msa_bg_thread = Thread(target=fetch_msas_in_background, args=(queries,result_dir,keep_existing_results,use_templates,get_msa_and_templates,msa_to_str, unserialize_msa,logger, msa_data_store,saved_template_paths, saved_unpaired_msa_paths))
    msa_bg_thread.start()
    
    first_job = True
    for job_number, (raw_jobname, query_sequence, a3m_lines) in enumerate(queries):
        jobname = safe_filename(raw_jobname)
        (unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality, template_features) = None, None, None, None, None
        
        #######################################
        # check if job has already finished
        #######################################
        # In the colab version and with --zip we know we're done when a zip file has been written
        result_zip = result_dir.joinpath(jobname).with_suffix(".result.zip")
        if keep_existing_results and result_zip.is_file():
            logger.info(f"Skipping {jobname} (result.zip)")
            continue
        # In the local version we use a marker file
        is_done_marker = result_dir.joinpath(jobname + ".done.txt")
        if keep_existing_results and is_done_marker.is_file():
            logger.info(f"Skipping {jobname} (already done)")
            continue

        seq_len = len("".join(query_sequence))
        logger.info(f"Query {job_number + 1}/{len(queries)}: {jobname} (length {seq_len})")

        ###########################################
        # generate MSA (a3m_lines) and templates
        ###########################################
        try:
            if use_templates or a3m_lines is None:

                while jobname not in msa_data_store:
                    logger.info(f"WAITING ON MSA for {jobname}, sleeping for 10s")
                    time.sleep(10)

                if msa_data_store[jobname] is None or msa_data_store[jobname] == 'error':
                    logger.info(f"ERROR: Could not get MSA for {jobname} have to skip")
                    raise RuntimeError(f"MSA retrieval failed for {jobname}")
                else:
                    (unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality, template_features) = msa_data_store.pop(jobname, None)

            if a3m_lines is not None:
                (unpaired_msa, paired_msa, query_seqs_unique, query_seqs_cardinality, template_features_) \
                = unserialize_msa(a3m_lines, query_sequence)
                if not use_templates: template_features = template_features_
        except Exception as e:
            logger.exception(f"Could not get MSA/templates for {jobname}: {e}")
            continue

        #######################
        # generate features
        #######################
        try:

            (feature_dict, domain_names) \
            = generate_input_feature(query_seqs_unique, query_seqs_cardinality, unpaired_msa, paired_msa,
                                     template_features, is_complex, model_type, max_seq=max_seq)
            
            # to allow display of MSA info during colab/chimera run (thanks tomgoddard)
            if feature_dict_callback is not None:
                feature_dict_callback(feature_dict)
        
        except Exception as e:
            logger.exception(f"Could not generate input features {jobname}: {e}")
            continue
        
        ######################
        # predict structures
        ######################
        try:
            # get list of lengths
            query_sequence_len_array = sum([[len(x)] * y 
                for x,y in zip(query_seqs_unique, query_seqs_cardinality)],[])
            
            # decide how much to pad (to avoid recompiling)
            if seq_len > pad_len:
                if isinstance(recompile_padding, float):
                    pad_len = math.ceil(seq_len * recompile_padding)
                else:
                    pad_len = seq_len + recompile_padding
                pad_len = min(pad_len, max_len)
                            
            # prep model and params
            if first_job:
                # if one job input adjust max settings
                if len(queries) == 1 and msa_mode != "single_sequence":
                    # get number of sequences
                    if "msa_mask" in feature_dict:
                        num_seqs = int(sum(feature_dict["msa_mask"].max(-1) == 1))
                    else:
                        num_seqs = int(len(feature_dict["msa"]))

                    if use_templates: num_seqs += 4
                    
                    # adjust max settings
                    max_seq = min(num_seqs, max_seq)
                    max_extra_seq = max(min(num_seqs - max_seq, max_extra_seq), 1)
                    logger.info(f"Setting max_seq={max_seq}, max_extra_seq={max_extra_seq}")

                model_runner_and_params = load_models_and_params(
                    num_models=num_models,
                    use_templates=use_templates,
                    num_recycles=num_recycles,
                    num_ensemble=num_ensemble,
                    model_order=model_order,
                    model_suffix=model_suffix,
                    data_dir=data_dir,
                    stop_at_score=stop_at_score,
                    rank_by=rank_by,
                    use_dropout=use_dropout,
                    max_seq=max_seq,
                    max_extra_seq=max_extra_seq,
                    use_cluster_profile=use_cluster_profile,
                    recycle_early_stop_tolerance=recycle_early_stop_tolerance,
                    use_fuse=use_fuse,
                    use_bfloat16=use_bfloat16,
                    save_all=save_all,
                )
                first_job = False

            results = predict_structure(
                prefix=jobname,
                result_dir=result_dir,
                feature_dict=feature_dict,
                is_complex=is_complex,
                use_templates=use_templates,
                template_domains=domain_names,
                sequences_lengths=query_sequence_len_array,
                pad_len=pad_len,
                model_type=model_type,
                model_runner_and_params=model_runner_and_params,
                num_relax=num_relax,
                rank_by=rank_by,
                stop_at_score=stop_at_score,
                prediction_callback=prediction_callback,
                use_gpu_relax=use_gpu_relax,
                random_seed=random_seed,
                num_seeds=num_seeds,
                save_all=save_all,
                save_single_representations=save_single_representations,
                save_pair_representations=save_pair_representations,
                save_recycles=save_recycles,
            )
            result_files = results["result_files"]
            ranks.append(results["rank"])
            metrics.append(results["metric"])

        except RuntimeError as e:
            # This normally happens on OOM. TODO: Filter for the specific OOM error message
            logger.error(f"Could not predict {jobname}. Not Enough GPU memory? {e}")
            continue

        ###############
        # save plots
        ###############

        # make msa plot
        msa_plot = plot_msa_v3(feature_dict, dpi=300)
        coverage_png = result_dir.joinpath(f"{jobname}_coverage.png")
        msa_plot.savefig(str(coverage_png), bbox_inches='tight')
        msa_plot.close()
        result_files.append(coverage_png)

        # load the scores
#         scores = []
#         for r in results["rank"][:5]:
#             scores_file = result_dir.joinpath(f"{jobname}_scores_{r}.json")
#             with scores_file.open("r") as handle:
#                 scores.append(json.load(handle))
        
#         # write alphafold-db format (pAE)
#         if "pae" in scores[0]:
#           af_pae_file = result_dir.joinpath(f"{jobname}_predicted_aligned_error_v1.json")
#           af_pae_file.write_text(json.dumps({
#               "predicted_aligned_error":scores[0]["pae"],
#               "max_predicted_aligned_error":scores[0]["max_pae"]}))
#           result_files.append(af_pae_file)

#           # make pAE plots
#           paes_plot = plot_paes([np.asarray(x["pae"]) for x in scores],
#               Ls=query_sequence_len_array, dpi=dpi)
#           pae_png = result_dir.joinpath(f"{jobname}_pae.png")
#           paes_plot.savefig(str(pae_png), bbox_inches='tight')
#           paes_plot.close()
#           result_files.append(pae_png)

#         # make pLDDT plot
#         plddt_plot = plot_plddts([np.asarray(x["plddt"]) for x in scores],
#             Ls=query_sequence_len_array, dpi=dpi)
#         plddt_png = result_dir.joinpath(f"{jobname}_plddt.png")
#         plddt_plot.savefig(str(plddt_png), bbox_inches='tight')
#         plddt_plot.close()
#         result_files.append(plddt_png)

        if use_templates:
            templates_file = result_dir.joinpath(f"{jobname}_template_domain_names.json")
            templates_file.write_text(json.dumps(domain_names))
            result_files.append(templates_file)

        result_files.append(result_dir.joinpath(jobname + ".a3m"))
        result_files += [bibtex_file, config_out_file]

        if zip_results:
            with zipfile.ZipFile(result_zip, "w") as result_zip:
                for file in result_files:
                    result_zip.write(file, arcname=file.name)
            
            # Delete only after the zip was successful, and also not the bibtex and config because we need those again
            for file in result_files[:-2]:
                file.unlink()
        else:
            is_done_marker.touch()
            
    msa_bg_thread.join()
    logger.info("Done")
    return {"rank":ranks,"metric":metrics}

def set_model_type(is_complex: bool, model_type: str) -> str:
    # backward-compatibility with old options
    old_names = {"AlphaFold2-multimer-v1":"alphafold2_multimer_v1",
                 "AlphaFold2-multimer-v2":"alphafold2_multimer_v2",
                 "AlphaFold2-multimer-v3":"alphafold2_multimer_v3",
                 "AlphaFold2-ptm":        "alphafold2_ptm",
                 "AlphaFold2":            "alphafold2"}
    model_type = old_names.get(model_type, model_type)
    if model_type == "auto":
        if is_complex:
            model_type = "alphafold2_multimer_v3"
        else:
            model_type = "alphafold2_ptm"
    return model_type

def main():
    parser = ArgumentParser()
    parser.add_argument("input",
        default="input",
        help="Can be one of the following: "
        "Directory with fasta/a3m files, a csv/tsv file, a fasta file or an a3m file",
    )
    parser.add_argument("results", help="Directory to write the results to")
    parser.add_argument("--stop-at-score",
        help="Compute models until plddt (single chain) or ptmscore (complex) > threshold is reached. "
        "This can make colabfold much faster by only running the first model for easy queries.",
        type=float,
        default=100,
    )
    parser.add_argument("--num-recycle",
        help="Number of prediction recycles."
        "Increasing recycles can improve the quality but slows down the prediction.",
        type=int,
        default=None,
    )
    parser.add_argument("--recycle-early-stop-tolerance",
        help="Specify convergence criteria."
        "Run until the distance between recycles is within specified value.",
        type=float,
        default=None,
    )
    parser.add_argument("--num-ensemble",
        help="Number of ensembles."
        "The trunk of the network is run multiple times with different random choices for the MSA cluster centers.",
        type=int,
        default=1,
    )
    parser.add_argument("--num-seeds",
        help="Number of seeds to try. Will iterate from range(random_seed, random_seed+num_seeds)."
        ".",
        type=int,
        default=1,
    )
    parser.add_argument("--random-seed",
        help="Changing the seed for the random number generator can result in different structure predictions.",
        type=int,
        default=0,
    )
    parser.add_argument("--num-models", type=int, default=5, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--recompile-padding",
        type=int,
        default=10,
        help="Whenever the input length changes, the model needs to be recompiled."
        "We pad sequences by specified length, so we can e.g. compute sequence from length 100 to 110 without recompiling."
        "The prediction will become marginally slower for the longer input, "
        "but overall performance increases due to not recompiling. "
        "Set to 0 to disable.",
    )
    parser.add_argument("--model-order", default="1,2,3,4,5", type=str)
    parser.add_argument("--host-url", default=DEFAULT_API_SERVER)
    parser.add_argument("--data")
    parser.add_argument("--msa-mode",
        default="mmseqs2_uniref_env",
        choices=[
            "mmseqs2_uniref_env",
            "mmseqs2_uniref",
            "single_sequence",
        ],
        help="Using an a3m file as input overwrites this option",
    )
    parser.add_argument("--saved-template-features-path", default="colabfold_template_store", type=str)
    parser.add_argument("--saved-unpaired-msa-path", default="colabfold_unpaired_msa_store", type=str)
    parser.add_argument("--model-type",
        help="predict strucutre/complex using the following model."
        'Auto will pick "alphafold2_ptm" for structure predictions and "alphafold2_multimer_v3" for complexes.',
        type=str,
        default="auto",
        choices=[
            "auto",
            "alphafold2",
            "alphafold2_ptm",
            "alphafold2_multimer_v1",
            "alphafold2_multimer_v2",
            "alphafold2_multimer_v3",
        ],
    )
    parser.add_argument("--amber",
        default=False,
        action="store_true",
        help="Use amber for structure refinement."
        "To control number of top ranked structures are relaxed set --num-relax.",
    )
    parser.add_argument("--num-relax",
        help="specify how many of the top ranked structures to relax using amber.",
        type=int,
        default=0,
    )
    parser.add_argument("--templates", default=False, action="store_true", help="Use templates from pdb")
    parser.add_argument("--custom-template-path",
        type=str,
        default=None,
        help="Directory with pdb files to be used as input",
    )
    parser.add_argument("--rank",
        help="rank models by auto, plddt or ptmscore",
        type=str,
        default="auto",
        choices=["auto", "plddt", "ptm", "iptm", "multimer"],
    )
    parser.add_argument("--pair-mode",
        help="rank models by auto, unpaired, paired, unpaired_paired",
        type=str,
        default="unpaired_paired",
        choices=["unpaired", "paired", "unpaired_paired"],
    )
    parser.add_argument("--sort-queries-by",
        help="sort queries by: none, length, random",
        type=str,
        default="length",
        choices=["none", "length", "random"],
    )
    parser.add_argument("--save-single-representations",
        default=False,
        action="store_true",
        help="saves the single representation embeddings of all models",
    )
    parser.add_argument("--save-pair-representations",
        default=False,
        action="store_true",
        help="saves the pair representation embeddings of all models",
    )
    parser.add_argument("--use-dropout",
        default=False,
        action="store_true",
        help="activate dropouts during inference to sample from uncertainity of the models",
    )
    parser.add_argument("--max-seq",
        help="number of sequence clusters to use",
        type=int,
        default=None,
    )
    parser.add_argument("--max-extra-seq",
        help="number of extra sequences to use",
        type=int,
        default=None,
    )
    parser.add_argument("--max-msa",
        help="defines: `max-seq:max-extra-seq` number of sequences to use",
        type=str,
        default=None,
    )
    parser.add_argument("--disable-cluster-profile",
        default=False,
        action="store_true",
        help="EXPERIMENTAL: for multimer models, disable cluster profiles",
    )
    parser.add_argument("--zip",
        default=False,
        action="store_true",
        help="zip all results into one <jobname>.result.zip and delete the original files",
    )
    parser.add_argument("--use-gpu-relax",
        default=False,
        action="store_true",
        help="run amber on GPU instead of CPU",
    )
    parser.add_argument("--save-all",
        default=False,
        action="store_true",
        help="save ALL raw outputs from model to a pickle file",
    )
    parser.add_argument("--save-recycles",
        default=False,
        action="store_true",
        help="save all intermediate predictions at each recycle",
    )
    parser.add_argument("--overwrite-existing-results", default=False, action="store_true")
    parser.add_argument("--disable-unified-memory",
        default=False,
        action="store_true",
        help="if you are getting tensorflow/jax errors it might help to disable this",
    )

    args = parser.parse_args()
    
    # disable unified memory
    if args.disable_unified_memory:
        for k in ENV.keys():
            if k in os.environ: del os.environ[k]

    setup_logging(Path(args.results).joinpath("log.txt"))

    version = importlib_metadata.version("colabfold")
    commit = get_commit()
    if commit:
        version += f" ({commit})"

    logger.info(f"Running colabfold {version}")

    data_dir = Path(args.data or default_data_dir)

    queries, is_complex = get_queries(args.input, args.sort_queries_by)
    model_type = set_model_type(is_complex, args.model_type)
        
    download_alphafold_params(model_type, data_dir)

    if args.msa_mode != "single_sequence" and not args.templates:
        uses_api = any((query[2] is None for query in queries))
        if uses_api and args.host_url == DEFAULT_API_SERVER:
            print(ACCEPT_DEFAULT_TERMS, file=sys.stderr)

    model_order = [int(i) for i in args.model_order.split(",")]

    assert args.recompile_padding >= 0, "Can't apply negative padding"

    # backward compatibility
    if args.amber and args.num_relax == 0:
        args.num_relax = args.num_models * args.num_seeds

    run(
        queries=queries,
        result_dir=args.results,
        use_templates=args.templates,
        custom_template_path=args.custom_template_path,
        num_relax=args.num_relax,
        msa_mode=args.msa_mode,
        model_type=model_type,
        num_models=args.num_models,
        num_recycles=args.num_recycle,
        recycle_early_stop_tolerance=args.recycle_early_stop_tolerance,
        num_ensemble=args.num_ensemble,
        model_order=model_order,
        is_complex=is_complex,
        keep_existing_results=not args.overwrite_existing_results,
        rank_by=args.rank,
        pair_mode=args.pair_mode,
        data_dir=data_dir,
        host_url=args.host_url,
        random_seed=args.random_seed,
        num_seeds=args.num_seeds,
        stop_at_score=args.stop_at_score,
        recompile_padding=args.recompile_padding,
        zip_results=args.zip,
        save_single_representations=args.save_single_representations,
        save_pair_representations=args.save_pair_representations,
        use_dropout=args.use_dropout,
        max_seq=args.max_seq,
        max_extra_seq=args.max_extra_seq,
        max_msa=args.max_msa,
        use_cluster_profile=not args.disable_cluster_profile,
        use_gpu_relax = args.use_gpu_relax,
        save_all=args.save_all,
        save_recycles=args.save_recycles,
        saved_template_paths = args.saved_template_features_path,
        saved_unpaired_msa_paths = args.saved_unpaired_msa_path
    )

if __name__ == "__main__":
    main()
