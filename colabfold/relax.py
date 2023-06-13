import os
from alphafold.common import protein, residue_constants
from argparse import ArgumentParser
from datetime import datetime

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


def relax_pdb(pdb_filename, use_gpu=True):

    print(f"working on {pdb_filename}")

    if "relax" not in dir():
        patch_openmm()
        from alphafold.common import residue_constants
        from alphafold.relax import relax

    if not os.path.exists(pdb_filename):
        return ''

    pdb_text = None
    with open(pdb_filename, 'r') as f:
        pdb_text = f.read()

    pdb_obj = protein.from_pdb_string(pdb_text)
    
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("relax start:", formatted_time)

    amber_relaxer = relax.AmberRelaxation(
        max_iterations=0,
        tolerance=2.39,
        stiffness=10.0,
        exclude_residues=[],
        max_outer_iterations=3,
        use_gpu=use_gpu)
    
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print("relax end:", formatted_time)
    
    relaxed_pdb_lines, _, _ = amber_relaxer.process(prot=pdb_obj)
    return relaxed_pdb_lines


def run(pdb_dir, result_dir, use_gpu_relax):

    print(f"Working on PDB files in folder {pdb_dir} and outputting to {result_dir}")
    print(f"Using {'GPU' if use_gpu_relax else 'CPU'}")

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    pdb_files = [file for file in os.listdir(pdb_dir) if file.endswith(".pdb")]
    for pdb_file in pdb_files:
         input_pdb = os.path.join(pdb_dir, pdb_file)
         pdb_lines = relax_pdb(input_pdb, use_gpu=use_gpu_relax)
         output_pdb_filename = os.path.join(result_dir, pdb_file + '_relaxed.pdb')
         with open(output_pdb_filename, 'w') as f:
            f.write(pdb_lines)

def main():
    parser = ArgumentParser()
    parser.add_argument("input",
        default="input",
        help="Directory with PDB files to relax",
    )
    parser.add_argument("results", help="Directory to write the results to")
    parser.add_argument("--use-gpu",
        default=False,
        action="store_true",
        help="run amber on GPU instead of CPU",
    )
    args = parser.parse_args()

    run(
        pdb_dir=args.input,
        result_dir=args.results,
        use_gpu_relax=args.use_gpu,
    )

if __name__ == "__main__":
    main()
