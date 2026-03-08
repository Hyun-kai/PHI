"""
src/bakers/utils/io.py

[모듈 설명]
데이터 입출력(I/O) 및 파일 처리를 담당하는 유틸리티 모듈입니다.
HDF5 파일의 로드/저장/병합, PDB 파일 생성, 파일명 파싱, 토폴로지(원자 정보) 복원 기능을 제공합니다.

[최종 수정 내역]
- [Import Scope Fix] 타입 힌트(Type Hint)에서 'Chem.Mol'을 참조할 때 발생하던 
  NameError를 해결하기 위해, rdkit의 Chem 및 rdDetermineBonds 모듈을 
  파일 최상단(Global)에서 명시적으로 Import 하도록 수정했습니다.
- [Chemical Sanitization Filter] 추출된 3D 좌표가 화학적으로 합당한 구조(멀쩡한 분자)인지 
  RDKit의 산화수(Valence) 및 결합 규칙 검사(SanitizeMol, MolFromMolBlock)를 통해 
  엄격하게 검증하는 필터(_is_chemically_valid) 로직이 유지되었습니다.
"""

import os
import h5py
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union

# 외부 라이브러리 (필수 의존성)
try:
    from ase import Atoms
    from ase.io import write
    from scipy.spatial.distance import pdist, squareform
except ImportError:
    print("[Warning] 'ase' or 'scipy' not found. PDB saving might fail.")

# [핵심 수정] RDKit 모듈을 최상단에서 Global Import 하여 타입 힌트 평가 시 발생하는 에러를 원천 차단합니다.
try:
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
except ImportError:
    print("[Warning] 'rdkit' not found. SDF saving and chemical validation will fail.")

# ==============================================================================
# 1. HDF5 I/O Operations
# ==============================================================================

def load_hdf5_data(filepath: str, sorted_by_energy: bool = True) -> Optional[Dict[str, np.ndarray]]:
    if not os.path.exists(filepath):
        print(f"[IO Error] File not found: {filepath}")
        return None

    try:
        with h5py.File(filepath, 'r') as f:
            xyzs = None
            if 'xyzs' in f: xyzs = np.array(f['xyzs'])
            elif 'positions' in f: xyzs = np.array(f['positions'])
            
            energies = None
            if 'values' in f: energies = np.array(f['values'])
            elif 'energies' in f: energies = np.array(f['energies'])
            elif 'energy' in f: energies = np.array(f['energy'])

            points = None
            if 'points' in f: points = np.array(f['points'])

            numbers = None
            if 'numbers' in f: numbers = np.array(f['numbers'])

            if energies is None:
                return None

            if sorted_by_energy:
                idx = np.argsort(energies)
                energies = energies[idx]
                if xyzs is not None: xyzs = xyzs[idx]
                if points is not None: points = points[idx]

            return {'xyzs': xyzs, 'energies': energies, 'points': points, 'numbers': numbers}

    except Exception as e:
        print(f"[IO Error] Failed to read {filepath}: {e}")
        return None


def save_results_hdf5(filepath: str, points: np.ndarray, values: np.ndarray, xyzs: np.ndarray, numbers: np.ndarray = None) -> bool:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('points', data=points, dtype='float32')
            f.create_dataset('values', data=values, dtype='float32')
            f.create_dataset('energies', data=values, dtype='float32') 
            
            if not isinstance(xyzs, np.ndarray):
                xyzs = np.array(xyzs)
            f.create_dataset('xyzs', data=xyzs, dtype='float32')
            
            if numbers is not None:
                if not isinstance(numbers, np.ndarray):
                    numbers = np.array(numbers)
                f.create_dataset('numbers', data=numbers, dtype='int32')
            
        print(f"    [Save] HDF5 Saved: {filepath}")
        return True
    except Exception as e:
        print(f"    [Save Error] {e}")
        return False


def merge_hdf5_files(file_list: List[str], output_path: str, verbose: bool = True) -> bool:
    if not file_list:
        if verbose: print(" [Merge Error] No files to merge.")
        return False

    all_points, all_values, all_xyzs = [], [], []
    final_numbers = None 
    processed_count = 0

    if verbose: print(f" [Merge] Start merging {len(file_list)} files...")

    for fpath in file_list:
        if not os.path.exists(fpath):
            if verbose: print(f" [Skip] File not found: {fpath}")
            continue
            
        try:
            with h5py.File(fpath, 'r') as f:
                has_energies = 'values' in f or 'energies' in f
                has_points = 'points' in f
                has_xyzs = 'xyzs' in f
                
                if has_points and has_energies and has_xyzs:
                    all_points.append(f['points'][:])
                    
                    if 'values' in f: val = f['values'][:]
                    else: val = f['energies'][:]
                    all_values.append(val)
                    
                    all_xyzs.append(f['xyzs'][:])
                    
                    if 'numbers' in f and final_numbers is None:
                        final_numbers = f['numbers'][:]
                        
                    processed_count += 1
                else:
                    if verbose: print(f" [Skip] Missing keys in {os.path.basename(fpath)}")
        except Exception as e:
            if verbose: print(f" [Error] Reading {os.path.basename(fpath)}: {e}")

    if processed_count == 0:
        if verbose: print(" [Merge Error] No valid data found.")
        return False

    try:
        final_points = np.concatenate(all_points, axis=0)
        final_values = np.concatenate(all_values, axis=0)
        final_xyzs = np.concatenate(all_xyzs, axis=0)
        
        save_results_hdf5(output_path, final_points, final_values, final_xyzs, numbers=final_numbers)
        
        if verbose: 
            print(f" [Merge] Combined {processed_count} files. Shape: {final_xyzs.shape}")
        return True
        
    except ValueError as e:
        if verbose: print(f" [Merge Error] Dimension mismatch: {e}")
        return False
    except Exception as e:
        if verbose: print(f" [Merge Error] Unexpected error: {e}")
        return False


# ==============================================================================
# 2. Atomic Number Mapping & Filename Parsing
# ==============================================================================

def get_atomic_numbers(residues: List[str], rotamers: List[int], residue_params_dict: Dict) -> np.ndarray:
    selections = []
    _numbers_list = []

    for res in residues:
        if res not in residue_params_dict:
            raise KeyError(f"Residue '{res}' not found in residue_params.")
        _numbers_list.append(np.array(residue_params_dict[res]['atoms']))

    for i, res in enumerate(residues):
        p = residue_params_dict[res]
        
        if len(residues) == 1:
            sel = list(range(len(p['atoms'])))
        elif i == 0:
            sel = p.get('n_term_indices', p['residue_indices'])
        elif i == len(residues) - 1:
            sel = p.get('c_term_indices', p['residue_indices'])
        else:
            sel = p['residue_indices']
            
        selections.append(sel)

    numbers = np.concatenate([nums[sel] for nums, sel in zip(_numbers_list, selections)])
    
    mask_heavy = numbers != 1
    mask_h = numbers == 1
    return np.concatenate([numbers[mask_heavy], numbers[mask_h]])


def parse_filename_info(filename: str) -> Tuple[List[str], List[int], int]:
    base = os.path.basename(filename).replace('.hdf5', '')
    
    target_length = 0
    clean_base = base.lower()
    
    if "polymer" in clean_base: target_length = 0 
    elif "octamer" in clean_base: target_length = 8
    elif "hexamer" in clean_base: target_length = 6
    elif "tetramer" in clean_base: target_length = 4
    elif "trimer" in clean_base: target_length = 3
    elif "dimer" in clean_base: target_length = 2
    
    for suffix in ['_polymer', '_octamer', '_hexamer', '_tetramer', '_trimer', '_dimer']:
        if base.lower().endswith(suffix):
            base = base[:-len(suffix)] 
            break

    explicit_repeats = 0
    if "_x" in base:
        try:
            parts = base.split("_x")
            base = parts[0] 
            num_str = "".join([c for c in parts[1] if c.isdigit()])
            if num_str: explicit_repeats = int(num_str)
        except: pass

    parts = base.split('-')
    residues = []
    rotamers = []
    
    try:
        for part in parts:
            if '_' in part:
                res, rot = part.rsplit('_', 1)
                residues.append(res)
                rotamers.append(int(rot))
    except ValueError:
        pass
        
    num_residues = len(residues)
    repeats = 1
    
    if target_length > 0 and num_residues > 0:
        repeats = max(1, target_length // num_residues)
    elif explicit_repeats > 0:
        repeats = explicit_repeats
        
    return residues, rotamers, repeats


# ==============================================================================
# 3. Validation, PDB/SDF Saving & RDKit Official Geometry Inference
# ==============================================================================

def _create_rdkit_mol_from_coords(numbers: np.ndarray, positions: np.ndarray, info: dict = None) -> Optional[Chem.Mol]:
    """
    RDKit의 공식 모듈인 rdDetermineBonds를 사용하여 3D 좌표로부터 
    결합(단일/이중/삼중)을 완벽하게 역추론하여 반환합니다.
    방향족 고리의 파괴 없이 정확한 결합선(Valence)이 산출됩니다.
    """
    mol = Chem.RWMol()
    conf = Chem.Conformer(len(numbers))
    
    # 1. 원자 추가 및 3D 좌표 세팅
    for i, (z, pos) in enumerate(zip(numbers, positions)):
        atom = Chem.Atom(int(z))
        mol.AddAtom(atom)
        conf.SetAtomPosition(i, pos.tolist())
        
    mol.AddConformer(conf)
    
    try:
        # 2. RDKit 공식 결합 추론 알고리즘 적용 (반응성/방향족성 완벽 인지)
        # DetermineConnectivity: 거리 기반으로 연결선(Topology)만 생성
        rdDetermineBonds.DetermineConnectivity(mol, useHueckel=False)
        # DetermineBondOrders: 이중/삼중 결합 및 전하/산화수를 공식 규칙에 맞게 재배치
        rdDetermineBonds.DetermineBondOrders(mol, charge=0)
        
        if info:
            for k, v in info.items():
                mol.SetProp(str(k), str(v))
                
        return mol.GetMol()
    except Exception as e:
        # 분자가 찌그러져 겹쳐있거나 극단적으로 비틀려 있으면 여기서 실패합니다 (훌륭한 필터 역할).
        return None


def _is_chemically_valid(numbers: np.ndarray, positions: np.ndarray) -> bool:
    """
    추출된 좌표가 화학적으로 합당한(멀쩡한) 분자인지 검증합니다.
    """
    try:
        # 1. 공식 결합 추론 및 Mol 객체 생성 시도
        mol = _create_rdkit_mol_from_coords(numbers, positions)
        if mol is None:
            return False
            
        # 2. 화학적 규칙(산화수, 고리구조, 원자가 등) 엄격 검사
        Chem.SanitizeMol(mol)
        
        # 3. 블록 변환(Round-trip)을 통한 심층 검증
        mol_block = Chem.MolToMolBlock(mol)
        verified_mol = Chem.MolFromMolBlock(mol_block, sanitize=True)
        
        if verified_mol is None:
            return False
            
        return True
    except Exception as e:
        # 산화수 초과 등 찌그러진 폭발 구조는 무조건 여기서 걸러집니다.
        return False


def save_pdb(filepath: str, numbers: np.ndarray, positions: np.ndarray, info: dict = None) -> bool:
    try:
        atoms = Atoms(numbers=numbers, positions=positions)
        if info:
            for k, v in info.items():
                atoms.info[k] = v
        write(filepath, atoms)
        return True
    except Exception as e:
        print(f"[IO Error] Failed to save PDB: {e}")
        return False


def save_sdf(filepath: str, numbers: np.ndarray, positions: np.ndarray, info: dict = None) -> bool:
    """RDKit 공식 알고리즘으로 추론된 완벽한 결합 정보를 SDF로 저장합니다."""
    try:
        mol = _create_rdkit_mol_from_coords(numbers, positions, info)
        
        if mol is None:
            return False
            
        with Chem.SDWriter(filepath) as writer:
            writer.write(mol)
        return True
    except Exception as e:
        print(f"    [SDF Error] Failed to save SDF: {e}")
        return False


def guess_elements_from_geometry(coords: np.ndarray) -> List[str]:
    n_atoms = len(coords)
    dists = squareform(pdist(coords))
    BOND_THRESHOLD = 1.7 
    
    adj = dists < BOND_THRESHOLD
    np.fill_diagonal(adj, False)
    
    elements = ['C'] * n_atoms
    carbonyl_carbons = []
    
    for i in range(n_atoms):
        neighbors = np.where(adj[i])[0]
        if len(neighbors) == 1:
            n_idx = neighbors[0]
            if dists[i, n_idx] < 1.2: 
                elements[i] = 'H'
            else: 
                elements[i] = 'O'
                carbonyl_carbons.append(n_idx) 
                
    for i in range(n_atoms):
        if elements[i] != 'C': continue
        deg = len(np.where(adj[i])[0])
        if deg == 3 and i not in carbonyl_carbons:
            neighbors = np.where(adj[i])[0]
            if any(n in carbonyl_carbons for n in neighbors):
                elements[i] = 'N'
                    
    return elements


# ==============================================================================
# 4. High-Level Extraction Workflow
# ==============================================================================

def extract_and_save_top_structures(target_file: str, output_dir: str, 
                                    top_n: int = 100, cluster_threshold: float = 45.0, 
                                    project_root: str = None, save_format: str = "both"):
    """
    [화학적 검증 필터 적용]
    HDF5 결과 파일에서 구조를 선별하되, 추출된 구조가 화학적으로 올바른지(Sanitize)
    먼저 검증한 뒤 합당한(Valid) 구조만 목표 개수(top_n)만큼 PDB/SDF로 저장합니다.
    """
    
    pdb_dir = os.path.join(output_dir, "pdb")
    sdf_dir = os.path.join(output_dir, "sdf")
    
    if save_format in ['pdb', 'both']: os.makedirs(pdb_dir, exist_ok=True)
    if save_format in ['sdf', 'both']: os.makedirs(sdf_dir, exist_ok=True)

    existing_pdbs = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')] if os.path.exists(pdb_dir) else []
    existing_sdfs = [f for f in os.listdir(sdf_dir) if f.endswith('.sdf')] if os.path.exists(sdf_dir) else []
    
    if (save_format in ['pdb', 'both'] and len(existing_pdbs) >= top_n) or \
       (save_format in ['sdf', 'both'] and len(existing_sdfs) >= top_n):
        print(f"    [Skip] Target files already exist in {output_dir}. Skipping extraction.")
        return

    data = load_hdf5_data(target_file, sorted_by_energy=True)
    if data is None or data['xyzs'] is None:
        print("    [Error] No coordinate data found for extraction.")
        return

    xyzs = data['xyzs']
    energies = data['energies']
    points = data['points']
    total_structures = len(xyzs)
    
    method_str = "Energy-sorted Top-N with Validation"

    atomic_numbers = None
    if data.get('numbers') is not None:
        atomic_numbers = data['numbers']
    else:
        residues, rotamers, repeats = parse_filename_info(target_file)
        params_path = None
        if project_root:
            check_path = os.path.join(project_root, '0_inputs', 'residue_params.py')
            if os.path.exists(check_path): params_path = check_path
        
        if params_path is None and os.path.exists('0_inputs/residue_params.py'):
            params_path = '0_inputs/residue_params.py'
            
        if params_path:
            try:
                from bakers.chem import topology
                params = topology.load_residue_params(params_path)
                full_res = residues * repeats
                full_rot = rotamers * repeats
                atomic_numbers = get_atomic_numbers(full_res, full_rot, params)
            except Exception as e:
                print(f"    [Warn] Topology reconstruction failed: {e}")
        else:
            print("    [Warn] residue_params.py not found. Using geometry-based element guessing.")

    base_name = os.path.basename(target_file).replace('.hdf5', '')
    print(f"    [Extraction] Method: {method_str} | Format: {save_format.upper()}")
    
    count_pdb, count_sdf = 0, 0
    saved_count = 0
    current_rank = 1
    
    # 찌그러진 구조를 건너뛰며 유효한 구조만 top_n 개가 될 때까지 스캔합니다.
    for idx in range(total_structures):
        if saved_count >= top_n:
            break
            
        coords = xyzs[idx]
        energy = energies[idx]
        current_numbers = atomic_numbers
        
        if current_numbers is None:
            # Fallback 추론
            current_numbers = np.array([Chem.GetPeriodicTable().GetAtomicNumber(e) for e in guess_elements_from_geometry(coords)])

        # [핵심 로직] 화학적 타당성(Sanitization) 엄격 검증
        if not _is_chemically_valid(current_numbers, coords):
            print(f"    [Sanitize Fail] Index {idx} has severe structural distortion. Skipping...")
            continue
            
        info = {'Energy': float(energy), 'Original_Index': int(idx)}

        if save_format in ['pdb', 'both']:
            save_path_pdb = os.path.join(pdb_dir, f"{base_name}_rank{current_rank}_idx{idx}.pdb")
            if save_pdb(save_path_pdb, current_numbers, coords, info=info):
                count_pdb += 1
                
        if save_format in ['sdf', 'both']:
            save_path_sdf = os.path.join(sdf_dir, f"{base_name}_rank{current_rank}_idx{idx}.sdf")
            if save_sdf(save_path_sdf, current_numbers, coords, info=info):
                count_sdf += 1
                
        saved_count += 1
        current_rank += 1
                
    if save_format in ['pdb', 'both']: print(f"    [Saved] {count_pdb} Valid PDB files saved to {pdb_dir}/")
    if save_format in ['sdf', 'both']: print(f"    [Saved] {count_sdf} Valid SDF files saved to {sdf_dir}/")
    if saved_count < top_n:
        print(f"    [Warn] Only found {saved_count} valid structures out of {total_structures} total generated.")


# ==============================================================================
# 5. Debug용 *.xyz 파일 생성
# ==============================================================================
def write_xyz(types, coords, msg="", fn=None, is_onehot=False):
    xyz = ""
    xyz += f"{coords.shape[0]}\n"
    xyz += msg + "\n"
    for i in range(coords.shape[0]):
        atom_type = types[i]
        xyz += f"{atom_type}\t{coords[i][0]}\t{coords[i][1]}\t{coords[i][2]}\n"
    if fn is not None:
        with open(fn, "w") as w:
            w.writelines(xyz[:-1])
    return xyz[:-1]