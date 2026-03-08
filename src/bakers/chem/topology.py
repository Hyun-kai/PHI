"""
src/bakers/chem/topology.py

[설명]
분자(Residue)의 위상(Topology) 구조를 분석하여 폴리머 조립 및 로타머 샘플링에 필요한 
핵심 정보(Backbone, Caps, Connection Anchors, DOFs)를 추출하는 모듈입니다.

[최종 수정 내역]
- [Absolute Role-based Ordering] 사용자가 정의한 완벽한 1:1 교차 맵핑 
  (9-43, 10-47, 11-39, 15-40, 16-41, 17-42)을 영구적으로 보장하기 위해, 
  Nuc(Lower) 앵커와 Elec(Upper) 앵커의 출력 배열 포맷을 역할에 맞게 분리했습니다.
- [Compatibility] 배열 포맷 변경에 대응하여 get_backbone_path 및 get_dofs 함수에서 
  참조하는 인덱스 기준점을 완벽하게 동기화했습니다.
"""
import os, sys
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Dict, Any, Set, Tuple, Optional

# 현재 스크립트 위치를 기준으로 프로젝트 루트(src) 경로를 시스템 패스에 추가합니다.
current_dir = os.path.dirname(os.path.abspath(__file__)) 
src_dir = os.path.dirname(os.path.dirname(current_dir))                  
if src_dir not in sys.path: sys.path.append(src_dir)

# bakers.chem.capping 모듈 임포트 (캡핑 그룹 탐지용)
try:
    from bakers.chem import capping
except ImportError:
    pass

# ==============================================================================
# 1. 헬퍼 함수 (Helper Functions)
# ==============================================================================

def get_neighbors(mol: Chem.Mol, atom_idx: int) -> List[int]:
    """특정 원자에 직접 결합된 이웃 원자들의 인덱스 리스트를 반환합니다."""
    return [a.GetIdx() for a in mol.GetAtomWithIdx(atom_idx).GetNeighbors()]

def is_terminal_methyl(mol: Chem.Mol, atom_idx: int) -> bool:
    """말단 메틸기(-CH3) 여부를 확인합니다."""
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetAtomicNum() != 6: return False
    if len([n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]) != 1: return False
    if atom.GetTotalNumHs() == 3: return True
    return False

def _pick_end_neighbor(mol: Chem.Mol, atom_idx: int, exclude_indices: Set[int]) -> Optional[int]:
    """
    이면각의 끝단 기준 원자를 선택하는 기본 함수.
    무거운 원자(N > O > C)를 우선 선택하여 화학적 방향성을 유지합니다.
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    cand = [a.GetIdx() for a in atom.GetNeighbors() if a.GetIdx() not in exclude_indices]
    if not cand: return None
    
    for target_z in [7, 8, 6]: 
        targets = [i for i in cand if mol.GetAtomWithIdx(i).GetAtomicNum() == target_z]
        if targets: return min(targets) 
    return min(cand)

def _build_anchor_sequence(mol: Chem.Mol, start_idx: int, allowed_indices: Set[int], target_length: int = 5) -> List[int]:
    """너비 우선 탐색(BFS)을 사용하여 일반 앵커 배열을 생성합니다."""
    seq, queue, visited = [start_idx], [start_idx], {start_idx}
    while queue and len(seq) < target_length:
        curr = queue.pop(0)
        nbrs = [(nbr.GetAtomicNum(), nbr.GetIdx()) for nbr in mol.GetAtomWithIdx(curr).GetNeighbors() 
                if nbr.GetIdx() in allowed_indices and nbr.GetIdx() not in visited]
        nbrs.sort(key=lambda x: (-x[0], x[1]))
        
        for _, idx in nbrs:
            if len(seq) < target_length:
                seq.append(idx); visited.add(idx); queue.append(idx)
    return seq

# ------------------------------------------------------------------------------
# [절대 위상 정렬 로직] 사용자 정의 매핑(1:1 Cross Mapping)을 완벽 보장
# ------------------------------------------------------------------------------
def _build_ordered_pyridine_anchor(mol: Chem.Mol, smarts_match: List[int], core_indices: List[int], role: str) -> List[int]:
    """
    링 원자들의 역할을 파악하여 지정된 역할(role)에 맞는 절대 순서 배열을 반환합니다.
    - role='nuc' (Lower): [C_core, C_para, C_term, C_ortho_term, N, C_ortho_core] -> [9, 10, 11, 15, 16, 17] 도출
    - role='elec' (Upper): [C_term, C_para, C_core, C_ortho_core, N, C_ortho_term] -> [43, 47, 39, 40, 41, 42] 도출
    """
    anchor_set = set(smarts_match)
    core_set = set(core_indices)
    
    n_idx, c_core, c_term = -1, -1, -1
    
    # 1. 핵심 원자(N, C_core, C_term) 식별
    for idx in smarts_match:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 7:
            n_idx = idx
        else:
            for nbr in atom.GetNeighbors():
                n_idx_nbr = nbr.GetIdx()
                if n_idx_nbr not in anchor_set and nbr.GetAtomicNum() > 1:
                    if n_idx_nbr in core_set:
                        c_core = idx
                    else:
                        c_term = idx
                        
    if n_idx == -1 or c_core == -1 or c_term == -1:
        return list(smarts_match)
        
    c_core_atom = mol.GetAtomWithIdx(c_core)
    c_term_atom = mol.GetAtomWithIdx(c_term)
    
    # 2. C_para 식별
    c_core_nbrs = {n.GetIdx() for n in c_core_atom.GetNeighbors() if n.GetIdx() in anchor_set}
    c_term_nbrs = {n.GetIdx() for n in c_term_atom.GetNeighbors() if n.GetIdx() in anchor_set}
    
    c_para_list = list(c_core_nbrs & c_term_nbrs)
    if not c_para_list: return list(smarts_match)
    c_para = c_para_list[0]
    
    # 3. N과 인접한 Ortho 탄소들 식별
    n_atom = mol.GetAtomWithIdx(n_idx)
    n_nbrs = {n.GetIdx() for n in n_atom.GetNeighbors() if n.GetIdx() in anchor_set}
    
    c_ortho_core_list = list(c_core_nbrs & n_nbrs)
    if not c_ortho_core_list: return list(smarts_match)
    c_ortho_core = c_ortho_core_list[0]
    
    c_ortho_term_list = list(c_term_nbrs & n_nbrs)
    if not c_ortho_term_list: return list(smarts_match)
    c_ortho_term = c_ortho_term_list[0]
    
    # [핵심] 역할에 따른 맞춤형 배열 생성 (사용자 검증 100% 일치)
    if role == 'nuc':
        # Lower (새 모노머)
        return [c_core, c_para, c_term, c_ortho_term, n_idx, c_ortho_core]
    else:
        # Elec (이전 모노머)
        return [c_term, c_para, c_core, c_ortho_core, n_idx, c_ortho_term]


def _extract_nuc_anchor(mol: Chem.Mol, anchor_idx: int, leave_indices: List[int], core_indices: List[int], cap_type: str = "") -> List[int]:
    if cap_type == 'ALKYNE_PYRIDINE':
        smarts = '[C:1]#[C:2]-[c:3]1[c:4][c:5](-[C:6]#[C:7]-[C;H3:8])[c:9][n:10][c:11]1'
        query = Chem.MolFromSmarts(smarts)
        for match in mol.GetSubstructMatches(query):
            map_idx = {query.GetAtomWithIdx(i).GetAtomMapNum(): idx for i, idx in enumerate(match) if query.GetAtomWithIdx(i).GetAtomMapNum() > 0}
            if map_idx.get(2) == anchor_idx:
                ring_atoms = [map_idx[3], map_idx[4], map_idx[5], map_idx[9], map_idx[10], map_idx[11]]
                return _build_ordered_pyridine_anchor(mol, ring_atoms, core_indices, role='nuc')
                
    seq = _build_anchor_sequence(mol, anchor_idx, set(leave_indices), 5)
    return seq[:3][::-1] if len(seq) < 5 else seq[::-1]

def _extract_elec_anchor(mol: Chem.Mol, anchor_idx: int, leave_indices: List[int], core_indices: List[int], cap_type: str = "") -> List[int]:
    if cap_type == 'ALKYNE_PYRIDINE':
        smarts = '[C:1]#[C:2]-[c:3]1[c:4][c:5](-[C:6]#[C:7]-[C;H3:8])[c:9][n:10][c:11]1'
        query = Chem.MolFromSmarts(smarts)
        for match in mol.GetSubstructMatches(query):
            map_idx = {query.GetAtomWithIdx(i).GetAtomMapNum(): idx for i, idx in enumerate(match) if query.GetAtomWithIdx(i).GetAtomMapNum() > 0}
            if map_idx.get(5) == anchor_idx:
                ring_atoms = [map_idx[3], map_idx[4], map_idx[5], map_idx[9], map_idx[10], map_idx[11]]
                return _build_ordered_pyridine_anchor(mol, ring_atoms, core_indices, role='elec')
                
    seq = _build_anchor_sequence(mol, anchor_idx, set(leave_indices) | set(core_indices), 5)
    return seq[:3] if len(seq) < 5 else seq

# ==============================================================================
# 2. 위상 분석 코어 로직 (Topology Analysis)
# ==============================================================================

def analyze_residue_topology(mol: Chem.Mol) -> Dict[str, Any]:
    analysis = capping.analyze_monomer(mol)
    monomer_type = analysis.get('monomer_type', 'UNKNOWN')
    nuc_caps = analysis.get('nuc_caps', [])
    elec_caps = analysis.get('elec_caps', [])
    core_indices = analysis.get('core_indices', [])

    for cap in nuc_caps + elec_caps:
        heavy_atoms = list(cap['leave_indices'])
        expanded = set(heavy_atoms)
        for idx in heavy_atoms:
            expanded.update([n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetAtomicNum() == 1])
        cap['leave_indices'] = list(expanded)

    nuc_anchor, elec_anchor = [], []
    nuc_cap_indices, elec_cap_indices = set(), set()

    if monomer_type == 'HETEROBIFUNCTIONAL' and nuc_caps and elec_caps:
        nuc_cap_indices = set(nuc_caps[0]['leave_indices'])
        elec_cap_indices = set(elec_caps[0]['leave_indices'])
        nuc_anchor = _extract_nuc_anchor(mol, nuc_caps[0]['anchor_idx'], nuc_caps[0]['leave_indices'], core_indices, nuc_caps[0].get('cap_type', ''))
        elec_anchor = _extract_elec_anchor(mol, elec_caps[0]['anchor_idx'], elec_caps[0]['leave_indices'], core_indices, elec_caps[0].get('cap_type', ''))
        
    elif monomer_type == 'DINUCLEOPHILE' and len(nuc_caps) >= 2:
        nuc_cap_indices, elec_cap_indices = set(nuc_caps[0]['leave_indices']), set(nuc_caps[1]['leave_indices'])
        nuc_anchor = _extract_nuc_anchor(mol, nuc_caps[0]['anchor_idx'], nuc_caps[0]['leave_indices'], core_indices, nuc_caps[0].get('cap_type', ''))
        elec_anchor = _extract_elec_anchor(mol, nuc_caps[1]['anchor_idx'], nuc_caps[1]['leave_indices'], core_indices, nuc_caps[1].get('cap_type', ''))
        
    elif monomer_type == 'DIELECTROPHILE' and len(elec_caps) >= 2:
        nuc_cap_indices, elec_cap_indices = set(elec_caps[0]['leave_indices']), set(elec_caps[1]['leave_indices'])
        nuc_anchor = _extract_nuc_anchor(mol, elec_caps[0]['anchor_idx'], elec_caps[0]['leave_indices'], core_indices, elec_caps[0].get('cap_type', ''))
        elec_anchor = _extract_elec_anchor(mol, elec_caps[1]['anchor_idx'], elec_caps[1]['leave_indices'], core_indices, elec_caps[1].get('cap_type', ''))

    all_indices = set(range(mol.GetNumAtoms()))
    n_term_indices = sorted(list(all_indices - elec_cap_indices)) 
    c_term_indices = sorted(list(all_indices - nuc_cap_indices))  

    return {
        'monomer_type': monomer_type,
        'residue_indices': sorted(list(all_indices - nuc_cap_indices - elec_cap_indices)),
        'n_term_indices': n_term_indices,
        'c_term_indices': c_term_indices,
        'nuc_anchor_indices': nuc_anchor,
        'elec_anchor_indices': elec_anchor,
        'lower_connect_indices': nuc_anchor,
        'upper_connect_indices': elec_anchor,
        'is_capped': bool(nuc_anchor and elec_anchor),
    }

# ==============================================================================
# 3. 백본 경로 및 자유도(DOF) 탐색
# ==============================================================================

def get_backbone_path(mol: Chem.Mol, topo_info: Dict[str, Any]) -> List[int]:
    nuc_conn = topo_info.get('nuc_anchor_indices', [])
    elec_conn = topo_info.get('elec_anchor_indices', [])
    if not nuc_conn or not elec_conn: return []
    try:
        # [호환성 패치] 새로 정의된 배열 룰에 따라 C_core 인덱스를 정확히 참조합니다.
        # Nuc(Lower)의 C_core는 0번, Elec(Upper)의 C_core는 2번에 위치합니다.
        nuc_core = nuc_conn[0]
        elec_core = elec_conn[2] if len(elec_conn) == 6 else elec_conn[-1]
        return list(Chem.rdmolops.GetShortestPath(mol, nuc_core, elec_core))
    except Exception: return []

def get_dofs(mol: Chem.Mol, exclude_indices: Set[int]) -> List[Tuple[int, int, int, int]]:
    dofs = []
    topo_info = analyze_residue_topology(mol) 
    bb_path = get_backbone_path(mol, topo_info)
    if not bb_path: return []

    core_indices = set(topo_info.get('residue_indices', []))
    bb_bonds_set = {tuple(sorted((bb_path[i], bb_path[i+1]))) for i in range(len(bb_path)-1)}
    
    # 1. 단일 결합 매칭
    single_matches = list(mol.GetSubstructMatches(Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')))
    
    # 2. 삼중 결합 축 매칭: [Aryl]-[Alkyne]#[Alkyne]-[Aryl] 
    triple_query = Chem.MolFromSmarts('[*:1]-[C:2]#[C:3]-[*:4]')
    triple_matches = list(mol.GetSubstructMatches(triple_query))

    processed_bonds = set()

    # --- A. 단일 결합 (Single Bonds) 처리 ---
    for u, v in single_matches:
        if u in exclude_indices or v in exclude_indices: continue
        if is_terminal_methyl(mol, u) or is_terminal_methyl(mol, v): continue

        atom_u, atom_v = mol.GetAtomWithIdx(u), mol.GetAtomWithIdx(v)
        is_amide = False
        if {atom_u.GetAtomicNum(), atom_v.GetAtomicNum()} == {6, 7}:
            c_atom = atom_u if atom_u.GetAtomicNum() == 6 else atom_v
            if any(nbr.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(c_atom.GetIdx(), nbr.GetIdx()).GetBondTypeAsDouble() == 2.0 for nbr in c_atom.GetNeighbors()):
                is_amide = True
        if is_amide: continue
        
        bond_tuple = tuple(sorted((u, v)))
        if bond_tuple not in bb_bonds_set or bond_tuple in processed_bonds: 
            continue
        processed_bonds.add(bond_tuple)

        idx_u, idx_v = bb_path.index(u), bb_path.index(v)
        if idx_u > idx_v: 
            u, v, idx_u, idx_v = v, u, idx_v, idx_u
        
        a = bb_path[idx_u - 1] if idx_u > 0 else _pick_end_neighbor(mol, u, {v})
        d = bb_path[idx_v + 1] if idx_v < len(bb_path) - 1 else _pick_end_neighbor(mol, v, {u})
        
        if a is not None and d is not None: 
            dofs.append((a, u, v, d))

    # --- B. 삼중 결합 (Alkyne Linker) 처리 ---
    for match in triple_matches:
        u, c1, c2, v = match[0], match[1], match[2], match[3]
        
        if u not in bb_path or v not in bb_path: 
            continue
            
        pseudo_bond = tuple(sorted((u, v)))
        if pseudo_bond in processed_bonds:
            continue
        processed_bonds.add(pseudo_bond)
        
        # 1. 방향 정렬: u를 반드시 코어(Core) 원자로, v를 캡(Cap) 원자로 정렬합니다.
        u_core = u in core_indices
        v_core = v in core_indices
        if not u_core and v_core:
            u, v, c1, c2 = v, u, c2, c1
        elif not u_core and not v_core:
            if bb_path.index(u) > bb_path.index(v):
                u, v, c1, c2 = v, u, c2, c1
                
        # [핵심 1] Core 측 Reference (a)
        a = None
        for nbr in get_neighbors(mol, u):
            if nbr in bb_path and nbr != c1:
                a = nbr
                break
        if a is None:
            a = _pick_end_neighbor(mol, u, exclude_indices={c1})
            
        # [핵심 2] Cap 측 Reference (d): 변경된 위상 배열에 맞춰 C_ortho 위치 참조
        d = None
        nuc_anchor = topo_info.get('nuc_anchor_indices', [])
        elec_anchor = topo_info.get('elec_anchor_indices', [])
        
        if nuc_anchor and v == nuc_anchor[0]:
            d = nuc_anchor[5]  # Nuc: 5번이 C_ortho_core
        elif elec_anchor and v == elec_anchor[2]:
            d = elec_anchor[3] # Elec: 3번이 C_ortho_core
        else:
            d = _pick_end_neighbor(mol, v, exclude_indices={c2})
                
        if a is not None and d is not None:
            dofs.append((a, u, v, d))
            
    return dofs

def identify_backbone_dofs(mol: Chem.Mol, dofs: List[Tuple[int, int, int, int]]) -> Dict[str, Any]:
    topo_info = analyze_residue_topology(mol)
    mapping = {'type': topo_info['monomer_type']}
    for i, dof in enumerate(dofs):
        mapping[f'bb_{i+1}'] = dof
    return mapping

def get_backbone_atoms(mol: Chem.Mol) -> Dict[str, Any]:
    dofs = get_dofs(mol, set())
    return identify_backbone_dofs(mol, dofs)

# ==============================================================================
# 4. 분석 유틸리티 (Utilities)
# ==============================================================================

def build_topological_mask(residues: List[str], residue_params_dict: Dict[str, Any]) -> np.ndarray:
    counts = [len(residue_params_dict[res]['atoms']) for res in residues]
    total_atoms = sum(counts)
    G = nx.Graph()
    G.add_nodes_from(range(total_atoms))
    offsets = [0] + list(np.cumsum(counts[:-1]))
    
    for i, res in enumerate(residues):
        for u, v in residue_params_dict[res]['bonds']: G.add_edge(u + offsets[i], v + offsets[i])
            
    for i in range(len(residues) - 1):
        u_idx = residue_params_dict[residues[i]]['upper_connect_indices'][0] + offsets[i]
        v_idx = residue_params_dict[residues[i+1]]['lower_connect_indices'][0] + offsets[i+1] 
        G.add_edge(u_idx, v_idx)

    path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    mask = np.zeros((total_atoms, total_atoms), dtype=bool)
    
    for i in range(total_atoms):
        for j in range(i + 1, total_atoms):
            if path_lengths.get(i, {}).get(j, 999) >= 4:
                mask[i, j] = mask[j, i] = True
    return mask

def load_residue_params(filepath: str) -> Dict[str, Any]:
    import importlib.util
    import os
    if not os.path.exists(filepath): 
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        spec = importlib.util.spec_from_file_location("residue_params_mod", filepath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.residue_params
    except Exception as e:
        raise RuntimeError(f"Failed to load residue params: {e}")
    
def check_clashes(numbers: np.ndarray, positions: np.ndarray, mask: np.ndarray, mode: str = 'strict') -> bool:
    natoms = len(numbers)
    if mode == 'strict': 
        hh_lim, strict_lim, loose_lim = 1.0**2, 0.8**2, 0.5**2
    else: 
        hh_lim, strict_lim, loose_lim = 0.4**2, 0.4**2, 0.4**2
    
    for i in range(natoms):
        for j in range(i + 1, natoms):
            dist_sq = np.sum((positions[i] - positions[j]) ** 2)
            if numbers[i] == 1 and numbers[j] == 1:
                if dist_sq < hh_lim: return True
            else:
                limit = strict_lim if mask[i, j] else loose_lim
                if dist_sq < limit: return True
    return False

# ==============================================================================
# [Debug & Verification] 디버그용 실행 블록
# ==============================================================================
if __name__ == "__main__":
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # 사용자가 제공한 복잡한 비표준 아미노산 SMILES (삼중결합 및 피리딘 포함)
    test_smiles = "CC(C)(C)C1=CC(C#CC2=CC(C#CC)=CN=C2)=C(NC3=C4C=CC5=C3NC6=C5C=C(C(C)(C)C)C=C6C#CC7=CN=CC(C#CC)=C7)C4=C1"
    
    # 1. 분자 객체 생성 및 수소 추가
    mol = Chem.MolFromSmiles(test_smiles)
    mol = Chem.AddHs(mol)
    
    # 2. 3D 좌표 생성 (Dihedral 각도 적용 시 오류가 없는지 확인하기 위함)
    # randomSeed를 고정하여 매번 동일한 초기 구조가 생성되도록 함
    AllChem.EmbedMolecule(mol, randomSeed=42)
    
    # 3. 배제할 인덱스 (테스트용이므로 빈 Set 사용)
    exclude_indices = set()

    print("=" * 60)
    print("[1] Topology Analysis (토폴로지 분석)")
    print("=" * 60)
    try:
        # 기존에 정의된 topology 분석 함수 호출
        topo_info = analyze_residue_topology(mol)
        print(f"Monomer Type       : {topo_info.get('monomer_type', 'N/A')}")
        print(f"Nuc Anchor (Entry) : {topo_info.get('nuc_anchor_indices', 'N/A')}")
        print(f"Elec Anchor (Exit) : {topo_info.get('elec_anchor_indices', 'N/A')}")
    except Exception as e:
        print(f"❌ Topology 분석 중 오류 발생: {e}")

    print("\n" + "=" * 60)
    print("[2] Degrees of Freedom (자유도) 추출 검증")
    print("=" * 60)
    try:
        # 새롭게 리팩토링된 get_dofs 함수 호출
        dofs = get_dofs(mol, exclude_indices)
        print(f"✅ 총 {len(dofs)}개의 회전 가능한 결합(자유도)이 추출되었습니다.\n")
        
        # 추출된 각 DOF의 세부 정보 출력
        for i, (a, u, v, d) in enumerate(dofs):
            # 원자 기호 가져오기
            atom_a = mol.GetAtomWithIdx(a).GetSymbol()
            atom_u = mol.GetAtomWithIdx(u).GetSymbol()
            atom_v = mol.GetAtomWithIdx(v).GetSymbol()
            atom_d = mol.GetAtomWithIdx(d).GetSymbol()
            
            # u와 v 사이의 결합 종류 파악
            bond = mol.GetBondBetweenAtoms(u, v)
            bond_type = bond.GetBondTypeAsDouble()
            
            # 결합 형태에 따른 라벨링
            if bond_type == 3.0:
                bond_str = "삼중결합 (Alkyne Linker)"
                link_visual = "≡"
            else:
                bond_str = "단일결합 (Single Bond)"
                link_visual = "-"
                
            print(f"🔹 DOF #{i+1} [{bond_str}]")
            print(f"  - 원자 인덱스 : (a:{a:<3} | u:{u:<3} | v:{v:<3} | d:{d:<3})")
            print(f"  - 원자 기호   : ({atom_a}) - ({atom_u}) {link_visual} ({atom_v}) - ({atom_d})")
            print("-" * 50)
            
    except Exception as e:
        print(f"❌ DOF 추출 중 오류 발생: {e}")