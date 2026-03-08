"""
src/bakers/chem/puckering.py

[기능 정의]
분자의 3D 컨포머(Conformer)를 생성, 최적화, 그리고 필터링(Clustering)하는 핵심 로직을 담당합니다.
Ring Puckering(고리 유연성)과 Sobol Sequence(사슬 유연성)를 결합한 하이브리드 샘플링을 수행합니다.

[수정 내역]
- 거리 계산 로직(RMSD, TFD)을 bakers.analytics.metrics 모듈로 이관
- cluster_ensemble 함수가 metrics 모듈을 호출하도록 리팩토링
"""

import os,sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import TorsionFingerprints
from rdkit.ML.Cluster import Butina

# ------------------------------------------------------------------------------
# [Path Setup] 모듈 임포트 경로 설정
# ------------------------------------------------------------------------------
# 현재 스크립트: src/bakers/chem/puckering.py
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../src/bakers/chem
bakers_dir = os.path.dirname(current_dir)              # .../src/bakers
src_dir = os.path.dirname(bakers_dir)                  # .../src

# src 디렉토리를 시스템 경로에 추가하여 'bakers' 패키지를 찾을 수 있게 함
if src_dir not in sys.path:
    sys.path.append(src_dir)

# ------------------------------------------------------------------------------
# [Import] 경로 설정 후 모듈 임포트
# ------------------------------------------------------------------------------
# 거리 계산 위임
try:
    from bakers.analytics import metrics
except ImportError as e:
    print(f"[Critical Error] Failed to import 'bakers' module. Check sys.path: {sys.path}")
    raise e

# Scipy 확인 (Sobol Sequence용)
try:
    from scipy.stats import qmc
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    # 경고는 호출부에서 처리

# ==============================================================================
# 1. Ring Detection & Analysis
# ==============================================================================

def detect_rings(mol):
    """분자 내 고리 원자 인덱스 반환"""
    if not mol: return []
    return list(mol.GetRingInfo().AtomRings())

def has_flexible_rings(mol, max_ring_size=7):
    """유연한 고리(4~7원환) 존재 여부 확인"""
    rings = detect_rings(mol)
    for ring in rings:
        if 4 <= len(ring) <= max_ring_size:
            return True
    return False

# ==============================================================================
# 2. Conformer Generation (Puckering)
# ==============================================================================

def embed_with_puckering(mol, n_confs, prune_thresh=0.5, random_seed=-1):
    """(내부용) ETKDGv3를 사용한 초기 임베딩"""
    ps = AllChem.ETKDGv3()
    ps.useRandomCoords = True
    ps.pruneRmsThresh = prune_thresh
    if random_seed != -1:
        ps.randomSeed = random_seed
    
    try:
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=ps)
        return list(cids)
    except Exception:
        return []

def generate_conformers(mol, n_templates=10, n_initial=200, tfd_thresh=0.2):
    """
    [핵심 생성 함수]
    이후 단계에서 이면각(DOF)을 샘플링하므로 측쇄 회전(Sobol)은 제거했습니다.
    대신 ETKDGv3로 다수의 초기 구조를 대량 생성한 뒤, TFD 거리 매트릭스와 
    Butina 클러스터링을 적용하여 고리 등 구조적 다양성이 확보된 대표 템플릿만 추출합니다.
    """
    # 1. 초기 컨포머 풀(Pool) 대량 생성
    # 기존의 5번 반복 루프를 제거하고 한 번에 n_initial 개수를 생성합니다.
    # (반복문 사용 시 기존 RDKit 설정에 따라 이전 생성된 컨포머가 덮어씌워질 위험이 있습니다)
    cids = embed_with_puckering(mol, n_confs=n_initial, random_seed=42)
    
    # Fallback 로직 (생성 실패 시)
    if not cids:
        cids = embed_with_puckering(mol, n_confs=n_initial, random_seed=0xf00d)
        if not cids:
            return mol  # 전부 실패하면 원본(빈 껍데기) 반환
            
    cids = list(dict.fromkeys(cids))
    diverse_cids = []

    # 2. TFD & Butina 클러스터링을 통한 다양성 추출
    if len(cids) > 1:
        try:
            # TFD(Torsion Fingerprint Deviation) 거리 행렬 계산
            tfd_matrix = TorsionFingerprints.GetTFDMatrix(mol)
            
            # Butina 알고리즘으로 구조 군집화 (isDistData=True 필수)
            clusters = Butina.ClusterData(tfd_matrix, mol.GetNumConformers(), tfd_thresh, isDistData=True)
            
            # 각 클러스터의 중심(Centroid) 구조를 대표로 선택
            for cluster in clusters:
                centroid_idx = cluster[0]
                # RDKit 내부 인덱스와 CID 매핑
                centroid_cid = mol.GetConformers()[centroid_idx].GetId()
                diverse_cids.append(centroid_cid)
                
                # 목표한 템플릿 개수를 채우면 중단
                if len(diverse_cids) >= n_templates:
                    break
        except Exception:
            # 회전 결합이 아예 없는 등 TFD 계산 불가 시 순차적 선택
            diverse_cids = cids[:n_templates]
    else:
        diverse_cids = cids

    # 클러스터 수가 n_templates보다 적을 경우, 남은 자리를 다른 구조로 보충
    if len(diverse_cids) < n_templates:
        for cid in cids:
            if cid not in diverse_cids:
                diverse_cids.append(cid)
            if len(diverse_cids) >= n_templates:
                break

    # 3. 최종 보관함(final_mol)에 엄선된 대표 구조만 담기
    final_mol = Chem.Mol(mol)
    final_mol.RemoveAllConformers()
    
    for cid in diverse_cids:
        try:
            conf = mol.GetConformer(cid)
            final_mol.AddConformer(conf, assignId=True)
        except ValueError:
            continue
            
    return final_mol

# ==============================================================================
# 3. Optimization & Properties
# ==============================================================================

def optimize_ensemble(mol, conf_ids, variant='MMFF94', max_iters=200):
    """MMFF를 사용해 구조 집합을 일괄 최적화"""
    mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
    if mp is None: return

    for cid in conf_ids:
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=cid)
        if ff:
            try: ff.Minimize(maxIts=max_iters)
            except: pass

def calculate_energies(mol, conf_ids):
    """각 컨포머의 에너지 계산 및 반환"""
    props = []
    mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')
    
    for cid in conf_ids:
        e = 999.0
        if mp:
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=cid)
            if ff: e = ff.CalcEnergy()
        props.append({'cid': cid, 'energy': e})
    
    return props

# ==============================================================================
# 4. Clustering (Delegated to Metrics)
# ==============================================================================

def cluster_ensemble(mol, props, method='rmsd', threshold=2.0, max_confs=10):
    """
    [통합 클러스터링 함수]
    Energy-Ordered Pruning을 수행하며, 거리 계산은 metrics 모듈에 위임합니다.
    
    Args:
        mol: RDKit Mol
        props: List of dicts [{'cid': 1, 'energy': -50.2}, ...]
        method: 'rmsd' or 'tfd'
        threshold: 거리 임계값
        max_confs: 최종 선택할 최대 개수
    """
    # 1. 에너지 오름차순 정렬
    props.sort(key=lambda x: x['energy'])
    
    selected_cids = []
    
    for cand in props:
        if len(selected_cids) >= max_confs:
            break
            
        cid = cand['cid']
        
        # 첫 번째 구조는 무조건 선택
        if not selected_cids:
            selected_cids.append(cid)
            continue
            
        is_distinct = True
        for existing_cid in selected_cids:
            dist = 999.0
            
            # [Refactored] 거리 계산 로직을 metrics 모듈로 이관하여 호출
            if method == 'tfd':
                dist = metrics.calculate_mol_tfd(mol, existing_cid, cid, use_weights=True)
            else:
                # RMSD (Heavy Atom Only는 metrics 내부에서 처리됨)
                dist = metrics.calculate_mol_rmsd(mol, existing_cid, cid, heavy_only=True)
                
            if dist < threshold:
                is_distinct = False # 너무 유사함
                break
        
        if is_distinct:
            selected_cids.append(cid)
            
    return selected_cids


if __name__ == "__main__":
    print("="*60)
    print(" [DEBUG] Puckering Module Test Pipeline")
    print("="*60)
    
    # 1. 테스트용 분자 준비
    # 유연한 6원환 고리와 측쇄를 모두 가져 테스트에 아주 적합한 Propylcyclohexane 사용
    smiles = "C1CCCCC1CCC"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol) # 3D 구조 생성을 위해서는 수소(H) 추가가 필수적입니다.
    
    print(f"\n[1] 분자 준비 완료 (SMILES: {smiles})")
    print(f" - 전체 원자 수 (수소 포함): {mol.GetNumAtoms()}")
    
    # 2. 고리 유연성 테스트 (detect_rings, has_flexible_rings)
    rings = detect_rings(mol)
    has_flex = has_flexible_rings(mol)
    print("\n[2] 고리 탐지 (Ring Detection)")
    print(f" - 발견된 고리 수: {len(rings)}")
    print(f" - 유연한 고리(4~7원환) 존재 여부: {has_flex}")
    
    # 3. 구조 대량 생성 및 TFD 대표 구조 추출 (generate_conformers)
    print("\n[3] 초기 구조 대량 생성 및 TFD 기반 다양성 추출")
    print(" - ETKDGv3 임베딩 및 Butina 클러스터링 진행 중...")
    n_templates = 5
    n_initial = 50
    
    # 함수 실행
    mol_3d = generate_conformers(mol, n_templates=n_templates, n_initial=n_initial, tfd_thresh=0.2)
    num_generated = mol_3d.GetNumConformers()
    
    print(f" - 내부 초기 생성 시도: {n_initial}개")
    print(f" - TFD 필터링 후 추출된 대표 템플릿 수: {num_generated}개 (목표: {n_templates}개)")
    if num_generated == 0:
        print(" [!] 컨포머 생성에 실패했습니다. 테스트를 종료합니다.")
        sys.exit()

    # 4. 역장 최적화 및 에너지 계산 (optimize_ensemble, calculate_energies)
    print("\n[4] 구조 최적화(MMFF94) 및 에너지 계산")
    cids = [conf.GetId() for conf in mol_3d.GetConformers()]
    
    # 최적화 전 에너지 확인 (최적화 효과를 검증하기 위함)
    pre_props = calculate_energies(mol_3d, cids)
    avg_pre_e = sum(p['energy'] for p in pre_props) / len(pre_props)
    print(f" - 최적화 전 평균 에너지: {avg_pre_e:.2f} kcal/mol")
    
    # 최적화 수행
    optimize_ensemble(mol_3d, cids, max_iters=200)
    
    # 최적화 후 에너지 확인
    post_props = calculate_energies(mol_3d, cids)
    avg_post_e = sum(p['energy'] for p in post_props) / len(post_props)
    print(f" - 최적화 후 평균 에너지: {avg_post_e:.2f} kcal/mol")
    
    # 가장 안정한 구조 찾기
    best_conf = min(post_props, key=lambda x: x['energy'])
    print(f" - 최저 에너지 컨포머 ID: {best_conf['cid']} ({best_conf['energy']:.2f} kcal/mol)")
    
    # 5. 최종 앙상블 클러스터링 테스트 (외부 metrics 모듈 연동)
    print("\n[5] 외부 모듈(metrics) 연동 클러스터링 테스트")
    try:
        # metrics 모듈이 정상적으로 로드되었다면 이 로직이 실행됩니다.
        selected_cids = cluster_ensemble(mol_3d, post_props, method='rmsd', threshold=0.5, max_confs=3)
        print(f" - RMSD 기반 클러스터링 성공!")
        print(f" - 최종 선택된 정예 컨포머 ID 목록 (최대 3개): {selected_cids}")
    except NameError:
        print(" - [경고] metrics 모듈을 찾을 수 없어 cluster_ensemble 테스트를 건너뜁니다.")
    except AttributeError as e:
        print(f" - [경고] metrics 모듈 내에 필요한 함수가 아직 구현되지 않았습니다: {e}")
    except Exception as e:
        print(f" - [에러] 알 수 없는 오류 발생: {e}")

    print("\n"+"="*60)
    print(" [DEBUG] 테스트 완료")
    print("="*60)