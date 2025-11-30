# utils/cleanup_ops.py

import numpy as np
from scipy.ndimage import distance_transform_edt
# 생태 민감도를 위해 biology_ops에서 Recovery Index 로직을 사용합니다.
# 실제 구현 시, biology_ops의 ecological_recovery_index가 필요합니다.
# from utils.biology_ops import ecological_recovery_index 

def create_priority_map(
    oil_conc_pred: np.ndarray, 
    recovery_index: np.ndarray, 
    H: int, W: int,
    alpha: float = 0.6, 
    beta: float = 0.2, 
    gamma: float = 0.2
) -> np.ndarray:
    """
    AI 예측 결과와 생태 민감도를 고려하여 정화 우선순위 맵을 생성합니다.
    Priority = alpha * Oil + beta * (1 - Recovery) + gamma * Proximity
    """
    
    # A. 오일 농도 (Oil): 예측 모델의 직접적인 출력
    oil_term = oil_conc_pred 

    # B. 생태 민감도 (Recovery Sensitivity)
    recovery_term = 1.0 - recovery_index

    # C. 해안 근접성 (Proximity) 시뮬레이션
    shore_mask = np.zeros((H, W), dtype=int)
    shore_mask[:, int(W * 0.95):] = 1 # 오른쪽 5%를 해안으로 가정
    
    distance_map = distance_transform_edt(1 - shore_mask) 
    normalized_distance = distance_map / distance_map.max()
    
    k_proximity = 5.0 # 집중도 상수
    proximity_term = np.exp(-k_proximity * normalized_distance)
    proximity_term /= proximity_term.max() 

    # D. 최종 우선순위 계산 (가중 합산)
    priority_map = (
        alpha * oil_term + 
        beta * recovery_term + 
        gamma * proximity_term
    )
    
    # 0~1로 최종 정규화
    priority_map /= priority_map.max() 
    return priority_map


def apply_cleanup_action(
    oil_map: np.ndarray, 
    priority_map: np.ndarray, 
    budget: float, 
    efficiency: float = 0.8
) -> tuple[np.ndarray, float]:
    """
    AI가 결정한 최적의 정화 행동을 시뮬레이션하여 오일 농도를 감소시킵니다.
    (가장 높은 우선순위 구역에 제한된 예산 내에서 투입한다고 가정)
    """
    H, W = oil_map.shape
    oil_new = oil_map.copy()
    
    initial_risk = np.sum(oil_map * priority_map)
    
    # 투입 구역 결정: 우선순위가 높은 순서대로
    flat_priority = priority_map.flatten()
    sorted_indices = np.argsort(flat_priority)[::-1] 

    remaining_budget = budget
    
    for flat_idx in sorted_indices:
        if remaining_budget <= 0:
            break
            
        y, x = np.unravel_index(flat_idx, (H, W))
        
        # 투입량 결정 (우선순위 구역에 오일 농도의 50%만큼 투입 시도)
        target_dose = oil_map[y, x] * 0.5 
        dose = min(remaining_budget, target_dose)

        if dose > 0:
            # 유처리제 효과: 제거량 = 현재 오일 * 투입량 * 효율
            removed_amount = oil_new[y, x] * dose * efficiency
            
            oil_new[y, x] = max(0.0, oil_new[y, x] - removed_amount)
            
            remaining_budget -= dose
            
    final_risk = np.sum(oil_new * priority_map)
    risk_reduction = initial_risk - final_risk

    return oil_new, risk_reduction
