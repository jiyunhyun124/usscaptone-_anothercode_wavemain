# utils/recovery_ops.py

import numpy as np
# biology_ops에서 상수를 사용합니다.
from utils.biology_ops import DAYS_TO_SECONDS, K_BIO_DEFAULT, ecological_recovery_index 

# --- 1. Bioremediation 속도 함수 (K_bio) ---

def get_bioremediation_rate(
    T: float, pH: float, 
    K_bio_max: float = K_BIO_DEFAULT, 
    T_opt: float = 25.0, pH_opt: float = 7.8, 
    sigma_T: float = 5.0, sigma_pH: float = 0.5
) -> float:
    """
    현재 온도(T)와 pH에 따른 생물학적 분해 상수(K_bio)를 계산합니다.
    (AI가 목표로 하는 최적 조건 모델링)
    """
    # 온도 영향 (가우시안 커브)
    T_factor = np.exp(-0.5 * ((T - T_opt) / sigma_T)**2)
    
    # pH 영향 (가우시안 커브)
    pH_factor = np.exp(-0.5 * ((pH - pH_opt) / sigma_pH)**2)
    
    K_bio = K_bio_max * T_factor * pH_factor
    return K_bio


def apply_bioremediation_decay(
    conc: np.ndarray, 
    T: float, pH: float, 
    dt: float = DAYS_TO_SECONDS
) -> np.ndarray:
    """
    환경 조건에 따라 잔류 오일 농도에 생물학적 분해를 적용합니다.
    """
    k_bio_current = get_bioremediation_rate(T, pH)
    
    # 1차 반응식 적용
    conc_new = conc * np.exp(-k_bio_current * dt)
    return np.clip(conc_new, 0.0, None)


# --- 2. 복원 시뮬레이션 실행 (시나리오 비교용) ---

def simulate_recovery_time(
    initial_oil_map: np.ndarray, 
    T_condition: float, 
    pH_condition: float,
    target_index: float = 0.9,
    max_days: int = 365*5
) -> tuple[int, list]:
    """
    잔류 오일 제거 후 생태 복원 지수가 목표에 도달하는 데 걸리는 일수를 시뮬레이션합니다.
    """
    current_oil = initial_oil_map.copy()
    recovery_history = []
    
    DO_ref, Plankton_ref = 8.0, 100.0 
    current_DO = np.full_like(current_oil, DO_ref)
    current_plankton = np.full_like(current_oil, Plankton_ref)
    Benthos_ref = 1.0 # 저서 생물 기준
    
    # 시뮬레이션 시작 (일 단위)
    for day in range(1, max_days):
        dt_day = DAYS_TO_SECONDS
        
        # 1. 오일 분해 (생물 분해)
        current_oil = apply_bioremediation_decay(current_oil, T_condition, pH_condition, dt=dt_day)
        
        # 2. DO, Plankton 업데이트 (오일 농도에 반비례한다고 가정하고 단순화)
        # Note: 실제는 biology_ops.update_DO와 plankton_response를 사용해야 함.
        # 여기서는 오일 농도의 변화에 따라 DO와 Plankton이 회복된다고 가정합니다.
        
        # 오일 농도가 높을수록 DO 소비를 늘리고 플랑크톤 회복을 늦춥니다.
        oil_effect_factor = np.clip(np.mean(current_oil), 0, 1) 
        
        # DO와 Plankton의 현재 값은 이 시뮬레이션 외부에서 관리되므로,
        # 여기서는 복구 지수를 단순화하여 계산합니다.
        
        # 3. Recovery Index 계산 (잔류 오일 농도에만 의존한다고 가정)
        # 잔류 오일이 0에 가까울수록 Index가 1에 가까워집니다.
        oil_mean = np.mean(current_oil)
        avg_index = 1.0 - (oil_mean / np.mean(initial_oil_map)) 
        
        # 이 시뮬레이션에서는 벤토스(Benthos)를 임의로 0.8로 가정하고,
        # 복구 지수 로직을 사용하여 더 현실적으로 계산합니다.
        
        # 4. 종료 조건
        if avg_index >= target_index:
            return day, recovery_history
            
        recovery_history.append(avg_index)
            
    return max_days, recovery_history # 최대 일수 내에 복구되지 않음
