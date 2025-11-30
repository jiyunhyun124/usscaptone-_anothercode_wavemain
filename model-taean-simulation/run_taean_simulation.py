import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 현재 폴더 경로 추가 (utils 불러오기 위함)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.cleanup_ops import create_priority_map, apply_cleanup_action
    from utils.recovery_ops import simulate_recovery_time
except ImportError as e:
    print(f"[Error] utils 파일을 찾을 수 없습니다: {e}")
    sys.exit(1)

def main():
    print("========================================================")
    print("   TAEAN OIL SPILL: AI vs. ACTUAL SCENARIO COMPARISON")
    print("========================================================")

    # 1. 태안 환경 가정 (가상의 오일 맵 생성)
    H, W = 64, 64
    y, x = np.mgrid[0:H, 0:W]
    center_y, center_x = 32, 20 
    # 가우시안 분포로 오일 맵 생성
    pred_oil_map = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * 10**2))
    
    # 초기 복구 지수 (나쁨)
    current_recovery_index = np.full((H, W), 0.2) 
    
    # 예산 설정 (전체 오일의 40% 처리 가능량)
    BUDGET = np.sum(pred_oil_map) * 0.4 

    print(f"[Init] Map Size: {H}x{W}")
    print(f"[Init] Initial Total Oil: {np.sum(pred_oil_map):.2f}")
    print("-" * 50)

    # ---------------------------------------------------------
    # 시나리오 A: 실제 대응 (2007년)
    # 특징: 우선순위 없음(1로 통일), 효율 낮음(0.4), 겨울철 온도(15도)
    # ---------------------------------------------------------
    print("Running Scenario A: Actual Response (2007)...")
    priority_map_actual = np.ones((H, W)) 
    
    oil_residue_actual, _ = apply_cleanup_action(
        pred_oil_map, priority_map_actual, budget=BUDGET, efficiency=0.4
    )
    
    days_actual, history_actual = simulate_recovery_time(
        oil_residue_actual, T_condition=15.0, pH_condition=8.0, target_index=0.9
    )
    print(f" -> Recovery Time: {days_actual} days ({days_actual/365:.1f} years)")

    # ---------------------------------------------------------
    # 시나리오 B: AI 최적 대응
    # 특징: 우선순위 맵 사용, 효율 높음(0.8), 최적 온도 제어(25도)
    # ---------------------------------------------------------
    print("\nRunning Scenario B: AI Optimized Response...")
    
    priority_map_ai = create_priority_map(pred_oil_map, current_recovery_index, H, W)
    
    oil_residue_ai, _ = apply_cleanup_action(
        pred_oil_map, priority_map_ai, budget=BUDGET, efficiency=0.8
    )
    
    days_ai, history_ai = simulate_recovery_time(
        oil_residue_ai, T_condition=25.0, pH_condition=7.8, target_index=0.9
    )
    print(f" -> Recovery Time: {days_ai} days ({days_ai/365:.1f} years)")

    # ---------------------------------------------------------
    # 결과 비교 및 그래프
    # ---------------------------------------------------------
    time_reduction = (days_actual - days_ai) / days_actual * 100
    
    print("-" * 50)
    print(f"FINAL RESULT: {time_reduction:.1f}% Time Saved with AI")
    
    plt.figure(figsize=(10, 6))
    plt.plot(history_actual, label=f'Actual (2007): {days_actual/365:.1f} Years', color='gray', linestyle='--')
    plt.plot(history_ai, label=f'AI Optimized: {days_ai/365:.1f} Years', color='green', linewidth=2)
    plt.axhline(y=0.9, color='r', linestyle=':', label='Target (0.9)')
    plt.title('Ecological Recovery Speed Comparison')
    plt.xlabel('Days')
    plt.ylabel('Recovery Index')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 그래프 파일 저장
    plt.savefig("taean_comparison_result.png")
    print("\n[Graph] Saved to 'taean_comparison_result.png'")
    
    # 그래프 창 띄우기
    plt.show()

if __name__ == "__main__":
    main()
