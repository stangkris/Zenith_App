import pandas as pd
import numpy as np
from strategies import run_strategy, enrich_indicators

def create_synthetic_data(n=200):
    # Create an uptrend then pullback
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    df = pd.DataFrame(index=dates)
    
    # Uptrend
    t = np.linspace(0, 20, n)
    price = 100 + t + 5 * np.sin(t)
    
    df["close"] = price
    df["high"] = price + 2
    df["low"] = price - 2
    df["open"] = price - 1
    df["volume"] = np.random.randint(100, 1000, n)
    
    # Add some structure for strategies
    df["close"] = df["close"] + np.random.normal(0, 1, n)
    
    # Enrich
    enrich_indicators(df)
    return df

def test_strategies():
    print("Creating synthetic data...")
    df = create_synthetic_data()
    print("Enriched data columns:", df.columns.tolist())
    
    strategies = [
        "SMC (BOS/OB/FVG)",
        "Momentum Breakout (EMA50 + VCP)",
        "Pullback Reversal (EMA200 + Fib + RSI)",
        "Volume Profile (POC)"
    ]
    
    print("\n--- Testing Strategy Execution ---")
    for s in strategies:
        try:
            print(f"Running {s}...", end=" ")
            res = run_strategy(df, s)
            print("OK")
            print(f"  Result keys: {list(res.keys())}")
            print(f"  TP: {res.get('tp')}, Entry: {res.get('entry')}, Risk: {res.get('entry') - res.get('sl'):.4f}")
            if res.get('tp') == res.get('entry') + 2 * (res.get('entry') - res.get('sl')):
                 print("  Note: TP is exactly 2R (Fallback or Fixed)")
            else:
                 print("  Note: TP is Structure-Based (Varied)")
                 
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_strategies()
