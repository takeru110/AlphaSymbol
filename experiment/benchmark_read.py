import time
import random
from pathlib import Path


def benchmark_single_line_read(file_path: str, num_trials: int = 10):
    """Benchmark reading a single random line from a large CSV file."""
    file_size = Path(file_path).stat().st_size
    
    # ファイルの行数を取得
    import subprocess
    result = subprocess.run(['wc', '-l', file_path], capture_output=True, text=True)
    total_lines = int(result.stdout.split()[0])
    
    print(f"File size: {file_size / (1024*1024*1024):.2f} GB")
    print(f"Total lines: {total_lines}")
    print("\nBenchmarking single line reads...")
    
    times = []
    for i in range(num_trials):
        # ランダムな行番号を選択
        target_line = random.randint(0, total_lines - 1)
        
        # 計測開始
        start_time = time.time()
        
        with open(file_path, 'r') as f:
            for j, line in enumerate(f):
                if j == target_line:
                    # 行を読み取ったら即座に終了
                    break
        
        # 計測終了
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)
        
        print(f"Trial {i+1}: {elapsed:.3f} seconds")
    
    avg_time = sum(times) / len(times)
    print(f"\nAverage time per read: {avg_time:.3f} seconds")
    print(f"Lines per second: {1/avg_time:.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark reading a single line from a large CSV file')
    parser.add_argument('--file_path', help='Path to the CSV file')
    parser.add_argument('-n', '--num-trials', type=int, default=10, help='Number of trials (default: 10)')
    args = parser.parse_args()
    
    benchmark_single_line_read(args.file_path, args.num_trials) 