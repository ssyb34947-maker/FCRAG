"""
Test suite for the RAG system components.
This script runs all unit tests for the RAG system components.
"""

import subprocess
import sys
from pathlib import Path

def run_test(test_file):
    """Run a single test file"""
    print(f"Running {test_file}...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            str(Path(__file__).parent / test_file),
            "-v"
        ], check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Test {test_file} failed:")
        print(e.output)
        return False

def run_all_tests():
    """Run all test files"""
    test_files = [
        "test_tokenizer.py",
        "test_dedup.py",
        "test_loader.py",
        "test_chunking.py",
        "test_embedding.py",
        "test_searcher.py",
        "test_reranker.py"
    ]
    
    results = []
    for test_file in test_files:
        result = run_test(test_file)
        results.append((test_file, result))
    
    # Print summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = 0
    failed = 0
    
    for test_file, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_file}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("-"*50)
    print(f"Total: {len(results)}, Passed: {passed}, Failed: {failed}")
    
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()