import sys
import os
from unittest.mock import MagicMock

# 필요 없는 모듈 pytest에서 자동으로 mock 처리
sys.modules['boto3'] = MagicMock()

# 현재 파일 기준으로 src 경로를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "../opt/mlops/src"))

import pytest

from utils.utils import auto_increment_run_suffix, model_dir

# -------------------------------
# auto_increment_run_suffix 테스트
# -------------------------------
def test_auto_increment_run_suffix_default_pad():
    assert auto_increment_run_suffix("run-001") == "run-002"
    assert auto_increment_run_suffix("exp-099") == "exp-100"

def test_auto_increment_run_suffix_custom_pad():
    assert auto_increment_run_suffix("test-7", pad=2) == "test-08"
    assert auto_increment_run_suffix("run-5", pad=4) == "run-0006"

# -------------------------------
# model_dir 테스트 (환경 의존 최소화)
# -------------------------------
def test_model_dir_contains_model_name():
    path = model_dir("itemCF")
    # OS별 경로 차이 대비
    assert "models/itemCF" in path.replace("\\", "/")

