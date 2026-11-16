"""
Suite 1: Token, Cost & Cache Regex Robustness.
Validates extraction across all Aider output variants:
- Standard sent/received
- Cache write format
- Cache hit format
- Combined cache write + cache hit
- Sub-cent and multi-decimal cost amounts
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import aider_bridge


def test_standard_output_no_cache():
    sample = "Tokens: 2.5k sent, 150 received. Cost: $0.00076 message, $0.00076 session."
    sent, write, hit, rcvd, cost_m, cost_s = aider_bridge.parse_cost_and_tokens(sample)
    assert sent == "2.5k"
    assert write == "0"
    assert hit == "0"
    assert rcvd == "150"
    assert cost_m == "0.00076"
    assert cost_s == "0.00076"


def test_cache_write_turn_1():
    sample = "Tokens: 3.2k sent, 1.8k cache write, 210 received. Cost: $0.00095 message, $0.00095 session."
    sent, write, hit, rcvd, cost_m, cost_s = aider_bridge.parse_cost_and_tokens(sample)
    assert sent == "3.2k"
    assert write == "1.8k"
    assert hit == "0"
    assert rcvd == "210"
    assert cost_m == "0.00095"


def test_cache_hit_turn_2():
    sample = "Tokens: 4.5k sent, 3.8k cache hit, 110 received. Cost: $0.00018 message, $0.00113 session."
    sent, write, hit, rcvd, cost_m, cost_s = aider_bridge.parse_cost_and_tokens(sample)
    assert sent == "4.5k"
    assert write == "0"
    assert hit == "3.8k"
    assert rcvd == "110"
    assert cost_m == "0.00018"
    assert cost_s == "0.00113"


def test_cache_write_and_hit_combined():
    sample = "Tokens: 5.1k sent, 600 cache write, 3.5k cache hit, 300 received. Cost: $0.00045 message, $0.00158 session."
    sent, write, hit, rcvd, cost_m, cost_s = aider_bridge.parse_cost_and_tokens(sample)
    assert sent == "5.1k"
    assert write == "600"
    assert hit == "3.5k"
    assert rcvd == "300"
    assert cost_m == "0.00045"


def test_sub_cent_and_zero_cost():
    sample = "Tokens: 100 sent, 10 received. Cost: $0.00 message, $0.00003 session."
    sent, write, hit, rcvd, cost_m, cost_s = aider_bridge.parse_cost_and_tokens(sample)
    assert sent == "100"
    assert rcvd == "10"
    assert cost_m == "0.00"
    assert cost_s == "0.00003"
