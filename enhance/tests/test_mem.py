from satenhance_enhance.mem import MIN_BLOCK, choose_block, estimate_block_bytes


def test_estimate_scales_with_channels_and_block():
    a = estimate_block_bytes(4, 512, 32, 4)
    assert estimate_block_bytes(10, 512, 32, 4) == int(a * 10 / 4)
    assert estimate_block_bytes(4, 256, 32, 4) < a


def test_plenty_of_memory_keeps_requested_block():
    assert choose_block(512, 4, 32, 4, avail=64 * 1024**3) == (512, None)


def test_huge_memory_keeps_requested_block():
    block, warn = choose_block(1024, 10, 32, 4, avail=10**15)
    assert (block, warn) == (1024, None)


def test_low_memory_lowers_block_with_warning():
    need_512 = estimate_block_bytes(10, 512, 32, 4)
    block, warn = choose_block(512, 10, 32, 4, avail=int(need_512 / 0.7 * 0.6))
    assert MIN_BLOCK <= block < 512 and "Lowered --block" in warn
    assert estimate_block_bytes(10, block, 32, 4) <= need_512


def test_hopeless_memory_warns_but_returns_minimum():
    block, warn = choose_block(512, 10, 32, 4, avail=1024)
    assert block == MIN_BLOCK and "out-of-memory" in warn
