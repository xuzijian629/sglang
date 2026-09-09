# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import types
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.allocation import alloc_for_decode
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ReqToTokenPool:
    def __init__(self):
        self.req_to_token = torch.zeros((4, 16), dtype=torch.int32)
        self.alloc_aux_to_lengths = Mock()

    def write(self, indices, values):
        self.req_to_token[indices] = values


class _RecordingAllocator(BaseTokenToKVPoolAllocator):
    def __init__(self):
        self.page_size = 4
        self.calls = []

    def alloc_decode(self, seq_lens, seq_lens_cpu, last_loc):
        self.calls.append((seq_lens, seq_lens_cpu, last_loc))
        return torch.tensor([31, 47], dtype=torch.int64)

    def clear(self):
        pass

    def alloc(self, need_size):
        raise NotImplementedError

    def free(self, free_index):
        raise NotImplementedError


class _FusedAllocator(_RecordingAllocator):
    def alloc_decode_and_write(self, **kwargs):
        self.calls.append(kwargs)
        out_cache_loc = torch.tensor([63, 79], dtype=torch.int64)
        kwargs["req_to_token"][kwargs["req_pool_indices"], kwargs["write_locs"]] = (
            out_cache_loc.to(torch.int32)
        )
        return out_cache_loc


class TestPagedDecodeAllocationDispatch(unittest.TestCase):
    def test_default_path_gathers_allocates_and_writes(self):
        allocator = _RecordingAllocator()
        req_to_token_pool = _ReqToTokenPool()
        req_pool_indices = torch.tensor([2, 1], dtype=torch.int64)
        seq_lens = torch.tensor([3, 5], dtype=torch.int64)
        seq_lens_cpu = seq_lens.clone()
        req_to_token_pool.req_to_token[req_pool_indices, seq_lens - 1] = torch.tensor(
            [17, 23], dtype=torch.int32
        )

        out_cache_loc = allocator.alloc_decode_and_write(
            req_to_token=req_to_token_pool.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            write_locs=seq_lens,
            token_per_req=1,
        )

        torch.testing.assert_close(out_cache_loc, torch.tensor([31, 47]))
        next_seq_lens, next_seq_lens_cpu, last_loc = allocator.calls[0]
        torch.testing.assert_close(next_seq_lens, torch.tensor([4, 6]))
        torch.testing.assert_close(next_seq_lens_cpu, torch.tensor([4, 6]))
        torch.testing.assert_close(last_loc, torch.tensor([17, 23], dtype=torch.int32))
        torch.testing.assert_close(
            req_to_token_pool.req_to_token[req_pool_indices, seq_lens],
            torch.tensor([31, 47], dtype=torch.int32),
        )

    @patch("sglang.srt.mem_cache.allocation.evict_from_tree_cache")
    @patch("sglang.srt.mem_cache.allocation._alloc_page_size", return_value=4)
    def test_alloc_for_decode_uses_allocator_override(self, _page_size, evict):
        allocator = _FusedAllocator()
        req_to_token_pool = _ReqToTokenPool()
        req_pool_indices = torch.tensor([2, 1], dtype=torch.int64)
        seq_lens = torch.tensor([3, 5], dtype=torch.int64)
        reqs = [
            types.SimpleNamespace(
                kv=types.SimpleNamespace(kv_allocated_len=3, kv_committed_len=3)
            ),
            types.SimpleNamespace(
                kv=types.SimpleNamespace(kv_allocated_len=5, kv_committed_len=5)
            ),
        ]
        tree_cache = types.SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            pretty_print=Mock(),
        )
        batch = types.SimpleNamespace(
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            req_pool_indices=req_pool_indices,
            req_pool_indices_cpu=req_pool_indices.clone(),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.clone(),
            model_config=types.SimpleNamespace(is_encoder_decoder=False),
            maybe_evict_swa=Mock(),
            reqs=reqs,
        )

        out_cache_loc = alloc_for_decode(batch, token_per_req=1)

        torch.testing.assert_close(out_cache_loc, torch.tensor([63, 79]))
        self.assertEqual(len(allocator.calls), 1)
        allocator_call = allocator.calls[0]
        self.assertIs(allocator_call["req_to_token"], req_to_token_pool.req_to_token)
        self.assertIs(allocator_call["req_pool_indices"], req_pool_indices)
        self.assertIs(allocator_call["seq_lens"], seq_lens)
        torch.testing.assert_close(
            req_to_token_pool.req_to_token[req_pool_indices, seq_lens],
            torch.tensor([63, 79], dtype=torch.int32),
        )
        batch.maybe_evict_swa.assert_called_once_with()
        evict.assert_called_once_with(tree_cache, 8)
        req_to_token_pool.alloc_aux_to_lengths.assert_called_once()
        self.assertEqual([req.kv.kv_allocated_len for req in reqs], [4, 6])
        self.assertEqual([req.kv.kv_committed_len for req in reqs], [4, 6])


if __name__ == "__main__":
    unittest.main()
